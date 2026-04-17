"""
sac_agent.py — Soft Actor-Critic (SAC) agent for BESS joint-market bidding.

Implements Section IV-C of Li et al. (2024):
  - Actor network   : MLP with tanh-squashed Gaussian policy
  - Critic networks : Clipped double-Q (Q1, Q2) + soft-updated target networks
  - Automatic entropy tuning via learnable log_alpha
  - Gradient clipping (max_norm=1.0) on all networks

Observation vector layout (72-dim, ERCOT-correct):
    obs = [SoC(1) | prices(5) | ttfe_feature(64) | hour_sin_cos(2)]

Action space (8-dim, ERCOT-correct):
    [v_dch, v_ch, a_spot_dch, a_spot_ch, a_regup, a_regdn, a_rrs, a_nsrs]
    Continuous in [-1, 1] (tanh squashed).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    HIDDEN_DIM, NUM_HIDDEN_LAYERS,
    LR_POLICY, LR_Q,
    GAMMA, TAU_TARGET, ALPHA_ENTROPY, TARGET_ENTROPY,
    BATCH_SIZE, STATE_DIM, ACTION_DIM,
)

# Fixed observation and action dimensions (ERCOT-correct)
OBS_DIM = STATE_DIM    # 72: SoC(1) + prices(5) + TTFE(64) + hour_sin_cos(2)
ACT_DIM = ACTION_DIM   # 8:  v_dch/v_ch + spot_dch/ch + regup/regdn + rrs + nsrs

# Small constant for numerical stability in log computations
LOG_STD_MIN = -20
LOG_STD_MAX = 2


# ─── MLP utility ──────────────────────────────────────────────────────────────

def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    num_hidden: int,
    out_dim: int,
    activation: nn.Module = None,
) -> nn.Sequential:
    """
    Build a fully-connected network:
        Linear → ReLU → [Linear → ReLU] * (num_hidden-1) → Linear → (optional activation)
    """
    if activation is None:
        activation = nn.Identity()

    layers = []
    current_dim = in_dim
    for _ in range(num_hidden):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim))
    layers.append(activation)
    return nn.Sequential(*layers)


# ─── Actor ────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """
    Gaussian policy with tanh squashing.

    Architecture: MLP(obs_dim → HIDDEN_DIM → HIDDEN_DIM → 2*act_dim)
    Outputs (mean, log_std) for a diagonal Gaussian.
    Actions are sampled via the reparameterization trick and squashed
    through tanh to lie in [-1, 1]^act_dim.

    Log-probability accounts for the tanh Jacobian correction (SAC appendix):
        log π(a|s) = Σ [ log N(u|μ,σ) - log(1 - tanh²(u)) ]
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_hidden: int = NUM_HIDDEN_LAYERS,
    ):
        super().__init__()
        self.act_dim = act_dim

        # Shared trunk
        trunk_layers = []
        current_dim = obs_dim
        for _ in range(num_hidden):
            trunk_layers.append(nn.Linear(current_dim, hidden_dim))
            trunk_layers.append(nn.ReLU())
            current_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)

        # Two separate heads for mean and log_std
        self.mean_head    = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(
        self,
        obs: torch.Tensor,
    ):
        """
        Args:
            obs: (batch, obs_dim)

        Returns:
            mean    : (batch, act_dim)  — pre-tanh mean
            log_std : (batch, act_dim)  — clamped log standard deviation
        """
        h = self.trunk(obs)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ):
        """
        Sample action using reparameterization trick.

        Args:
            obs:           (batch, obs_dim)
            deterministic: if True, return tanh(mean) without noise

        Returns:
            action  : (batch, act_dim)  in [-1, 1]
            log_pi  : (batch, 1)        log probability of sampled action
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        if deterministic:
            # Greedy action — used for evaluation
            action = torch.tanh(mean)
            log_pi = torch.zeros(obs.shape[0], 1, device=obs.device)
            return action, log_pi

        # Reparameterization: u = mean + std * epsilon,  epsilon ~ N(0,I)
        dist = torch.distributions.Normal(mean, std)
        u    = dist.rsample()                           # (batch, act_dim)

        # Tanh squash
        action = torch.tanh(u)

        # Log probability with Jacobian correction for tanh squashing
        # log π(a|s) = log N(u; μ, σ) - Σ log(1 - tanh²(u))
        log_pi_gaussian = dist.log_prob(u)              # (batch, act_dim)
        # Numerically stable version: log(1 - tanh²(u)) = 2*(log2 - softplus(-2u) - u)
        log_jacobian = torch.log(1.0 - action.pow(2) + 1e-6)
        log_pi = (log_pi_gaussian - log_jacobian).sum(dim=-1, keepdim=True)  # (batch, 1)

        return action, log_pi

    def get_mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation: tanh(mean), no noise."""
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


# ─── Critic ───────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """
    Single Q-network: MLP(obs_dim + act_dim → HIDDEN_DIM → HIDDEN_DIM → 1).
    """

    def __init__(
        self,
        obs_dim:    int = OBS_DIM,
        act_dim:    int = ACT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_hidden: int = NUM_HIDDEN_LAYERS,
    ):
        super().__init__()
        self.net = _build_mlp(
            in_dim=obs_dim + act_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            out_dim=1,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs    : (batch, obs_dim)
            action : (batch, act_dim)

        Returns:
            q : (batch, 1)
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


# ─── SAC Agent ────────────────────────────────────────────────────────────────

class SACAgent:
    """
    Soft Actor-Critic with automatic entropy tuning.

    Components:
        actor          — Gaussian policy
        critic1/2      — clipped double-Q networks
        critic1_target /
        critic2_target — soft-updated copies of critics
        log_alpha      — learnable temperature parameter

    All networks live on `device`.
    """

    def __init__(
        self,
        obs_dim:    int           = OBS_DIM,
        act_dim:    int           = ACT_DIM,
        hidden_dim: int           = HIDDEN_DIM,
        num_hidden: int           = NUM_HIDDEN_LAYERS,
        lr_policy:  float         = LR_POLICY,
        lr_q:       float         = LR_Q,
        gamma:      float         = GAMMA,
        tau:        float         = TAU_TARGET,
        alpha_init: float         = ALPHA_ENTROPY,
        device:     torch.device  = torch.device("cpu"),
    ):
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.gamma      = gamma
        self.tau        = tau
        self.device     = device

        # ── Networks ──────────────────────────────────────────────────────────
        self.actor = Actor(obs_dim, act_dim, hidden_dim, num_hidden).to(device)

        self.critic1 = Critic(obs_dim, act_dim, hidden_dim, num_hidden).to(device)
        self.critic2 = Critic(obs_dim, act_dim, hidden_dim, num_hidden).to(device)

        # Target critics — initialized as hard copies; updated via soft update
        self.critic1_target = Critic(obs_dim, act_dim, hidden_dim, num_hidden).to(device)
        self.critic2_target = Critic(obs_dim, act_dim, hidden_dim, num_hidden).to(device)
        self._hard_update_targets()

        # Target networks are never directly trained — no grad needed
        for p in self.critic1_target.parameters():
            p.requires_grad_(False)
        for p in self.critic2_target.parameters():
            p.requires_grad_(False)

        # ── Entropy tuning ────────────────────────────────────────────────────
        # TARGET_ENTROPY = -ACTION_DIM * 0.5 = -4.0  (tighter than default)
        self.target_entropy = TARGET_ENTROPY
        self.log_alpha = torch.tensor(
            np.log(alpha_init), dtype=torch.float32,
            device=device, requires_grad=True
        )

        # ── Optimizers ────────────────────────────────────────────────────────
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),   lr=lr_policy)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_q)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_q)
        self.alpha_optimizer  = torch.optim.Adam([self.log_alpha],           lr=lr_policy)

    # ─── Action selection ─────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select action for a single observation (inference time).

        Args:
            obs:           (obs_dim,) numpy array
            deterministic: if True, use tanh(mean) — for evaluation

        Returns:
            action: (act_dim,) numpy array in [-1, 1]
        """
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        self.actor.eval()
        action, _ = self.actor.sample(obs_t, deterministic=deterministic)
        self.actor.train()
        return action.squeeze(0).cpu().numpy()

    # ─── SAC update ───────────────────────────────────────────────────────────

    def update(self, batch: dict) -> dict:
        """
        Perform one SAC gradient update step.

        Args:
            batch: dict from ReplayBuffer.sample() — all tensors on device
                keys: obs, actions, rewards, next_obs, dones

        Returns:
            dict with scalar losses:
                q1_loss, q2_loss, actor_loss, alpha_loss, alpha (current value)
        """
        obs      = batch["obs"]        # (B, obs_dim)
        actions  = batch["actions"]    # (B, act_dim)
        rewards  = batch["rewards"]    # (B, 1)
        next_obs = batch["next_obs"]   # (B, obs_dim)
        dones    = batch["dones"]      # (B, 1)

        alpha = self.log_alpha.exp().detach()

        # ── Critic update ─────────────────────────────────────────────────────

        with torch.no_grad():
            # Sample next action from current policy
            next_actions, next_log_pi = self.actor.sample(next_obs)

            # Clipped double-Q target
            q1_next = self.critic1_target(next_obs, next_actions)
            q2_next = self.critic2_target(next_obs, next_actions)
            min_q_next = torch.min(q1_next, q2_next)

            # Bellman target: r + γ*(1-done)*(min_Q_target - α*log_π)
            q_target = rewards + self.gamma * (1.0 - dones) * (
                min_q_next - alpha * next_log_pi
            )

        # Q1 loss
        q1_pred = self.critic1(obs, actions)
        q1_loss = F.mse_loss(q1_pred, q_target)

        self.critic1_optimizer.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        self.critic1_optimizer.step()

        # Q2 loss
        q2_pred = self.critic2(obs, actions)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.critic2_optimizer.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic2_optimizer.step()

        # ── Actor update ──────────────────────────────────────────────────────

        # Sample actions from current policy (with reparameterization)
        pi_actions, log_pi = self.actor.sample(obs)

        # Actor loss: α*log_π - min_Q  (maximize Q, minimize entropy term)
        q1_pi = self.critic1(obs, pi_actions)
        q2_pi = self.critic2(obs, pi_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (alpha * log_pi - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # ── Alpha (entropy temperature) update ───────────────────────────────

        # Loss: -log_alpha * (log_pi + target_entropy)
        # Detach log_pi so only log_alpha is differentiated
        alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
        self.alpha_optimizer.step()

        # ── Soft update target critics ────────────────────────────────────────
        self._soft_update_targets()

        return {
            "q1_loss"    : q1_loss.item(),
            "q2_loss"    : q2_loss.item(),
            "actor_loss" : actor_loss.item(),
            "alpha_loss" : alpha_loss.item(),
            "alpha"      : self.log_alpha.exp().item(),
        }

    # ─── Target network updates ───────────────────────────────────────────────

    def _hard_update_targets(self) -> None:
        """Copy critic weights to target networks (called once at init)."""
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def _soft_update_targets(self) -> None:
        """
        Exponential moving average update of target networks:
            θ_target ← τ * θ + (1 - τ) * θ_target
        """
        tau = self.tau
        for target_param, param in zip(
            self.critic1_target.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(
            self.critic2_target.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save all network weights and optimizer states to a .pt file.

        Args:
            path: full file path (e.g. "outputs/checkpoints/sac_ep100.pt")
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor_state"            : self.actor.state_dict(),
                "critic1_state"          : self.critic1.state_dict(),
                "critic2_state"          : self.critic2.state_dict(),
                "critic1_target_state"   : self.critic1_target.state_dict(),
                "critic2_target_state"   : self.critic2_target.state_dict(),
                "log_alpha"              : self.log_alpha.detach().cpu(),
                "actor_opt_state"        : self.actor_optimizer.state_dict(),
                "critic1_opt_state"      : self.critic1_optimizer.state_dict(),
                "critic2_opt_state"      : self.critic2_optimizer.state_dict(),
                "alpha_opt_state"        : self.alpha_optimizer.state_dict(),
            },
            path,
        )
        print(f"[SACAgent] Saved checkpoint → {path}")

    def load(self, path: str) -> None:
        """
        Load all network weights and optimizer states from a .pt file.

        Args:
            path: full file path to an existing checkpoint
        """
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state"])
        self.critic1.load_state_dict(ckpt["critic1_state"])
        self.critic2.load_state_dict(ckpt["critic2_state"])
        self.critic1_target.load_state_dict(ckpt["critic1_target_state"])
        self.critic2_target.load_state_dict(ckpt["critic2_target_state"])
        self.log_alpha.data = ckpt["log_alpha"].to(self.device)
        self.actor_optimizer.load_state_dict(ckpt["actor_opt_state"])
        self.critic1_optimizer.load_state_dict(ckpt["critic1_opt_state"])
        self.critic2_optimizer.load_state_dict(ckpt["critic2_opt_state"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_opt_state"])
        print(f"[SACAgent] Loaded checkpoint ← {path}")
