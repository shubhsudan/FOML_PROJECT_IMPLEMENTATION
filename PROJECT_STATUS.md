# TempDRL — Project Status Handoff

## What This Project Is

Implementation of **TempDRL** (Temporal-Aware Deep Reinforcement Learning) for BESS (Battery Energy Storage System) joint-market bidding, based on the paper by Li et al. (2024).

The agent learns to bid a 10 MWh / 2 MW BESS across **7 simultaneous markets** (spot + 6 FCAS ancillary services) using SAC + a Transformer-based Temporal Feature Extractor (TTFE).

**Server:** Narnia GPU cluster, RIT — `ss2401@narnia.gccis.rit.edu`, GPU index 26 (NVIDIA A16, 15 GB)
**Conda env:** `tempdrl` — activate with `~/miniconda3/envs/tempdrl/bin/python`
**Project dir on Narnia:** `~/tempdrl/`
**Project dir locally:** `/Users/shubhsudan/Desktop/RIT/SEM 2/FOML/PROJECT/IMPLEMENTATION/`

---

## Codebase Structure

```
IMPLEMENTATION/
├── main_phase2.py          # Entry point — SAC training
├── src/
│   ├── config.py           # All hyperparameters
│   ├── data_loader.py      # Parquet loader, episode splitter
│   ├── environment.py      # BESS MDP (7 markets, 288 steps/day)
│   ├── ttfe.py             # Transformer feature extractor
│   ├── sac_agent.py        # SAC: Actor, Critic, update()
│   ├── replay_buffer.py    # Circular replay buffer
│   └── trainer.py          # Training loop, run_episode(), evaluate()
└── outputs/
    ├── checkpoints/
    │   ├── sac_final.pt        ← end of training
    │   ├── best_model.pt       ← BEST (eval reward 1388.5 @ ep 17100)
    │   └── sac_ep*.pt          ← milestone snapshots every eval_every eps
    └── logs/
        ├── train_log.csv       ← per-episode: reward, violations, alpha, losses
        └── eval_log.csv        ← per-eval: mean_reward, std_reward, mean_violations
```

---

## Key Architecture Details

### State Space (72-dim)
`[SoC, spot, FR, FL, SR, SL, DR, DL, f_1...f_64]`
- 1 SoC + 7 prices + 64-dim TTFE embedding

### Action Space (6-dim, tanh output)
`[v_dch, v_ch, a_spot, a_fast, a_slow, a_delay]`
- Binary charge/discharge flags + 4 bid sizes

### TTFE (Temporal Feature Extractor)
- Input: L=12 timestep window of scaled prices → (12, 7)
- 2× MHA layers, 4 heads, EMBED_DIM=64, global avg pooling
- Runs in **batched mode**: all 288 steps pre-computed in one `(288, 12, 7)` forward pass per episode

### SAC Hyperparameters (current)
| Param | Value |
|---|---|
| Hidden dim | 512, 2 layers |
| LR (all) | **5e-5** (halved from 1e-4 after manual guard at ep 16,813) |
| Gamma | 0.99 |
| Tau target | 0.005 |
| Alpha entropy | 0.05 (init), auto-tuned |
| Target entropy | `-act_dim * 0.5 = -3.0` |
| Batch size | 1024 |
| Buffer size | 100,000 |
| Grad steps/ep | 72 |

---

## Training Run Summary

### Phase 2 — SAC Training (COMPLETE)

| Phase | Episodes | Notes |
|---|---|---|
| Initial run | ep 1–642 | Baseline, pre-fixes |
| Post Fix B/C | ep 643–9838 | Alpha reset, grad clip, tighter target entropy |
| Speed fixes + resume | ep 9839–16813 | Batched TTFE, batch=1024, grad_steps=72 |
| **Manual guard reload** | **ep 16,814** | Alpha drifted 0.05→0.70, reward collapsed to 236. Reloaded `best_model.pt`, reset alpha=0.05, halved LRs to 5e-5 |
| Final stretch | ep 16,814–30,000 | Stable training, finished cleanly |

**Total env steps: 3,797,856**
**Training time: ~8.5 hours (after speed fixes)**

### Key Results

| Checkpoint | Eval Reward | Violations | Notes |
|---|---|---|---|
| ep 9,850 | 830.6 | 7.16 | First good checkpoint |
| ep 13,300 | 1,086 | 1.63 | Pre-collapse best |
| **ep 17,100** | **1,388.5** | **0.81** | **ALL-TIME BEST → `best_model.pt`** |
| ep 24,800 | 1,011 | 0.77 | Stable convergence |
| ep 28,700 | 1,222 | 0.51 | Late peak |
| ep 30,000 | 1,071 | 0.30 | Final |

### Eval Reward Trajectory (key moments)
```
ep 9850:   830   ← first stable checkpoint
ep 13300: 1086   ← climbing
ep 17100: 1388   ← ALL-TIME BEST (best_model.pt)
ep 16800:  236   ← collapse (alpha drift, before guard)
ep 16950: 1376   ← immediate recovery after manual reload
ep 26200: 1110   ← violations hit 0.09 (best ever)
ep 28700: 1222   ← late surge
ep 30000: 1071   ← final
```

---

## Fixes Applied (Chronological)

### Round 1 — Instability Fixes
- `ALPHA_ENTROPY`: 0.2 → 0.05
- `LR_*`: 3e-4 → 1e-4
- `clip_grad_norm_` added (max_norm=1.0) to all optimizers incl. log_alpha
- `target_entropy`: `-act_dim` → `-act_dim * 0.5 = -3.0`
- Reward clipping: `np.clip(reward, -500, 500)` before buffer push

### Round 2 — Fix B/C
- **Fix B**: On resume, reset `log_alpha = log(0.05)` and recreate `alpha_optimizer`
- **Fix C**: Divergence guard requires `best_eval_reward > 0` before reloading

### Speed Fixes
- **Fix 1**: Batched TTFE — all 288 features in one `(288,12,7)` forward pass
- **Fix 2**: `BATCH_SIZE` 256 → 1024
- **Fix 3**: `grad_steps_per_ep` 288 → 72
- Result: 9s/ep → **2.5s/ep** (3.6× speedup), GPU 14% → 29%

### Manual Guard (ep 16,813)
- Alpha drifted to 0.70, eval reward collapsed to 236, violations ~10
- Reloaded `best_model.pt`, reset alpha=0.05, **halved all LRs to 5e-5**
- Recovery was immediate: reward jumped to 1,376 within 150 episodes

---

## Current State

**Training: COMPLETE ✓**

Best checkpoint to use: **`~/tempdrl/outputs/checkpoints/best_model.pt`**
- Eval reward: **1,388.5**
- Mean violations: **0.81 / 288 steps**
- Saved at episode 17,100

All logs available:
- `outputs/logs/train_log.csv` — 30,000 rows of episode metrics
- `outputs/logs/eval_log.csv` — ~600 rows of eval metrics (every 50 eps)

---

## What Remains / Next Steps

1. **Pull results from Narnia** — download `best_model.pt`, `sac_final.pt`, both CSV logs
2. **Evaluation / analysis** — run final greedy rollouts on eval set, plot reward curves
3. **Baseline comparison** — compare against spot-only / FCAS-only / random policy
4. **Paper results** — Table II equivalent: mean daily revenue, SoC constraint satisfaction rate
5. **Ablation** (optional) — TTFE vs no-TTFE, joint vs single-market

### To pull logs locally:
```bash
sshpass -p 'Shubh@260713' scp -r ss2401@narnia.gccis.rit.edu:~/tempdrl/outputs/ \
  "/Users/shubhsudan/Desktop/RIT/SEM 2/FOML/PROJECT/IMPLEMENTATION/outputs/"
```

### To re-run evaluation on best model:
Modify `main_phase2.py` to set `NUM_EPISODES=0` and run a standalone eval, or write a separate `evaluate.py` that loads `best_model.pt` and runs `evaluate()` from `trainer.py`.

---

## Known Issues / Watch-Outs

- **Alpha drift**: Auto-entropy tuning causes alpha to creep up over time (0.05 → ~1.0). The halved LRs slow this down but don't prevent it. For future runs, consider fixing alpha or adding a hard upper-bound clip.
- **GPU utilization**: Only 29% even after speed fixes. The 288-step episode rollout (CPU-bound data shuffling + env stepping) is the remaining bottleneck. Vectorized environments would push this further.
- **nvidia-smi query**: GPU index query `--query-gpu=26,...` is invalid syntax on Narnia — use `| grep "^26,"` filter instead.
- **SSH control master**: Narnia SSH sometimes serves cached results via control socket. Use `-o ControlPath=none -o ControlMaster=no` for fresh connections.
- **`engine="fastparquet"`**: Required in `pd.read_parquet()` due to pyarrow 19 incompatibility with the data files.
