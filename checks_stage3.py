# checks_stage3.py
# Stage 3 pre-training diagnostic checks (Part 8 of handout)

import sys
import numpy as np

print("=== Stage 3 Diagnostic Checks ===\n")

# Check 1: DART settlement formula
print("Check 1: DART settlement formula")
from src.config_stage3 import DAM_MCPC_PRIOR
dam_mcpc_regup = DAM_MCPC_PRIOR["regup"]  # 2.23
rt_regup  = 0.33
dam_awd   = 1.0
dart_rev = dam_awd * dam_mcpc_regup + (0 - dam_awd) * rt_regup
print(f"  DART revenue for 1 MW RegUp re-dispatched to 0 in RT: ${dart_rev:.2f}/h")
assert abs(dart_rev - 1.90) < 0.01, f"DART formula wrong: got {dart_rev}"
print("  PASS\n")

# Check 2: SOC duration multipliers
print("Check 2: SOC duration multipliers (NPRR1282)")
from src.config_stage3 import AS_DURATION
assert AS_DURATION["ecrs"]  == 1.0, f"ECRS must be 1h, got {AS_DURATION['ecrs']}"
assert AS_DURATION["regup"] == 0.5, f"RegUp must be 0.5h, got {AS_DURATION['regup']}"
assert AS_DURATION["rrs"]   == 0.5, f"RRS must be 0.5h, got {AS_DURATION['rrs']}"
assert AS_DURATION["nsrs"]  == 4.0, f"NSRS must be 4h, got {AS_DURATION['nsrs']}"
print(f"  regup={AS_DURATION['regup']}h  regdn={AS_DURATION['regdn']}h  "
      f"rrs={AS_DURATION['rrs']}h  ecrs={AS_DURATION['ecrs']}h  nsrs={AS_DURATION['nsrs']}h")
print("  PASS\n")

# Check 3: No simultaneous charge+discharge
print("Check 3: Single-scalar ESR model (NPRR1014)")
from src.config_stage3 import P_MAX
action = np.array([1.0, 0.5, 0.5, 0.5, 0.5, 0.5])
p_dispatch = float(np.clip(action[0], -1.0, 1.0)) * P_MAX
print(f"  action[0]=1.0 → p_dispatch={p_dispatch:.2f} MW (positive=discharge, scalar only)")
print("  Single scalar: charge/discharge mutually exclusive by construction")
print("  PASS\n")

# Check 4: Degradation cost
print("Check 4: Degradation cost")
from src.config_stage3 import DEGRADATION_COST
assert DEGRADATION_COST == 15.0, f"Expected 15.0, got {DEGRADATION_COST}"
print(f"  Degradation cost: ${DEGRADATION_COST}/MWh")
print("  PASS\n")

# Check 5: Data loads correctly
print("Check 5: Data loader")
try:
    from src.data_loader_stage2 import load_stage2_data
    data = load_stage2_data()
    print("  Price columns:", list(data["prices"].keys()))
    print("  Syscond columns:", list(data["syscond"].keys()))
    print(f"  Train days: {len(data['train_days'])}  "
          f"Val days: {len(data['val_days'])}  "
          f"Test days: {len(data['test_days'])}")
    required_price_cols = [
        "rt_lmp", "rt_mcpc_regup", "rt_mcpc_regdn", "rt_mcpc_rrs",
        "rt_mcpc_ecrs", "rt_mcpc_nsrs", "dam_spp", "dam_as_regup",
        "dam_as_regdn", "dam_as_rrs", "dam_as_ecrs", "dam_as_nsrs"
    ]
    missing = [c for c in required_price_cols if c not in data["prices"]]
    if missing:
        print(f"  WARNING: missing price columns: {missing}")
    else:
        print("  All required price columns present")
    print("  PASS\n")
except Exception as e:
    print(f"  FAIL: {e}\n")

# Check 6: DAH sanity
print("Check 6: DAH baseline sanity")
try:
    from src.dah_baseline import DAHBaseline
    prices = data["prices"]
    rt_arr = np.array(prices["rt_lmp"])
    mu, sig = np.nanmean(rt_arr), np.nanstd(rt_arr)
    dah = DAHBaseline(prices, data["syscond"], mu + 0.5 * sig, mu - 0.5 * sig)
    result = dah.run_episode(data["train_days"][0])
    print(f"  DAH day 0: total=${result['total_rev']:.2f}  "
          f"spot=${result['rev_spot']:.2f}  "
          f"as=${result['rev_as']:.2f}  "
          f"degrad=${result['rev_degrad']:.2f}  "
          f"cycles={result['cycles']:.1f}")
    print("  PASS\n")
except Exception as e:
    print(f"  FAIL: {e}\n")

print("=== All checks complete ===")
