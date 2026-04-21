import sys, numpy as np
sys.path.insert(0, 'src')
from data_loader_stage2 import load_stage2_data
from data_bridge_stage3 import make_stage3_splits
from dah_baseline import DAHBaseline
from config_stage3 import DAM_MCPC_PRIOR

raw  = load_stage2_data()
data = make_stage3_splits(raw)

rt_arr    = data['train']['prices']['rt_lmp']
disch_thr = float(np.nanmean(rt_arr) + 0.5 * np.nanstd(rt_arr))
charg_thr = float(np.nanmean(rt_arr) - 0.5 * np.nanstd(rt_arr))

val_prices  = data['val']['prices']
val_syscond = data['val']['syscond']
val_days    = data['val']['days']
dah = DAHBaseline(val_prices, val_syscond, disch_thr, charg_thr)

print("=== DAH Val Breakdown (per day) ===")
print("Thresholds: discharge=$%.2f  charge=$%.2f" % (disch_thr, charg_thr))
print()
print("%-4s  %-10s  %-9s  %-9s  %-9s  %-6s" % ("Day","DAH_total","spot","AS","degrad","cycles"))
print("-"*58)

totals, spots, ass_, degrds, cycs = [], [], [], [], []
for i, ep_start in enumerate(val_days):
    r = dah.run_episode(ep_start)
    totals.append(r['total_rev'])
    spots.append(r['rev_spot'])
    ass_.append(r['rev_as'])
    degrds.append(r['rev_degrad'])
    cycs.append(r['cycles'])
    print("%-4d  $%-9.2f  $%-8.2f  $%-8.2f  $%-8.2f  %.1f" % (
        i+1, r['total_rev'], r['rev_spot'], r['rev_as'], r['rev_degrad'], r['cycles']))

print("-"*58)
print("%-4s  $%-9.2f  $%-8.2f  $%-8.2f  $%-8.2f  %.1f" % (
    "mean", np.mean(totals), np.mean(spots), np.mean(ass_), np.mean(degrds), np.mean(cycs)))
print("%-4s  $%-9.2f  $%-8.2f  $%-8.2f  $%-8.2f" % (
    "std", np.std(totals), np.std(spots), np.std(ass_), np.std(degrds)))

print()
print("=== Price Environment (Val Set) ===")
rt_val = val_prices['rt_lmp']
print("RT LMP:  mean=$%.2f  std=$%.2f  max=$%.2f  min=$%.2f  p95=$%.2f" % (
    np.nanmean(rt_val), np.nanstd(rt_val), np.nanmax(rt_val), np.nanmin(rt_val),
    np.nanpercentile(rt_val, 95)))
print()

print("DART spread analysis (DAM prior vs actual RT MCPC in val set):")
print("%-8s  %-10s  %-12s  %-14s  %-12s" % (
    "Product", "DAM_prior", "RT_mean", "DART_spread", "AS_rev/day"))
print("-"*62)
for prod in ['regup','regdn','rrs','ecrs','nsrs']:
    arr  = val_prices["rt_mcpc_" + prod]
    dam  = DAM_MCPC_PRIOR[prod]
    rt_m = float(np.nanmean(arr))
    dart = dam - rt_m
    # DAH bids 1MW for 24h, DART revenue per day = 1 MW * dart * 24h
    rev  = dart * 24.0
    print("%-8s  $%-9.2f  $%-11.3f  $%-13.3f  $%-11.2f" % (
        prod, dam, rt_m, dart, rev))

print()
print("=== TempDRL vs DAH — Training Run Summary ===")
print("(from current run, TTFE frozen, best checkpoint at ep 250 = $147.68)")
print()
eps  = [50,100,150,200,250,300,350,400,450,500,550,600,650,700]
vals = [113.02,137.83,124.48,140.56,147.68,116.50,146.22,108.46,135.91,121.68,140.92,121.90,119.95,125.23]
dah_v = 223.95

print("%-5s  %-10s  %-10s  %-8s  %-10s" % ("Ep","TempDRL","DAH","Ratio","Gap"))
print("-"*48)
for ep, v in zip(eps, vals):
    gap = dah_v - v
    print("%-5d  $%-9.2f  $%-9.2f  %-7.1f%%  $%-9.2f" % (ep, v, dah_v, v/dah_v*100, gap))

print()
print("Current gap to close: $%.2f/day (DAH=$%.2f, best_TempDRL=$147.68)" % (dah_v - 147.68, dah_v))
print("Gap as %% of DAH: %.1f%%" % ((dah_v - 147.68)/dah_v*100))
print()
print("=== Why DAH is hard to beat ===")
total_dart = sum([(DAM_MCPC_PRIOR[p] - float(np.nanmean(val_prices["rt_mcpc_"+p])))*24 for p in ['regup','regdn','rrs','ecrs','nsrs']])
print("DAH AS revenue entirely from DART spread: ~$%.2f/day (5 products x 1MW x 24h)" % total_dart)
print("DAH spot revenue: ~$%.2f/day (greedy LMP arbitrage)" % np.mean(spots))
print("DAH degradation: ~$%.2f/day" % np.mean(degrds))
print("TempDRL must learn to: (a) match DAH DART capture and (b) beat spot arbitrage.")
