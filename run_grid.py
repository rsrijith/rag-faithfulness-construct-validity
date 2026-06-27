"""Scale runner with bootstrap CIs: score base vs perturbed over N real items, report per-operator mean delta
with a 95% bootstrap CI and a CI-based verdict (so 'drop'/'flat' is statistical, not a point-threshold).

Usage: python run_grid.py <metric> <dataset> [n]
  metric : bertscore | hhem | summac
  dataset: vitaminc | hotpot
"""

import sys
import numpy as np
from battery import operators as ops
from data.load import LOADERS

EPS = 0.05           # meaningful-movement band on the [0,1] score scale
N_BOOT = 2000


def metric_by_name(name):
    if name == "bertscore":
        from metrics.bertscore_metric import BertScoreMetric; return BertScoreMetric()
    if name == "hhem":
        from metrics.hhem_metric import HHEMMetric; return HHEMMetric()
    if name == "summac":
        from metrics.summac_metric import SummaCMetric; return SummaCMetric()
    raise SystemExit(f"unknown metric {name}")


def boot_ci(deltas, rng):
    d = np.array(deltas)
    means = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(N_BOOT)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(expected, lo, hi):
    if expected == "drop":                      # valid metric must drop meaningfully
        if hi < -EPS: return "CATCHES (valid)"   # whole CI below -EPS
        if lo >= -EPS: return "BLIND (invalid)"  # CI never reaches a meaningful drop
        return "weak/ambiguous"                  # CI straddles -EPS
    else:                                        # invariance: must stay flat
        if lo > EPS or hi < -EPS:                # whole CI outside the flat band = significant move
            return "CONFOUNDED (invalid)"
        return "STABLE (valid)"                  # CI overlaps 0 / within band


def main():
    mname = sys.argv[1] if len(sys.argv) > 1 else "hhem"
    dname = sys.argv[2] if len(sys.argv) > 2 else "vitaminc"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    items = LOADERS[dname](n=n)
    metric = metric_by_name(mname)
    rng = np.random.default_rng(0)
    base = {it["id"]: metric.score(it) for it in items}
    print(f"\n### {metric.name} on {dname} (n={len(items)})   [EPS={EPS}, {N_BOOT} bootstrap]")
    for op in ops.RULE_BASED:
        if op.__name__ == "s3_citation_relocation" and not metric.citation_aware:
            print(f"  {op.__name__:24s}  N/A (metric not citation-aware)"); continue
        deltas, exp = [], None
        for it in items:
            res = op(it); exp = res["expected"]
            if not res["applied"]:
                continue
            deltas.append(metric.score(res["item"]) - base[it["id"]])
        if len(deltas) < 5:
            print(f"  {op.__name__:24s}  (fired on {len(deltas)} items, too few)"); continue
        m, lo, hi = boot_ci(deltas, rng)
        print(f"  {op.__name__:24s}  n={len(deltas):3d}  delta={m:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  -> {verdict(exp, lo, hi)}")


if __name__ == "__main__":
    main()
