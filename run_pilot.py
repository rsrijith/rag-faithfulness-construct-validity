"""Pilot runner: score base vs. perturbed items with one or more metrics, check against committed predictions.

Usage: python run_pilot.py
Prints, per (metric, operator): mean base score, mean perturbed score, mean delta, and whether the OBSERVED
behavior matches the committed prediction (drop for sensitivity, flat for invariance). This is the end-to-end
proof that the pipeline (operator -> perturbed item -> metric -> delta) works on real metrics.
"""

import json
import os
import sys
from battery import operators as ops

# touch=flat threshold (on the [0,1] scale) — below this |delta| we call it "flat"
EPS = 0.03


def load_items(path):
    return json.load(open(path))


def behavior(expected, delta):
    """Did the metric behave as a VALID metric should? (not whether our prediction was right — that's compared
    separately). For sensitivity: a valid metric drops (delta <= -EPS). For invariance: stays flat (|delta|<EPS)."""
    if expected == "drop":
        return "DROPS (valid)" if delta <= -EPS else "BLIND (invalid)"
    else:  # flat
        return "STABLE (valid)" if abs(delta) < EPS else "CONFOUNDED (invalid)"


def run(metric, items):
    print(f"\n##### metric: {metric.name}  (method={metric.method}, citation_aware={metric.citation_aware})")
    rows = []
    base_scores = {}
    for item in items:
        base_scores[item["id"]] = metric.score(item)
    for op in ops.RULE_BASED:
        # citation-relocation only matters to citation-aware metrics; others are N/A (answer text unchanged)
        if op.__name__ == "s3_citation_relocation" and not metric.citation_aware:
            print(f"  {op.__name__:24s}  N/A (metric not citation-aware)")
            continue
        deltas = []
        for item in items:
            res = op(item)
            if not res["applied"]:
                continue
            b = base_scores[item["id"]]
            p = metric.score(res["item"])
            deltas.append(p - b)
        if not deltas:
            print(f"  {op.__name__:24s}  (did not fire on any item)")
            continue
        mean_d = sum(deltas) / len(deltas)
        # expected is the same across items for an operator
        exp = ops.RULE_BASED and op(items[0])["expected"]
        print(f"  {op.__name__:24s}  n={len(deltas):2d}  mean_delta={mean_d:+.3f}  ->  {behavior(exp, mean_d)}  (a valid metric must {'DROP' if exp=='drop' else 'STAY FLAT'})")


def main():
    here = os.path.dirname(__file__)
    items = load_items(os.path.join(here, "battery", "sample_items.json"))
    metric_name = sys.argv[1] if len(sys.argv) > 1 else "bertscore"
    if metric_name == "bertscore":
        from metrics.bertscore_metric import BertScoreMetric
        metric = BertScoreMetric()
    else:
        raise SystemExit(f"unknown metric '{metric_name}'")
    run(metric, items)
    print("\n(note: 'BLIND'/'CONFOUNDED' = the metric itself is invalid on that probe; compare to predictions.md)")


if __name__ == "__main__":
    main()
