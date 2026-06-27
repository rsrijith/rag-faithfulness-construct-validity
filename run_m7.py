"""M7 parametric-knowledge leakage: does the metric score from world knowledge instead of the given context?

Construction (no generation): take a faithful (claim, evidence) item and REPLACE its evidence with another
item's (topically unrelated) evidence. The claim is still world-true, but the PROVIDED context no longer
supports it. A valid groundedness metric must DROP (context doesn't support the claim). A metric that stays
high is leaking parametric/world knowledge -- the Ramprasad & Wallace failure.

NLI/classifier metrics should drop fully (they only see entailment); LLM-judges may stay higher (leak). The
contrast is the finding. This runner does local metrics now; pass a judge model for the judge arm.

Usage: python run_m7.py <metric|judge:model> [n]
"""

import copy
import sys
import numpy as np
from data.load import load_vitaminc

EPS, N_BOOT = 0.05, 2000


def leak(item, donor):
    out = copy.deepcopy(item)
    out["passages"] = [{"id": "D1", "text": donor["passages"][0]["text"]}]
    return out


def boot(d, rng):
    d = np.array(d)
    ms = [rng.choice(d, len(d), replace=True).mean() for _ in range(N_BOOT)]
    return float(d.mean()), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def scorer(spec):
    if spec.startswith("judge:"):
        model = spec.split(":", 1)[1]
        from metrics import judge as J
        return f"judge[{model}]", lambda it: J.openai_judge(it, task="groundedness", model=model)[0]
    if spec == "hhem":
        from metrics.hhem_metric import HHEMMetric; m = HHEMMetric(); return m.name, m.score
    if spec == "bertscore":
        from metrics.bertscore_metric import BertScoreMetric; m = BertScoreMetric(); return m.name, m.score
    raise SystemExit(spec)


def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else "hhem"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    items = load_vitaminc(n=n)
    name, score = scorer(spec)
    rng = np.random.default_rng(0)
    deltas = []
    for i, it in enumerate(items):
        donor = items[(i + 1) % len(items)]            # mismatched evidence from another item
        sb = score(it)
        sl = score(leak(it, donor))
        if sb is not None and sl is not None:
            deltas.append(sl - sb)
    m, lo, hi = boot(deltas, rng)
    verdict = "DROPS (no leak)" if hi < -EPS else ("LEAKS (stays high)" if lo >= -EPS else "partial leak")
    print(f"\nM7 leakage  {name}  n={len(deltas)}  delta={m:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  -> {verdict}")
    print("  (valid metric must DROP: provided context no longer supports the world-true claim)")


if __name__ == "__main__":
    main()
