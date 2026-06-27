"""M6, GroundLM-faithful version: plausible citation swap (among on-topic supporting passages) + inline [Dx]
markers + a CONTENT-only prompt (blind judge can ignore the markers) vs an ATTRIBUTION prompt (must check them).

Tests whether the citation-blindness reproduces at scale under faithful conditions:
  prediction (GroundLM): content framing -> BLIND (delta ~ 0) ; attribution framing -> CATCHES (delta < 0)

Usage: python run_m6_faithful.py [model] [n]
"""

import sys
import numpy as np
from battery import operators as ops
from data.load import load_hotpot_citation
from metrics import judge as J

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
EPS, N_BOOT = 0.05, 2000


def boot(d, rng):
    d = np.array(d)
    ms = [rng.choice(d, len(d), replace=True).mean() for _ in range(N_BOOT)]
    return float(d.mean()), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def main():
    items = load_hotpot_citation(n=N)
    rng = np.random.default_rng(0)
    print(f"\nM6 FAITHFUL: plausible swap + inline markers  (model={MODEL}, n={len(items)} HotpotQA)\n")
    tin = tout = 0
    for framing in ["content_inline", "attribution_inline"]:
        deltas = []
        for it in items:
            res = ops.s3b_citation_swap_plausible(it)
            if not res["applied"]:
                continue
            swapped = res["item"]
            sb, _b, ub = J.openai_judge(it, task=framing, model=MODEL)
            sr, _r, ur = J.openai_judge(swapped, task=framing, model=MODEL)
            tin += ub[0] + ur[0]; tout += ub[1] + ur[1]
            if sb is not None and sr is not None:
                deltas.append(sr - sb)
        m, lo, hi = boot(deltas, rng)
        verdict = "CATCHES" if hi < -EPS else ("BLIND" if lo >= -EPS else "weak/ambiguous")
        tag = "(predicted BLIND)" if framing == "content_inline" else "(predicted CATCHES)"
        print(f"  {framing:20s} delta={m:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  (n={len(deltas)})  -> {verdict}  {tag}")
    print(f"\n  If content_inline is BLIND and attribution_inline CATCHES, the GroundLM blindness reproduces.")
    print(f"  tokens: in={tin} out={tout}")


if __name__ == "__main__":
    main()
