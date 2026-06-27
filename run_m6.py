"""M6 headline at scale: are LLM-judges blind to citation relocation, and does PROMPT framing flip it?

Same cited input to both framings; only the instruction differs:
  - groundedness framing -> predicted BLIND on relocation
  - attribution  framing -> predicted CATCHES
Real multi-passage HotpotQA items; bootstrap CI on the mean delta. Cheap workhorse model. Bounded spend.

Usage: python run_m6.py [model] [n]
"""

import sys
import numpy as np
from battery import operators as ops
from data.load import load_hotpot_citation
from metrics import judge as J

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
EPS, N_BOOT = 0.05, 2000


def judge(item, task):
    s, _t, u = J.openai_judge(item, task=task, model=MODEL, answer_mode="cited")
    return s, u


def boot(d, rng):
    d = np.array(d)
    ms = [rng.choice(d, len(d), replace=True).mean() for _ in range(N_BOOT)]
    return float(d.mean()), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def main():
    items = load_hotpot_citation(n=N)
    rng = np.random.default_rng(0)
    print(f"\nM6 citation-relocation x judge framing  (model={MODEL}, n={len(items)} HotpotQA items)\n")
    tin = tout = 0
    for framing in ["groundedness", "attribution"]:
        deltas = []
        for it in items:
            reloc = ops.s3_citation_relocation(it)["item"]
            sb, ub = judge(it, framing)
            sr, ur = judge(reloc, framing)
            tin += ub[0] + ur[0]; tout += ub[1] + ur[1]
            if sb is not None and sr is not None:
                deltas.append(sr - sb)
        m, lo, hi = boot(deltas, rng)
        verdict = "CATCHES (valid)" if hi < -EPS else ("BLIND (invalid)" if lo >= -EPS else "weak/ambiguous")
        print(f"  {framing:14s} framing:  delta={m:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  (n={len(deltas)})  -> {verdict}")
    print(f"\n  prediction: groundedness -> BLIND ; attribution -> CATCHES")
    print(f"  tokens: in={tin} out={tout}")


if __name__ == "__main__":
    main()
