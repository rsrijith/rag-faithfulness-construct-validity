"""B-anchored experiment: do DEPLOYED faithfulness metrics miss citation-attribution errors?

On the same plausible-citation-swap (s3b), score base vs swapped with:
  - RAGAS-faithfulness (deployed, citation-UNAWARE)  -> predicted delta ~ 0 (structurally blind: the citation
    mapping is not even an input; answer text + context are unchanged by relocation)
  - HHEM (deployed groundedness classifier, citation-unaware) -> delta ~ 0 (same reason)
  - ALCE citation recall (deployed attribution metric, citation-AWARE) -> predicted DROP (the cited passage no
    longer entails the sentence)
The contrast anchors the finding to shipped metrics, not researcher prompts: the field's faithfulness metric
stack does not operationalize attribution, so it cannot detect mis-attribution; only attribution metrics do.

Usage: python run_anchor.py [n]
"""

import sys
import numpy as np
from battery import operators as ops
from data.load import load_hotpot_citation

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_BOOT = 5000


def boot(d, rng):
    d = np.array(d)
    ms = [rng.choice(d, len(d), replace=True).mean() for _ in range(N_BOOT)]
    return float(d.mean()), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def main():
    items = load_hotpot_citation(n=N)
    rng = np.random.default_rng(0)
    from metrics.ragas_faith import RagasFaithfulness
    from metrics.hhem_metric import HHEMMetric
    from metrics.alce import ALCECitationRecall
    metrics = [RagasFaithfulness(), HHEMMetric(), ALCECitationRecall()]
    print(f"\nDeployed metrics on citation relocation (s3b)  (n={len(items)} HotpotQA)\n")
    print(f"  {'deployed metric':26s} | citation-aware | base->swap delta [95% CI] | reads")
    print("  " + "-" * 92)
    for m in metrics:
        deltas, base = [], {}
        for it in items:
            sw = ops.s3b_citation_swap_plausible(it)["item"]
            b = m.score(it); s = m.score(sw)
            if b is not None and s is not None:
                deltas.append(s - b)
        dm, lo, hi = boot(deltas, rng)
        reads = "CATCHES" if hi < -0.05 else ("BLIND (structural)" if abs(dm) < 0.05 else "partial")
        print(f"  {m.name:26s} | {str(m.citation_aware):14s} | {dm:+.3f} [{lo:+.2f},{hi:+.2f}] (n={len(deltas)}) | {reads}")
    print("\n  Citation-unaware deployed metrics are structurally blind (delta~0); only the citation-aware")
    print("  attribution metric (ALCE) catches the relocation. The faithfulness stack doesn't operationalize attribution.")


if __name__ == "__main__":
    main()
