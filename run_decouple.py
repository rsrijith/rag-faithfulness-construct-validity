"""Real-output DECOUPLING (the new headline, per the lock review).

On REAL generated cited-RAG answers (NOT perturbed, NOT citation-filtered), score each sentence two ways:
  - groundedness: is the sentence supported by the FULL context?            (HHEM)
  - attribution:  is the sentence supported by its SPECIFIC cited passage?  (HHEM AND independent roberta-MNLI)
If the model grounds well but mis-cites, groundedness is high while attribution varies -> the two constructs
DECOUPLE on real output. This is empirical and non-tautological: on real data they COULD correlate; showing
they don't (where they could) is the finding. Independent NLI backbone breaks the HHEM circularity.

Reports: Pearson/Spearman(groundedness, attribution); rate of grounded-but-mis-attributed sentences
(grounded>0.6 AND attribution<0.4). Caches the generated set.
"""

import json
import os
import sys
import re
import numpy as np
from scipy.stats import pearsonr, spearmanr
from data.gen_rag import _gen, _SENT_CITE

CACHE = os.path.join(os.path.dirname(__file__), "data", "decouple_cache.json")


def build(n):
    if os.path.exists(CACHE):
        d = json.load(open(CACHE))
        if len(d) >= n:
            return d[:n]
    from data.load import load_hotpot_citation
    raw = load_hotpot_citation(n=n * 3, n_distractors=3)  # more distractors => more chances to mis-cite
    out = []
    for r in raw:
        if len(out) >= n:
            break
        pid2text = {p["id"]: p["text"] for p in r["passages"]}
        try:
            gen = _gen(r["question"], r["passages"])
        except Exception:
            continue
        sents = []
        for seg, cite in _SENT_CITE.findall(gen):
            s = seg.strip().strip("-• ").split("\n")[-1].strip()
            if cite in pid2text and len(s) > 12:
                sents.append({"text": s, "cite": cite})
        if sents:  # NO citation-correctness filter -- keep natural (possibly mis-cited) output
            out.append({"id": r["id"], "passages": r["passages"], "answer": sents})
    json.dump(out, open(CACHE, "w"), indent=1)
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    items = build(n)
    from metrics.hhem_metric import HHEMMetric
    from metrics.sentnli_metric import SentNLIMetric
    hhem, nli = HHEMMetric(), SentNLIMetric()

    def score(passages, sent_text):  # returns (grounded_full, attr_cited) under a given scorer
        pass

    grounded, attr_hhem, attr_nli = [], [], []
    for it in items:
        full = it["passages"]
        pid2 = {p["id"]: p["text"] for p in full}
        for s in it["answer"]:
            cited = pid2.get(s["cite"])
            if cited is None:
                continue
            grounded.append(hhem.score({"passages": full, "answer": [{"text": s["text"], "cite": s["cite"]}]}))
            mini = {"passages": [{"id": "C", "text": cited}], "answer": [{"text": s["text"], "cite": "C"}]}
            attr_hhem.append(hhem.score(mini))
            attr_nli.append(nli.score(mini))
    g, ah, an = np.array(grounded), np.array(attr_hhem), np.array(attr_nli)
    n_sent = len(g)
    print(f"\nDECOUPLING on {len(items)} generated cited-RAG answers, {n_sent} sentences\n")
    print(f"  mean groundedness (full context):     {g.mean():.3f}")
    print(f"  mean attribution  (HHEM, cited):      {ah.mean():.3f}")
    print(f"  mean attribution  (roberta-MNLI, cited): {an.mean():.3f}")
    print(f"  Pearson(grounded, attr_HHEM)   = {pearsonr(g, ah)[0]:+.3f}   Spearman = {spearmanr(g, ah)[0]:+.3f}")
    print(f"  Pearson(grounded, attr_MNLI)   = {pearsonr(g, an)[0]:+.3f}   Spearman = {spearmanr(g, an)[0]:+.3f}")
    gm_hhem = np.mean((g > 0.6) & (ah < 0.4))
    gm_nli = np.mean((g > 0.6) & (an < 0.4))
    print(f"  grounded(>0.6)-but-mis-attributed(<0.4):  HHEM {gm_hhem:.1%}   MNLI {gm_nli:.1%}")
    print("\n  Decoupling = groundedness high, attribution lower/uncorrelated. The grounded-but-miscited")
    print("  fraction is the populated blind spot the groundedness stack scores as faithful.")


if __name__ == "__main__":
    main()
