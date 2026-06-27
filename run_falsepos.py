"""MAKE-OR-BREAK: do deployed sentence-level / per-citation metrics PENALIZE correct multi-hop RAG answers?

For each generated sentence:
  1. INDEPENDENT correctness: an LLM judge (gpt-4o-mini) confirms the sentence is supported by the passages
     (possibly by COMBINING them). This de-circularizes correctness from the NLI metrics under test.
  2. single-passage support: max over individual passages of HHEM(passage, sentence). <0.6 => no single passage
     entails => the sentence is MULTI-HOP grounded (needs combination).
  multi-hop set = LLM-correct AND max_single < 0.6  (genuinely correct, multi-passage-grounded answers)
  single-hop set = LLM-correct AND max_single > 0.6  (control)

Then score BOTH sets with deployed metrics and report the FALSE-POSITIVE rate (correct answer scored <0.5 =
wrongly flagged unfaithful):
  - HHEM (full-context cross-encoder)        [predicted: handles multi-hop -> low FP]
  - SummaC-ZS (sentence-level NLI, deployed)  [predicted: penalizes multi-hop -> high FP]
  - SentNLI (sentence-level NLI)              [predicted: high FP]
  - ALCE-recall (per-citation, deployed)      [predicted: high FP -- single cited passage can't entail]
If the sentence-level/per-citation metrics flag correct multi-hop answers at high rate while HHEM doesn't,
the false-positive/granularity hypothesis holds.
"""

import json
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def llm_correct(passages, sent):
    from openai import OpenAI
    ctx = "\n".join(f"[{p['id']}] {p['text']}" for p in passages)
    prompt = (f"Passages:\n{ctx}\n\nStatement: {sent}\n\nIs the statement fully supported by the passages "
              "(combining information across passages if needed)? Answer yes or no only.")
    r = OpenAI().chat.completions.create(model="gpt-4o-mini", temperature=0, max_completion_tokens=4,
                                         messages=[{"role": "user", "content": prompt}])
    return (r.choices[0].message.content or "").strip().lower().startswith("y")


def main():
    items = json.load(open(os.path.join(os.path.dirname(__file__), "data", "decouple_cache.json")))
    from metrics.hhem_metric import HHEMMetric
    from metrics.summac_metric import SummaCMetric
    from metrics.sentnli_metric import SentNLIMetric
    from metrics.alce import ALCECitationRecall
    hhem, summac, sentnli, alce = HHEMMetric(), SummaCMetric(), SentNLIMetric(), ALCECitationRecall()

    def H1(passage, sent):
        return hhem.score({"passages": [{"id": "C", "text": passage}], "answer": [{"text": sent, "cite": "C"}]})

    multihop, singlehop = [], []   # each entry: (item, sentence)
    for it in items:
        pid2 = {p["id"]: p["text"] for p in it["passages"]}
        for s in it["answer"]:
            if s["cite"] not in pid2:
                continue
            if not llm_correct(it["passages"], s["text"]):
                continue  # only CORRECT sentences
            max_single = max(H1(t, s["text"]) for t in pid2.values())
            (multihop if max_single < 0.6 else singlehop).append((it, s))

    def fp_rates(group, label):
        print(f"\n  {label} (n={len(group)} correct sentences):")
        if not group:
            return
        for name, fn in [("HHEM (full-ctx)", lambda it, s: hhem.score({"passages": it["passages"], "answer": [s]})),
                         ("SummaC-ZS (sent-NLI)", lambda it, s: summac.score({"passages": it["passages"], "answer": [s]})),
                         ("SentNLI (sent-NLI)", lambda it, s: sentnli.score({"passages": it["passages"], "answer": [s]})),
                         ("ALCE (per-citation)", lambda it, s: alce.score({"passages": it["passages"], "answer": [s]}))]:
            vals = np.array([fn(it, s) for it, s in group])
            print(f"     {name:22s}  mean={vals.mean():.3f}  FALSE-POSITIVE(<0.5)={np.mean(vals < 0.5):.0%}")

    print(f"\nFALSE-POSITIVE on CORRECT answers (multi-hop vs single-hop control):")
    fp_rates(multihop, "MULTI-HOP grounded (correct, no single passage entails)")
    fp_rates(singlehop, "SINGLE-HOP grounded (correct, a single passage entails) [control]")
    print("\n  Hypothesis holds if sentence-level/per-citation metrics have HIGH false-positive on multi-hop")
    print("  but LOW on single-hop, while HHEM (full-context) stays low on both.")


if __name__ == "__main__":
    main()
