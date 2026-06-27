"""Multi-hop swing: do sentence-level NLI metrics fail on claims entailed only by COMBINING two passages?

Generate a declarative multi-hop claim from HotpotQA bridge items (gpt-4o-mini), validate that it is supported
by the two passages TOGETHER and needs BOTH (gpt-4o-mini). Then score the validated claim (with FULL context)
with:
  - SentNLI (sentence-level NLI; predicted to FAIL -- no single context sentence entails a multi-hop claim)
  - HHEM (full-context cross-encoder; predicted to handle it better)
  - judge gpt-4o-mini (predicted to handle it)
Failure = scoring a validated, fully-supported multi-hop claim as UNsupported. Caches generations.

Usage: python run_multihop.py [n]
"""

import json
import os
import sys
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from data.load import load_hotpot_citation

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
CACHE = os.path.join(os.path.dirname(__file__), "data", "multihop_cache.json")
GEN = "gpt-4o-mini"


def gen_claim(c, q, d1, d2):
    p = (f"Question: {q}\nPassage A: {d1}\nPassage B: {d2}\n\n"
         "Write ONE declarative sentence that answers the question and is true ONLY by combining facts from "
         "BOTH passages (each passage supplies part of it). Output only the sentence.")
    return c.chat.completions.create(model=GEN, messages=[{"role": "user", "content": p}],
                                     max_completion_tokens=80).choices[0].message.content.strip()


def validate(c, claim, d1, d2):
    p = (f"Passage A: {d1}\nPassage B: {d2}\nClaim: {claim}\n\n"
         "Is the claim fully supported by A and B TOGETHER, AND does it require BOTH (not derivable from either "
         "passage alone)? Answer yes or no only.")
    a = c.chat.completions.create(model=GEN, messages=[{"role": "user", "content": p}],
                                  max_completion_tokens=4).choices[0].message.content.strip().lower()
    return a.startswith("y")


def build_items(n):
    if os.path.exists(CACHE):
        items = json.load(open(CACHE))
        if len(items) >= n:
            return items[:n]
    c = OpenAI()
    raw = load_hotpot_citation(n=n * 3)
    items = []
    for r in raw:
        if len(items) >= n:
            break
        if len(r["passages"]) < 2:
            continue
        d1, d2 = r["passages"][0]["text"], r["passages"][1]["text"]
        claim = gen_claim(c, r["question"], d1, d2)
        if not validate(c, claim, d1, d2):
            continue
        items.append({"id": r["id"], "question": r["question"],
                      "passages": [{"id": "D1", "text": d1}, {"id": "D2", "text": d2}],
                      "answer": [{"text": claim, "cite": "D1"}], "label": "faithful_multihop"})
    json.dump(items, open(CACHE, "w"), indent=1)
    return items


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    items = build_items(n)
    print(f"\nMulti-hop: {len(items)} validated multi-hop claims (supported by combining both passages)\n")
    from metrics.sentnli_metric import SentNLIMetric
    from metrics.hhem_metric import HHEMMetric
    from metrics import judge as J
    sent, hhem = SentNLIMetric(), HHEMMetric()
    scorers = [("SentNLI (sent-level)", lambda it: sent.score(it)),
               ("HHEM (full-context)", lambda it: hhem.score(it)),
               ("judge gpt-4o-mini", lambda it: J.openai_judge(it, task="groundedness", model="gpt-4o-mini")[0])]
    for name, fn in scorers:
        vals = [fn(it) for it in items]
        vals = [v for v in vals if v is not None]
        miss = sum(v < 0.5 for v in vals) / len(vals)
        print(f"  {name:22s}  mean={np.mean(vals):.3f}  miss-rate(<0.5)={miss:.0%}  (low = fails multi-hop)")
    print("\n  Failure = scoring a validated, fully-supported multi-hop claim as unsupported.")


if __name__ == "__main__":
    main()
