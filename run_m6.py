"""M6 headline cell: are LLM-judges blind to citation relocation, and does the PROMPT framing flip it?

Same input to both framings (the cited answer + passages); only the prompt instruction differs:
  - groundedness framing: 'is the answer supported by the context?'   -> predicted BLIND on relocation
  - attribution  framing: 'is each sentence supported by ITS cited passage?' -> predicted CATCHES
Tests base vs S3-relocated on every item with >=2 passages, with a cheap workhorse model. Bounded spend.
"""

import json
import os
import sys
from battery import operators as ops
from metrics import judge as J

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
here = os.path.dirname(__file__)
items = [it for it in json.load(open(os.path.join(here, "battery", "sample_items.json")))
         if len(it["passages"]) >= 2]


def judge(item, task):
    s, _t, u = J.openai_judge(item, task=task, model=MODEL, answer_mode="cited")
    return s, u


def main():
    print(f"M6 citation-relocation x judge framing  (model={MODEL}, n={len(items)} items)\n")
    tin = tout = 0
    for framing in ["groundedness", "attribution"]:
        deltas, basev, relv = [], [], []
        for it in items:
            reloc = ops.s3_citation_relocation(it)["item"]
            sb, ub = judge(it, framing)
            sr, ur = judge(reloc, framing)
            tin += ub[0] + ur[0]; tout += ub[1] + ur[1]
            if sb is not None and sr is not None:
                basev.append(sb); relv.append(sr); deltas.append(sr - sb)
        mb = sum(basev) / len(basev); mr = sum(relv) / len(relv); md = sum(deltas) / len(deltas)
        verdict = "CATCHES (valid)" if md <= -0.15 else "BLIND (invalid)"
        print(f"  {framing:14s} framing:  base={mb:.2f}  relocated={mr:.2f}  delta={md:+.2f}  -> {verdict}")
    print(f"\n  prediction: groundedness framing -> BLIND ; attribution framing -> CATCHES")
    print(f"  tokens: in={tin} out={tout}  (~cents)")


if __name__ == "__main__":
    main()
