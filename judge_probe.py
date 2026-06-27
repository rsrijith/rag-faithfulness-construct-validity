"""Offload + thinking probe: can cheap/fast models judge faithfulness, and does thinking change the
headline citation-relocation result? Small, bounded API spend (~20 calls).

For each model config, measures:
  GROUNDEDNESS task on s1: base (should score HIGH) vs negation-flipped (should score LOW)  -> is it competent?
  ATTRIBUTION  task on s1: base (HIGH) vs citation-relocated (LOW only if the judge CATCHES it) -> payoff cell
Prints score separation, tokens, and latency so we can pick the cheapest competent tier.
"""

import json
import os
import time
from battery import operators as ops
from metrics import judge as J

here = os.path.dirname(__file__)
items = {it["id"]: it for it in json.load(open(os.path.join(here, "battery", "sample_items.json")))}
base = items["s1"]
negation = ops.s1_negation_flip(base)["item"]
relocated = ops.s3_citation_relocation(base)["item"]

# (label, fn(item, task)->(score,text,usage))
CONFIGS = [
    ("haiku (no-think)", lambda it, t: J.anthropic_judge(it, t, model="claude-haiku-4-5-20251001", thinking=False)),
    ("haiku (thinking)", lambda it, t: J.anthropic_judge(it, t, model="claude-haiku-4-5-20251001", thinking=True)),
    ("gpt-4o-mini", lambda it, t: J.openai_judge(it, t, model="gpt-4o-mini")),
    ("gpt-4.1-nano", lambda it, t: J.openai_judge(it, t, model="gpt-4.1-nano")),
    ("gpt-5-nano (reason=minimal)", lambda it, t: J.openai_judge(it, t, model="gpt-5-nano", reasoning_effort="minimal")),
    ("gpt-5-nano (reason=low)", lambda it, t: J.openai_judge(it, t, model="gpt-5-nano", reasoning_effort="low")),
]


def call(fn, item, task):
    t0 = time.time()
    try:
        score, _text, usage = fn(item, task)
        return score, usage, time.time() - t0, None
    except Exception as e:
        return None, (0, 0), time.time() - t0, repr(e)[:80]


def main():
    print(f"{'config':30s} | GROUNDEDNESS base/neg (sep) | ATTRIBUTION base/reloc (caught?) | tok_out | sec")
    print("-" * 115)
    total_in = total_out = 0
    for label, fn in CONFIGS:
        gb, ub, _, eg = call(fn, base, "groundedness")
        gn, un, _, _ = call(fn, negation, "groundedness")
        ab, uab, _, _ = call(fn, base, "attribution")
        ar, uar, sec_sum, ea = call(fn, relocated, "attribution")
        toks = sum(u[1] for u in [ub, un, uab, uar])
        total_in += sum(u[0] for u in [ub, un, uab, uar]); total_out += toks
        secs = sec_sum  # last call timing as a rough per-call latency proxy
        def f(x): return f"{x:.2f}" if isinstance(x, float) else "ERR"
        sep = (f"{f(gb)}/{f(gn)}") + ("  GOOD" if (gb is not None and gn is not None and gb - gn > 0.3) else "  weak")
        caught = (f"{f(ab)}/{f(ar)}") + ("  CATCHES" if (ab is not None and ar is not None and ab - ar > 0.3) else "  BLIND")
        err = f"  !{eg or ea}" if (eg or ea) else ""
        print(f"{label:30s} | {sep:26s} | {caught:30s} | {toks:5d} | {secs:.1f}{err}")
    print("-" * 115)
    print(f"probe token usage: in={total_in} out={total_out} (a few cents total)")
    print("Reads: GROUNDEDNESS 'GOOD' = cheap model competent on clear cases. ATTRIBUTION 'BLIND' = judge")
    print("scores relocated citation ~as high as base (the predicted failure). Compare no-think vs thinking/reason.")


if __name__ == "__main__":
    main()
