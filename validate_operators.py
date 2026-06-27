"""Operator-validation harness (the pre-run step).

Runs the rule-based operators on the sample items and auto-checks that each transform actually did what it
claims: a sensitivity operator must materially change the answer's truth/attribution; an invariance operator
must change only surface form, leaving the claims intact. Prints base vs perturbed for eyeballing (the human
spot-check), plus a machine check per cell. No models, no API, no network.
"""

import json
import os
from battery import operators as ops


def answer_text(item):
    return " ".join(s["text"] for s in item["answer"])


def cite_map(item):
    return [s.get("cite") for s in item["answer"]]


def auto_check(name, base, res):
    """Return (ok, detail) — a cheap automatic validity check per operator family."""
    pert = res["item"]
    if not res["applied"]:
        return None, res.get("note", "did not fire on this item")
    if name == "S1_negation_flip":
        ok = answer_text(pert) != answer_text(base)
        return ok, "answer truth-bearing text changed" if ok else "no change"
    if name == "S2_number_swap":
        ok = answer_text(pert) != answer_text(base) and any(c for c in res["changes"])
        return ok, "a number was changed" if ok else "no number changed"
    if name == "S3_citation_relocation":
        ok = cite_map(pert) != cite_map(base) and answer_text(pert) == answer_text(base)
        return ok, "citation remapped, answer text identical" if ok else "citation unchanged or text altered"
    if name == "I2_supported_padding":
        pad = res.get("pad_text", "")
        longer = len(answer_text(pert)) > len(answer_text(base))
        stripped = [s["text"].replace(" " + pad, "") for s in pert["answer"]]
        claims_intact = stripped == [s["text"] for s in base["answer"]]
        grounded = pad and any(pad in p["text"] for p in base["passages"])
        ok = longer and claims_intact and grounded
        return ok, "appended context-entailed text, claims intact" if ok else "padding not grounded / claims altered"
    return None, "no check defined"


def main():
    here = os.path.dirname(__file__)
    items = json.load(open(os.path.join(here, "battery", "sample_items.json")))
    n_pass = n_fail = n_skip = 0
    for item in items:
        print("=" * 90)
        print(f"[{item['id']}] Q: {item['question']}")
        print(f"   base answer: {answer_text(item)}")
        print(f"   base cites : {cite_map(item)}")
        for op in ops.RULE_BASED:
            res = op(item)
            name = res["operator"]
            ok, detail = auto_check(name, item, res)
            tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
            if ok is None:
                n_skip += 1
            elif ok:
                n_pass += 1
            else:
                n_fail += 1
            arrow = "must DROP" if res["expected"] == "drop" else "must STAY FLAT"
            print(f"   - {name:24s} [{tag}] ({arrow:12s} | {res['target_construct']})  {detail}")
            if ok is not None:
                print(f"       perturbed: {answer_text(res['item'])}")
                if name == "S3_citation_relocation":
                    print(f"       cites    : {cite_map(res['item'])}")
    print("=" * 90)
    print(f"operator-validity checks:  PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}")
    print(f"implemented rule-based operators: {[o.__name__ for o in ops.RULE_BASED]}")
    print(f"pending stubs (need LLM/constructed data): {[o.__name__ for o in ops.STUBS]}")


if __name__ == "__main__":
    main()
