"""Perturbation operators for the construct-validity battery.

Each operator transforms a faithful RAG item into a trick input and declares the score movement a VALID metric
must show:
  - sensitivity operators are construct-ALTERING -> a valid metric must DROP   (expected="drop")
  - invariance  operators are construct-PRESERVING -> a valid metric must STAY  (expected="flat")

Only the rule-based operators are implemented here (no model / no API needed): S1, S2, S3, I2. The operators
that require an LLM (I1 paraphrase, I3 hedge) or hand-constructed passages (S4, S5, S6) are declared as stubs
so the registry is complete and the gaps are explicit.

Item schema:
  {id, question, passages:[{id,text}], answer:[{text, cite}], label:"faithful"}
An operator returns:
  {operator, kind, expected, target_construct, applied:bool, item:<perturbed>, changes:[...]}
`applied=False` means the operator could not fire on this item (e.g. no number to swap); such cells are skipped,
not counted as a pass or fail.
"""

import re
import copy

# bidirectional polarity / quantifier antonyms for S1
_ANTONYMS = {
    "increased": "decreased", "decreased": "increased",
    "increase": "decrease", "decrease": "increase",
    "rose": "fell", "fell": "rose", "raised": "lowered", "lowered": "raised",
    "higher": "lower", "lower": "higher", "more": "less", "less": "more",
    "all": "no", "always": "never", "never": "always",
    "can": "cannot", "is": "is not", "are": "are not", "was": "was not", "were": "were not",
}
# quantifier/negation phrases handled as phrase-level flips
_PHRASE_FLIPS = [
    ("not all", "all"),
    ("all", "not all"),
    ("did not increase", "increased"),
    ("increased", "did not increase"),
]


def _flip_one_sentence(text):
    """Return (new_text, change) flipping exactly one polarity cue, or (text, None) if none found."""
    low = text.lower()
    # try phrase-level flips first (longest, most truth-changing)
    for src, dst in _PHRASE_FLIPS:
        idx = low.find(src)
        if idx != -1:
            new = text[:idx] + dst + text[idx + len(src):]
            return new, f"'{src}' -> '{dst}'"
    # token-level antonym flip
    tokens = re.findall(r"\w+|\W+", text)
    for i, tok in enumerate(tokens):
        key = tok.lower()
        if key in _ANTONYMS:
            repl = _ANTONYMS[key]
            if tok[:1].isupper():
                repl = repl[:1].upper() + repl[1:]
            tokens[i] = repl
            return "".join(tokens), f"'{tok}' -> '{repl}'"
    return text, None


def s1_negation_flip(item):
    """Sensitivity: flip a polarity/quantifier cue so the claim becomes false. Valid metric must drop."""
    out = copy.deepcopy(item)
    changes = []
    for sent in out["answer"]:
        new, ch = _flip_one_sentence(sent["text"])
        if ch:
            sent["text"] = new
            changes.append(ch)
            break  # one flip is enough to make the answer unfaithful
    return _result("S1_negation_flip", "sensitivity", "drop", "content_validity", out, changes)


_NUM_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")


def s2_number_swap(item):
    """Sensitivity: replace a number/percentage/amount with one not in the context. Valid metric must drop."""
    out = copy.deepcopy(item)
    changes = []
    for sent in out["answer"]:
        m = _NUM_RE.search(sent["text"])
        if m:
            orig = m.group(0)
            digits = re.sub(r"[^\d.]", "", orig)
            try:
                val = float(digits)
                new_val = val + (10 if val >= 10 else 1)
                new_num = orig.replace(digits, (str(int(new_val)) if new_val.is_integer() else str(new_val)))
            except ValueError:
                continue
            sent["text"] = sent["text"][:m.start()] + new_num + sent["text"][m.end():]
            changes.append(f"'{orig}' -> '{new_num}'")
            break
    return _result("S2_number_swap", "sensitivity", "drop", "content_validity", out, changes)


def s3_citation_relocation(item, seed=0):
    """Sensitivity: reattach each claim's citation to a different (non-supporting) passage. Text unchanged.
    Valid attribution metric / judge must drop; needs >=2 passages."""
    out = copy.deepcopy(item)
    pids = [p["id"] for p in out["passages"]]
    changes = []
    if len(pids) < 2:
        return _result("S3_citation_relocation", "sensitivity", "drop", "convergent_criterion", out, changes,
                       applied=False)
    for i, sent in enumerate(out["answer"]):
        orig = sent.get("cite")
        alts = [p for p in pids if p != orig]
        if alts:
            new = alts[(i + seed) % len(alts)]
            sent["cite"] = new
            changes.append(f"sent{i}: cite {orig} -> {new}")
    return _result("S3_citation_relocation", "sensitivity", "drop", "convergent_criterion", out, changes)


def i2_supported_padding(item):
    """Invariance: append text COPIED FROM THE CONTEXT (trivially grounded). The claims stay fully supported,
    so a valid groundedness metric must stay flat. A drop => length/verbosity confound; a rise => gameability.
    (v2: the old version appended ungrounded filler, which a groundedness metric correctly penalizes — that was
    a sensitivity probe in disguise. Padding must be context-entailed to isolate the discriminant-validity test.)"""
    out = copy.deepcopy(item)
    # grab the passage that supports the first answer sentence; append its text (guaranteed entailed by context)
    cited = out["answer"][0].get("cite")
    pad = next((p["text"] for p in out["passages"] if p["id"] == cited), out["passages"][0]["text"])
    out["answer"][-1] = dict(out["answer"][-1])
    out["answer"][-1]["text"] = out["answer"][-1]["text"] + " " + pad
    r = _result("I2_supported_padding", "invariance", "flat", "discriminant_validity", out,
                [f"appended {len(pad)} chars of context-entailed text"])
    r["pad_text"] = pad
    return r


# --- stubs: complete the registry, make the gaps explicit ------------------------------------------------

def _stub(name, kind, expected, construct, reason):
    def op(item, **kw):
        return _result(name, kind, expected, construct, copy.deepcopy(item), [], applied=False, note=reason)
    op.__name__ = name
    return op


s4_counterfactual_context = _stub("S4_counterfactual_context", "sensitivity", "drop", "convergent_criterion",
                                  "requires hand-constructed false-but-grounded passages")
s5_cherry_pick = _stub("S5_cherry_pick", "sensitivity", "drop", "content_validity",
                       "requires passages where a fragment supports and the rest contradicts")
s6_multi_hop_split = _stub("S6_multi_hop_split", "sensitivity", "drop", "content_validity",
                           "requires claims entailed only by combining two passages")
i1_paraphrase = _stub("I1_paraphrase", "invariance", "flat", "discriminant_validity",
                      "requires an LLM to generate a meaning-preserving paraphrase (run in API env)")
i3_hedge = _stub("I3_hedge_true_claim", "invariance", "flat", "discriminant_validity",
                 "requires an LLM to rewrite a true claim in hedged language (run in API env)")


def _result(operator, kind, expected, construct, item, changes, applied=True, note=None):
    r = {"operator": operator, "kind": kind, "expected": expected, "target_construct": construct,
         "applied": applied and bool(changes), "item": item, "changes": changes}
    if note:
        r["note"] = note
    return r


# registry of implemented rule-based operators (the pilot set)
RULE_BASED = [s1_negation_flip, s2_number_swap, s3_citation_relocation, i2_supported_padding]
STUBS = [s4_counterfactual_context, s5_cherry_pick, s6_multi_hop_split, i1_paraphrase, i3_hedge]
