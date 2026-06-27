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


def _salience(m):
    """Prefer the load-bearing quantity: %/$ /decimal numbers, and de-prioritize bare 4-digit years."""
    t = m.group(0)
    return ("%" in t or "$" in t or "." in t, not re.fullmatch(r"\d{4}", t))


def s2_number_swap(item):
    """Sensitivity: replace a salient number/percentage/amount with one not in the context. Valid metric must
    drop. (v2: rebuilds the new number from its value rather than orig.replace -- the old code silently no-op'd
    comma-formatted numbers like '$1,000'; and targets the salient quantity, not the first number/year.)"""
    out = copy.deepcopy(item)
    changes = []
    for sent in out["answer"]:
        toks = list(_NUM_RE.finditer(sent["text"]))
        if not toks:
            continue
        m = max(toks, key=_salience)
        orig = m.group(0)
        num = re.sub(r"[^\d.]", "", orig)
        try:
            val = float(num)
        except ValueError:
            continue
        new_val = val + (10 if val >= 10 else 1)
        new_str = str(int(new_val)) if float(new_val).is_integer() else str(new_val)
        new_num = ("$" if orig.startswith("$") else "") + new_str + ("%" if orig.endswith("%") else "")
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


def s3b_citation_swap_plausible(item):
    """Sensitivity (plausible variant of S3): permute citations AMONG the answer's own cited passages, which are
    the on-topic supporting passages. The cited source stays topically relevant but no longer specifically
    supports its sentence -> a content judge may stay blind; an attribution judge should catch it. This is the
    GroundLM-faithful version (plausible scramble vs S3's relocation to an unrelated distractor)."""
    out = copy.deepcopy(item)
    cites = [s.get("cite") for s in out["answer"]]
    if len(list(dict.fromkeys(cites))) < 2:
        return _result("S3b_citation_swap", "sensitivity", "drop", "convergent_criterion", out, [], applied=False)
    rotated = cites[1:] + cites[:1]   # rotate by 1 so no sentence keeps its own source
    changes = []
    for i, s in enumerate(out["answer"]):
        if rotated[i] != cites[i]:
            s["cite"] = rotated[i]
            changes.append(f"sent{i}: {cites[i]}->{rotated[i]}")
    return _result("S3b_citation_swap", "sensitivity", "drop", "convergent_criterion", out, changes)


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
def s5_cherry_pick(item):
    """Sensitivity (partial-support / cherry-pick): keep the supporting passage but ADD a context sentence that
    contradicts the claim (the negation of the answer). The claim is still entailed by the original passage, but
    the context now also REFUTES it. A max-entailment metric finds the supporting sentence and stays high
    (blind to the contradiction); a robust metric should lower its score given the conflicting evidence."""
    out = copy.deepcopy(item)
    neg, ch = _flip_one_sentence(out["answer"][0]["text"])
    if not ch:
        return _result("S5_cherry_pick", "sensitivity", "drop", "content_validity", out, [], applied=False)
    out["passages"].append({"id": f"D{len(out['passages']) + 1}", "text": neg})
    return _result("S5_cherry_pick", "sensitivity", "drop", "content_validity", out, [f"added contradiction: {ch}"])
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
