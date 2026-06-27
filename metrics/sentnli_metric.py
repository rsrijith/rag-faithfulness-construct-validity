"""Sentence-level NLI groundedness metric (the SummaC/per-sentence family) -- transparent, self-contained.

For each answer sentence, take the MAX entailment probability over individual context sentences, then aggregate
(min across answer sentences). This is the canonical granularity that should FAIL multi-hop: a claim entailed
only by COMBINING two context sentences has no single context sentence that entails it, so max-entailment is
low even when the full context supports the claim. Uses roberta-large-mnli (standard NLI). Not citation-aware.
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .base import item_context, item_answer


def _sents(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.strip()) > 3]


class SentNLIMetric:
    name = "SentNLI"
    method = "nli"
    citation_aware = False

    def __init__(self, model="roberta-large-mnli"):
        self._tok = AutoTokenizer.from_pretrained(model)
        self._m = AutoModelForSequenceClassification.from_pretrained(model).eval()
        # roberta-large-mnli label order: 0=contradiction, 1=neutral, 2=entailment
        self._ent = 2

    @torch.no_grad()
    def _entail(self, premises, hypothesis):
        if not premises:
            return 0.0
        enc = self._tok([(p, hypothesis) for p in premises], padding=True, truncation=True,
                        max_length=512, return_tensors="pt")
        probs = self._m(**enc).logits.softmax(-1)[:, self._ent]
        return float(probs.max())

    def score(self, item) -> float:
        ctx_sents = []
        for p in item["passages"]:
            ctx_sents += _sents(p["text"])
        ans_sents = _sents(item_answer(item)) or [item_answer(item)]
        # min over answer sentences of (max entailment over context sentences)
        return min(self._entail(ctx_sents, a) for a in ans_sents)
