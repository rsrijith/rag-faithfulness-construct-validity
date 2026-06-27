"""Vectara HHEM-2.1-Open as a deployed groundedness classifier (cross-encoder).

Scores (premise=context, hypothesis=answer) -> probability of factual consistency in [0,1]. A real
faithfulness metric (unlike the BERTScore baseline): it attempts entailment, so the sensitivity probes
(S1/S2) should make it DROP if it is valid. Not citation-aware -> N/A for citation relocation.
"""

from transformers import AutoModelForSequenceClassification
from .base import item_context, item_answer


class HHEMMetric:
    name = "HHEM-2.1-Open"
    method = "classifier"
    citation_aware = False

    def __init__(self):
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", trust_remote_code=True
        )

    def score(self, item) -> float:
        pair = (item_context(item), item_answer(item))
        # HHEM API: model.predict([(premise, hypothesis)]) -> tensor of consistency probs
        out = self._model.predict([pair])
        return float(out[0])
