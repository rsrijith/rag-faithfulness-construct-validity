"""SummaC-ZS as a deployed NLI groundedness metric (sentence-level entailment aggregation).

Zero-shot, parameter-free variant: segments context + answer into sentences, builds an NLI entailment matrix,
aggregates. Scores (document=context, summary=answer) in [0,1]. Not citation-aware.
"""

from summac.model_summac import SummaCZS
from .base import item_context, item_answer


class SummaCMetric:
    name = "SummaC-ZS"
    method = "nli"
    citation_aware = False

    def __init__(self):
        self._m = SummaCZS(granularity="sentence", model_name="vitc", device="cpu", imager_load_cache=False)

    def score(self, item) -> float:
        r = self._m.score([item_context(item)], [item_answer(item)])
        # SummaC-ZS scores can fall outside [0,1]; clamp for a comparable scale across metrics
        return max(0.0, min(1.0, float(r["scores"][0])))
