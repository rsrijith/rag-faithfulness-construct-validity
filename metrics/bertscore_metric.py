"""BERTScore as the overlap/embedding baseline (predicted BLIND to meaning-flips, the punching bag).

Scores answer vs. context with BERTScore-F1. Not citation-aware -> N/A for citation relocation.
"""

from bert_score import BERTScorer
from .base import item_context, item_answer


class BertScoreMetric:
    name = "BERTScore"
    method = "overlap"
    citation_aware = False

    def __init__(self):
        # roberta-large, default; downloaded on first use
        self._scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    def score(self, item) -> float:
        cand = [item_answer(item)]
        ref = [item_context(item)]
        _, _, f1 = self._scorer.score(cand, ref)
        v = float(f1[0])
        # rescale_with_baseline can go slightly <0 / >1; clamp to [0,1] for a comparable scale
        return max(0.0, min(1.0, v))
