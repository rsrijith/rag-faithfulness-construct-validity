"""SummaC-ZS as a deployed NLI groundedness metric (sentence-level entailment aggregation).

Zero-shot, parameter-free variant: segments context + answer into sentences, builds an NLI entailment matrix,
aggregates. Scores (document=context, summary=answer) in [0,1]. Not citation-aware.
"""

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# SummaC (older code) calls batch_encode_plus(truncation=True, truncation_strategy="only_first"); transformers
# 4.57 rejects both together. Drop the redundant kwarg so SummaC runs on long inputs.
_orig_bep = PreTrainedTokenizerBase.batch_encode_plus
def _patched_bep(self, *a, **kw):
    if "truncation" in kw and "truncation_strategy" in kw:
        kw.pop("truncation_strategy")
    return _orig_bep(self, *a, **kw)
PreTrainedTokenizerBase.batch_encode_plus = _patched_bep

from summac.model_summac import SummaCZS  # noqa: E402
from .base import item_context, item_answer  # noqa: E402


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
