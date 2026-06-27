"""ALCE-style citation recall (Gao et al., EMNLP 2023) -- a DEPLOYED attribution metric.

For each answer sentence, check whether the SPECIFIC passage it cites entails it. The citation mapping IS an
input -> citation_aware=True. ALCE uses TRUE (T5-11B NLI) as the scorer; we use HHEM as the entailment backbone
(a deployed NLI classifier). Relocating a citation to a non-supporting passage should drop the score (the cited
passage no longer entails the sentence) -> this metric CATCHES citation-attribution errors, unlike RAGAS
faithfulness which never sees the citation.
"""

from .hhem_metric import HHEMMetric


class ALCECitationRecall:
    name = "ALCE-citation-recall"
    method = "nli"
    citation_aware = True

    def __init__(self):
        self._hhem = HHEMMetric()

    def score(self, item) -> float:
        pid2text = {p["id"]: p["text"] for p in item["passages"]}
        scores = []
        for s in item["answer"]:
            cited = pid2text.get(s.get("cite"))
            if cited is None:
                continue
            # HHEM(premise=cited passage, hypothesis=sentence): does THIS cited passage support the sentence?
            mini = {"passages": [{"id": "C", "text": cited}], "answer": [{"text": s["text"], "cite": "C"}]}
            scores.append(self._hhem.score(mini))
        return sum(scores) / len(scores) if scores else 0.0
