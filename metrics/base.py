"""Common metric interface for the battery.

Every metric scores how well an `answer` is supported by its `context` (the retrieved passages), returning a
float in [0,1] where higher = more faithful/grounded. Citation-aware metrics also receive the per-claim
citation map; metrics that ignore citations simply don't use it (and are N/A for the citation-relocation
operator).
"""

from typing import Protocol, List, Dict, Optional


def item_context(item) -> str:
    return "\n".join(p["text"] for p in item["passages"])


def item_answer(item) -> str:
    return " ".join(s["text"] for s in item["answer"])


class Metric(Protocol):
    name: str
    method: str          # 'overlap' | 'nli' | 'classifier' | 'llm-judge'
    citation_aware: bool

    def score(self, item) -> float:
        ...
