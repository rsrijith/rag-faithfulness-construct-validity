"""RAGAS faithfulness (Es et al., EACL 2024) -- a DEPLOYED metric, the canonical one.

Algorithm (faithful to the paper, backbone-agnostic; we use a Claude/OpenAI backbone):
  1. LLM decomposes the answer into atomic statements.
  2. LLM verifies whether each statement can be inferred from the CONTEXT.
  3. faithfulness F = (#supported statements) / (#total statements).

Key structural property: the input is (answer text, context). The CITATION MAPPING is never consumed -> RAGAS
faithfulness is by construction blind to citation-attribution errors (relocating a citation leaves answer text
and context unchanged, so F is unchanged). citation_aware=False.
"""

import os
import re
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
from .base import item_context, item_answer  # noqa: E402


class RagasFaithfulness:
    name = "RAGAS-faithfulness"
    method = "llm-judge"
    citation_aware = False

    def __init__(self, provider="anthropic", model="claude-haiku-4-5-20251001"):
        self.provider, self.model = provider, model

    def _llm(self, prompt, max_tokens=512):
        if self.provider == "anthropic":
            import anthropic
            m = anthropic.Anthropic().messages.create(
                model=self.model, max_tokens=max_tokens, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in m.content if b.type == "text")
        else:
            from openai import OpenAI
            r = OpenAI().chat.completions.create(
                model=self.model, temperature=0, max_completion_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content or ""

    def score(self, item) -> float:
        ans, ctx = item_answer(item), item_context(item)
        dec = self._llm(f"Break the following answer into a numbered list of simple, atomic factual "
                        f"statements. Output ONLY the list.\n\nAnswer: {ans}")
        claims = [re.sub(r"^\s*\d+[.)]\s*", "", ln).strip() for ln in dec.splitlines()
                  if re.match(r"^\s*\d+[.)]", ln)]
        claims = [c for c in claims if len(c) > 5] or [ans]
        verdicts = []
        for c in claims:
            v = self._llm(f"CONTEXT:\n{ctx}\n\nStatement: {c}\n\nCan this statement be inferred from the "
                          f"context? Answer ONLY 'yes' or 'no'.", max_tokens=4).strip().lower()
            verdicts.append(v.startswith("y"))
        return sum(verdicts) / len(verdicts) if verdicts else 0.0
