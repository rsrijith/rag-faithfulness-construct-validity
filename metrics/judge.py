"""Minimal LLM-judge harness for the offload/thinking probe.

Two judge tasks:
  - groundedness: does the answer follow from the context? (0-100)
  - attribution:  is each answer sentence supported by ITS cited passage specifically? (0-100)
Supports Anthropic (Claude, optional extended thinking) and OpenAI (incl. gpt-5 reasoning_effort).
Returns (score in [0,1], raw_text). A hard max-token cap keeps cost bounded.
"""

import os
import re
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

GROUNDEDNESS = (
    "You are grading whether an ANSWER is fully supported by the provided CONTEXT.\n"
    "Score 0-100: 100 = every claim in the answer is supported by the context; 0 = unsupported/contradicted.\n"
    "Reply with ONLY the integer.\n\nCONTEXT:\n{ctx}\n\nANSWER:\n{ans}\n\nScore:"
)
ATTRIBUTION = (
    "You are grading CITATION correctness. Each answer sentence is tagged with the passage it cites.\n"
    "Score 0-100: 100 = each sentence is supported by the SPECIFIC passage it cites; 0 = the cited passage "
    "does not support the sentence (even if some other passage might).\n"
    "Reply with ONLY the integer.\n\nPASSAGES:\n{ctx}\n\nCITED ANSWER:\n{ans}\n\nScore:"
)


# GroundLM-faithful inline framing: citations appear as natural [Dx] markers; the CONTENT prompt never tells
# the judge to check them (so a blind judge can ignore them), the ATTRIBUTION prompt does.
CONTENT_INLINE = (
    "Is the ANSWER well supported by the provided CONTEXT?\n"
    "Score 0-100: 100 = the answer's claims are supported by the context; 0 = unsupported/contradicted.\n"
    "Reply with ONLY the integer.\n\nCONTEXT:\n{ctx}\n\nANSWER:\n{ans}\n\nScore:"
)
ATTRIBUTION_INLINE = (
    "Each answer sentence ends with a bracketed source marker like [D2].\n"
    "Score 0-100 whether each sentence is supported by the SPECIFIC passage its marker points to "
    "(0 if the marked passage does not support that sentence, even if a different passage would).\n"
    "Reply with ONLY the integer.\n\nPASSAGES:\n{ctx}\n\nANSWER:\n{ans}\n\nScore:"
)
PROMPTS = {"groundedness": GROUNDEDNESS, "attribution": ATTRIBUTION,
           "content_inline": CONTENT_INLINE, "attribution_inline": ATTRIBUTION_INLINE}
INLINE_TASKS = {"content_inline", "attribution_inline"}


def _ans_inline(item):
    return " ".join(f"{s['text']} [{s.get('cite')}]" for s in item["answer"])


def _ctx(item):
    return "\n".join(f"[{p['id']}] {p['text']}" for p in item["passages"])


def _ans_plain(item):
    return " ".join(s["text"] for s in item["answer"])


def _ans_cited(item):
    return "\n".join(f"- (cites {s.get('cite')}) {s['text']}" for s in item["answer"])


def _parse(text):
    m = re.search(r"\d{1,3}", text or "")
    if not m:
        return None
    return max(0.0, min(1.0, int(m.group(0)) / 100.0))


def _render(item, task, answer_mode):
    if answer_mode == "inline" or task in INLINE_TASKS:
        return _ans_inline(item)
    cited = answer_mode == "cited" or (answer_mode is None and task == "attribution")
    return _ans_cited(item) if cited else _ans_plain(item)


def anthropic_judge(item, task="groundedness", model="claude-haiku-4-5-20251001", thinking=False, max_tokens=2048,
                    answer_mode=None):
    import anthropic
    c = anthropic.Anthropic()
    prompt = PROMPTS[task].format(ctx=_ctx(item), ans=_render(item, task, answer_mode))
    kw = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if thinking:
        kw["thinking"] = {"type": "enabled", "budget_tokens": 1024}
    msg = c.messages.create(**kw)
    text = "".join(b.text for b in msg.content if b.type == "text")
    usage = (msg.usage.input_tokens, msg.usage.output_tokens)
    return _parse(text), text.strip(), usage


def openai_judge(item, task="groundedness", model="gpt-4o-mini", reasoning_effort=None, max_tokens=2048,
                 answer_mode=None):
    from openai import OpenAI
    c = OpenAI()
    prompt = PROMPTS[task].format(ctx=_ctx(item), ans=_render(item, task, answer_mode))
    kw = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": max_tokens}
    if reasoning_effort is not None:
        kw["reasoning_effort"] = reasoning_effort
    r = c.chat.completions.create(**kw)
    text = r.choices[0].message.content
    usage = (r.usage.prompt_tokens, r.usage.completion_tokens)
    return _parse(text), (text or "").strip(), usage
