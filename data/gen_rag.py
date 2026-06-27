"""Generative-RAG dataset: LLM-GENERATED cited answers (not extractive gold sentences).

Addresses the reviewers' deepest external-validity point: HotpotQA/2Wiki "answers" are verbatim gold sentences,
not model output. Here an LLM writes a 2-sentence answer with inline [Dx] citations from the provided passages
(real RAG generation). We then VALIDATE each sentence's citation against its cited passage with HHEM (keep only
correctly-cited, faithful base items), so a subsequent citation swap genuinely breaks a correct citation.
Cached to data/genrag_cache.json.
"""

import json
import os
import re

CACHE = os.path.join(os.path.dirname(__file__), "genrag_cache.json")
GEN_MODEL = "claude-haiku-4-5-20251001"
_SENT_CITE = re.compile(r"(.+?)\s*\[(D\d+)\]", re.S)


def _gen(question, passages):
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    ctx = "\n".join(f"[{p['id']}] {p['text']}" for p in passages)
    prompt = (f"Passages:\n{ctx}\n\nQuestion: {question}\n\n"
              "Write a 2-sentence answer using ONLY these passages. End EACH sentence with the single passage "
              "it draws from, as a marker like [D1]. Output only the two sentences.")
    m = anthropic.Anthropic().messages.create(model=GEN_MODEL, max_tokens=200, temperature=0,
                                              messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in m.content if b.type == "text")


def build_genrag(n=100, seed=0):
    if os.path.exists(CACHE):
        items = json.load(open(CACHE))
        if len(items) >= n:
            return items[:n]
    from data.load import load_hotpot_citation
    from metrics.hhem_metric import HHEMMetric
    hhem = HHEMMetric()
    raw = load_hotpot_citation(n=n * 4, seed=seed)
    items = []
    for r in raw:
        if len(items) >= n:
            break
        pid2text = {p["id"]: p["text"] for p in r["passages"]}
        try:
            gen = _gen(r["question"], r["passages"])
        except Exception:
            continue
        answer, ok = [], True
        for seg, cite in _SENT_CITE.findall(gen):
            sent = seg.strip().strip("-• ").split("\n")[-1].strip()
            if cite not in pid2text or len(sent) < 12:
                continue
            # validate: the cited passage must actually support the generated sentence
            mini = {"passages": [{"id": "C", "text": pid2text[cite]}], "answer": [{"text": sent, "cite": "C"}]}
            if hhem.score(mini) < 0.5:
                ok = False
                break
            answer.append({"text": sent, "cite": cite})
        # need >=2 sentences citing >=2 distinct passages (so a swap has somewhere to go)
        if ok and len(answer) >= 2 and len({a["cite"] for a in answer}) >= 2:
            items.append({"id": f"genrag_{len(items)}", "question": r["question"],
                          "passages": r["passages"], "answer": answer, "label": "faithful_generated"})
    json.dump(items, open(CACHE, "w"), indent=1)
    return items


def load_genrag(n=100, **kw):
    return build_genrag(n=n)


if __name__ == "__main__":
    import sys
    its = build_genrag(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
    print(f"built {len(its)} generated cited-RAG items")
    print(json.dumps(its[0], indent=1)[:700])
