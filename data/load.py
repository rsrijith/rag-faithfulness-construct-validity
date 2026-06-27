"""Real-data loaders -> the battery item schema {question, passages:[{id,text}], answer:[{text,cite}], label}.

Two ungated sources:
  - VitaminC (tals/vitaminc): SUPPORTS (evidence, claim) pairs -> faithful single-passage groundedness items.
    Covers content/discriminant/reliability cells for all groundedness metrics + the groundedness judge.
  - HotpotQA (distractor): gold supporting sentences cited to their source paragraphs -> faithful multi-passage
    citation items. Covers S3 / M6 / attribution. No answer generation needed (answer = verbatim support).
"""

import re
from datasets import load_dataset


def load_vitaminc(n=150, split="test", seed=0):
    ds = load_dataset("tals/vitaminc", split=split)
    ds = ds.filter(lambda r: r["label"] == "SUPPORTS")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    items = []
    for i, r in enumerate(ds):
        ev = r["evidence"].strip()
        claim = r["claim"].strip()
        if len(ev) < 20 or len(claim) < 12:
            continue
        items.append({
            "id": f"vitc_{i}",
            "question": r.get("page", ""),
            "passages": [{"id": "D1", "text": ev}],
            "answer": [{"text": claim, "cite": "D1"}],
            "label": "faithful",
        })
    return items


def load_hotpot_citation(n=100, split="validation", seed=0, n_distractors=1):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split).shuffle(seed=seed)
    items, used = [], 0
    for r in ds:
        if used >= n:
            break
        titles = r["context"]["title"]
        sents = r["context"]["sentences"]
        title2sents = {t: s for t, s in zip(titles, sents)}
        sup_titles = r["supporting_facts"]["title"]
        sup_sids = r["supporting_facts"]["sent_id"]
        # build one cited claim per distinct supporting paragraph (verbatim supporting sentence)
        passages, answer, seen = [], [], {}
        pid = 0
        ok = True
        for t, sid in zip(sup_titles, sup_sids):
            if t not in title2sents or sid >= len(title2sents[t]):
                ok = False
                break
            if t not in seen:
                pid += 1
                did = f"D{pid}"
                seen[t] = did
                passages.append({"id": did, "text": " ".join(title2sents[t]).strip()})
            sent = title2sents[t][sid].strip()
            if len(sent) > 15:
                answer.append({"text": sent, "cite": seen[t]})
        if not ok or len(passages) < 2 or len(answer) < 2:
            continue
        # add a distractor passage or two (non-supporting) so relocation has somewhere wrong to point
        for t in titles:
            if pid - len(seen) >= n_distractors:
                break
            if t not in seen:
                pid += 1
                passages.append({"id": f"D{pid}", "text": " ".join(title2sents[t]).strip()})
        items.append({
            "id": f"hotpot_{used}",
            "question": r["question"].strip(),
            "passages": passages,
            "answer": answer,
            "label": "faithful",
        })
        used += 1
    return items


def load_2wiki_citation(n=100, split="validation", seed=0, n_distractors=1):
    """Second cited-RAG dataset (different corpus, same extractive construction as HotpotQA). 2Wiki context is a
    list of [title, [sentences]] pairs; supporting_facts a list of [title, sent_id] pairs."""
    ds = load_dataset("voidful/2WikiMultihopQA", split=split).shuffle(seed=seed)
    items, used = [], 0
    for r in ds:
        if used >= n:
            break
        title2sents = {t: s for t, s in r["context"]}
        passages, answer, seen, pid, ok = [], [], {}, 0, True
        for t, sid in r["supporting_facts"]:
            if t not in title2sents or sid >= len(title2sents[t]):
                ok = False
                break
            if t not in seen:
                pid += 1
                seen[t] = f"D{pid}"
                passages.append({"id": f"D{pid}", "text": " ".join(title2sents[t]).strip()})
            sent = title2sents[t][sid].strip()
            if len(sent) > 15:
                answer.append({"text": sent, "cite": seen[t]})
        if not ok or len(passages) < 2 or len(answer) < 2:
            continue
        for t, _ in r["context"]:
            if pid - len(seen) >= n_distractors:
                break
            if t not in seen:
                pid += 1
                passages.append({"id": f"D{pid}", "text": " ".join(title2sents[t]).strip()})
        items.append({"id": f"2wiki_{used}", "question": r["question"].strip(),
                      "passages": passages, "answer": answer, "label": "faithful"})
        used += 1
    return items


def _genrag(n=100, **kw):
    from data.gen_rag import load_genrag  # lazy: gen_rag generates via LLM + validates with HHEM
    return load_genrag(n=n)


LOADERS = {"vitaminc": load_vitaminc, "hotpot": load_hotpot_citation, "2wiki": load_2wiki_citation,
           "genrag": _genrag}


if __name__ == "__main__":
    import sys, json
    name = sys.argv[1] if len(sys.argv) > 1 else "vitaminc"
    items = LOADERS[name](n=3)
    print(f"{name}: {len(items)} items")
    print(json.dumps(items[:2], indent=2)[:1400])
