# STATUS: KILLED / BACK BURNER (2026-06-26)

This project is **paused (killed for now)**. Not abandoned destructively — the repo is a real, timestamped
artifact and the negative results are documented — but no paper is being drafted.

## Why killed
Four candidate empirical findings were each pursued and each **dissolved under proper de-confounding**:
1. **M6 citation-attribution blindness** — a verbatim-extractive-answer artifact (vanished on generated answers).
2. **Leg-2 content-vs-attribution judge contrast** — same verbatim artifact (demoted).
3. **Decoupling (groundedness vs attribution on real output)** — multi-hop artifact; genuine mis-attribution
   rate = **0%** when properly defined (models cite correctly).
4. **False-positive / granularity (sentence-level/per-citation metrics penalize correct multi-hop)** — the
   clean number (ALCE 96%) is confounded by a forced-single-citation generation prompt; SummaC-ZS is a scale
   artifact; the strong signal belongs to a strawman NLI.

The only robust result (deployed faithfulness metrics give delta=0 on citation relocation because they don't
consume the citation mapping) is, by unanimous reviewer assessment, analytically near-trivial and cannot carry
a paper. Two independent review panels (8 reviewer-passes total) converged on this. See `results.md` for the
full honest log and `EB1/Papers/TMLR Survey/{05_methodology_review,06_lock_review}.md` for the review records.

## Revisit condition (back burner)
Reconsider ONLY as a **cautionary / methodology workshop note** ("how RAG-faithfulness-metric evaluation setups
manufacture false metric-failures, with a battery to catch them" + the four artifact case studies), and ONLY
if the author's currently-submitted papers are not accepted. Not a TMLR flagship.

## What's salvageable / reusable
- The construct-validity stress-test **battery + harness** (operators, deployed-metric wrappers, runners).
- The **de-confounding methodology** (independent verifiers, LLM-confirmed correctness, verbatim-vs-paraphrase
  and forced-citation controls) — this is the genuine methodological asset.
- Timestamped predictions (`predictions.md`) and a fully honest results log.
