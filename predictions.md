# Predicted-Movement Grid (committed before the run)

This file records, BEFORE the battery is run, what each metric is predicted to do on each perturbation. Its
commit date is the priority timestamp: the diff between this commit and the later results commits is the
evidence that the framework *predicted* metric behavior rather than *explained* it after the fact. Predictions
are not edited once results exist — mismatches are recorded in the results file.

## How to read this
Each cell is a (metric × operator) prediction. Operators are either:
- **Sensitivity** (construct-ALTERING — faithfulness was broken): a valid metric MUST drop. Codes:
  **CATCHES** = predicted to correctly drop (passes) · **BLIND** = predicted to fail to drop (real failure).
- **Invariance** (construct-PRESERVING — only surface form changed): a valid metric MUST stay flat. Codes:
  **STABLE** = predicted to correctly stay flat (passes) · **CONFOUNDED** = predicted to move (real failure).
- **N/A** = operator does not apply to this metric. **?** = genuine uncertainty (a payoff cell — see end).

A prediction of BLIND/CONFOUNDED is a claim the metric is invalid against that construct. A "?" is where the
framework's prediction is genuinely at risk — those cells are the scientific payoff.

## Metric shortlist (10 — at least one per measurement method)

| # | Metric | Stage | Method | LLM-judge? | Compute note |
|---|---|---|---|---|---|
| 1 | BERTScore | baseline | embedding overlap | no | local, cheap |
| 2 | AutoAIS | attribution | NLI (T5-11B) | no | heavy — T5-11B; 4-bit or smaller-NLI substitute |
| 3 | ALCE (prec + recall) | attribution | NLI via TRUE | no | uses TRUE/T5-11B; same compute note |
| 4 | AlignScore | groundedness | trained RoBERTa | no | local, cheap |
| 5 | SummaC | groundedness | NLI aggregation | no | local, cheap |
| 6 | MiniCheck | groundedness | trained checker | no | local, cheap |
| 7 | HHEM (2.1-Open) | groundedness | cross-encoder classifier | no | local, cheap (CPU-OK) |
| 8 | RAGAS-faithfulness | groundedness | LLM-judge | yes | backbones = Claude + Gemini |
| 9 | G-Eval | general | LLM-judge | yes | backbones = Claude + Gemini |
| 10 | FaithJudge | groundedness | LLM-judge | yes | backbones = Claude + Gemini |

Judges (8–10) run on ≥2 backbones so bias rows are not single-model artifacts. NLI/classifier metrics (2–7)
are deterministic. BERTScore is the overlap baseline for the content-validity rows. Metric selection and
characterizations follow each metric's primary paper.

## Operators (concrete perturbations)

**Sensitivity (must drop):**
- **S1 Negation/quantifier flip** — flip a negation/quantifier so a grounded claim becomes false while staying
  lexically close ("revenue increased" → "revenue did not increase"; "all" → "some").
- **S2 Number/entity swap** — replace a number/date/entity with one not in the context.
- **S3 Citation relocation** — keep the answer text; reattach each claim's citation to a non-supporting source /
  scramble the citation mapping. (Reuses the citation-relocation probe from prior work; reused instrument.)
- **S4 Counterfactual context (grounded-but-wrong)** — answer faithful to a planted passage that is factually
  false; external ground truth says the claim is wrong. (Target construct declared per cell: groundedness vs
  truth.)
- **S5 Cherry-pick / partial-support** — claim supported by a fragment of the cited doc while the rest of that
  doc contradicts it.
- **S6 Multi-hop split** — claim entailed only by combining two passages; no single passage entails it alone.

**Invariance (must stay flat):**
- **I1 Meaning-preserving paraphrase** — reword a faithful answer with no truth change.
- **I2 Content-free padding** — append true-but-irrelevant sentences / markdown formatting to a faithful answer.
- **I3 Hedge a true claim** — rewrite a true, grounded claim in hedged/low-confidence language.
- **I4 Judge-prompt paraphrase + rerun** — reword the judge rubric (meaning preserved), rerun at fixed input.
- **I5 Score-scale reframe** — score the same items on 1–5 vs 1–10; rank order should be preserved.
- **I6 Generator-identity label swap** — tell the judge the answer was written by "GPT-4" vs "a human" vs the
  judge's own family, content held fixed.

Operator validity is human-checked before any verdict counts (a "paraphrase" that alters meaning, or a
"negation flip" that doesn't change truth, is discarded). Admission criterion is in the codebook.

---

## Grid 1 — Content validity (sensitivity; must drop)

| Operator | BERTScore | AutoAIS | ALCE | AlignScore | SummaC | MiniCheck | HHEM | RAGAS-J | G-Eval | FaithJudge |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 negation/quantifier | BLIND | CATCHES | CATCHES | ? | ? | CATCHES | ? | CATCHES | CATCHES | CATCHES |
| S2 number/entity swap | BLIND | ? | ? | ? | ? | CATCHES | ? | CATCHES | CATCHES | CATCHES |
| S5 cherry-pick/partial | BLIND | ? | ? | BLIND | BLIND | ? | BLIND | ? | ? | ? |
| S6 multi-hop split | BLIND | BLIND | BLIND | ? | BLIND | ? | ? | CATCHES | CATCHES | ? |

Rationale: overlap (BERTScore) is blind to every meaning-flip that preserves lexical surface (well-established).
NLI/trained metrics *should* catch negation/entity flips (their training target) — but prior work (FactCC,
Falsesum) shows entity/number swaps are exactly where NLI quietly fails, so those are "?" not assumed CATCHES.
Partial-support (S5) is documented as the hardest category for every metric — predicted broad failure, the row
where almost everything is BLIND. Multi-hop (S6) should blind sentence-level NLI (no single passage entails) but
holistic LLM-judges may aggregate — split prediction.

## Grid 2 — Convergent/criterion validity (sensitivity; must drop)

| Operator | AutoAIS | ALCE-prec | ALCE-rec | AlignScore | SummaC | MiniCheck | HHEM | RAGAS-J | G-Eval | FaithJudge |
|---|---|---|---|---|---|---|---|---|---|---|
| S3 citation relocation | CATCHES | CATCHES | CATCHES | N/A | N/A | N/A | N/A | **BLIND** | **BLIND** | **BLIND** |
| S4 grounded-but-wrong (truth construct) | BLIND | BLIND | BLIND | BLIND | BLIND | BLIND | BLIND | ? | ? | ? |

Rationale: **S3 citation relocation** — attribution metrics that check the *specific cited passage* (AutoAIS,
ALCE precision/recall) should CATCH a relocated citation; LLM-judges score on content plausibility and are
predicted **BLIND** (consistent with prior citation-gameability findings). Groundedness metrics that score
against the whole context, not the citation mapping, are N/A. **S4 grounded-but-wrong** is the definitional row:
under the *groundedness* construct every metric "correctly" stays high (not a defect); under the *truth*
construct every metric is BLIND — the family conflates grounded with true. Payoff cell: LLM-judges might DROP
here by leaking parametric knowledge (they "know" the planted passage is false) — a *different* failure
(ignoring the provided context). Marked "?".

## Grid 3 — Discriminant validity (invariance; must stay flat)

| Operator | BERTScore | AutoAIS | ALCE | AlignScore | SummaC | MiniCheck | HHEM | RAGAS-J | G-Eval | FaithJudge |
|---|---|---|---|---|---|---|---|---|---|---|
| I1 paraphrase | ? | STABLE | STABLE | CONFOUNDED | CONFOUNDED | STABLE | ? | STABLE | STABLE | STABLE |
| I2 content-free padding | STABLE | STABLE | STABLE | ? | ? | ? | ? | CONFOUNDED | CONFOUNDED | CONFOUNDED |
| I3 hedge a true claim | STABLE | STABLE | STABLE | STABLE | STABLE | STABLE | STABLE | CONFOUNDED | CONFOUNDED | CONFOUNDED |
| I6 generator-identity swap | N/A | N/A | N/A | N/A | N/A | N/A | N/A | CONFOUNDED | ? | CONFOUNDED |

Rationale: paraphrase *should* be invisible to entailment metrics, but prior stress-testing found
AlignScore/SummaC move on paraphrase → predicted CONFOUNDED for those, STABLE for the rest, "?" for overlap.
Content-free padding is the canonical judge gameability → judges CONFOUNDED; decomposition metrics may also
inflate so those are "?". Hedging a true claim is the fluency/confidence confound → judges CONFOUNDED,
entailment metrics STABLE (they ignore tone) — a clean predicted contrast. Generator-identity (I6) =
self-preference, model-specific → CONFOUNDED where judge family = claimed author, "?" cross-family.

## Grid 4 — Reliability (invariance / test-retest; must stay flat)

| Operator | AlignScore | SummaC | MiniCheck | HHEM | RAGAS-J | G-Eval | FaithJudge |
|---|---|---|---|---|---|---|---|
| I4 judge-prompt paraphrase + rerun | N/A | N/A | N/A | N/A | CONFOUNDED | CONFOUNDED | ? |
| I5 score-scale reframe | N/A | N/A | N/A | N/A | CONFOUNDED | CONFOUNDED | N/A |
| domain-shift threshold swing | CONFOUNDED | CONFOUNDED | ? | CONFOUNDED | N/A | N/A | N/A |

Rationale: judge prompt/scale instability is documented but model-specific → predicted CONFOUNDED, FaithJudge
"?" (few-shot exemplars may stabilize it). Domain-shift threshold swing is the classifier reliability row
(reported 6–14 pt cross-domain swings for HHEM/AlignScore) → CONFOUNDED; MiniCheck reportedly transfers best →
"?". I4/I5 are N/A to non-judges; the threshold row is N/A to judges.

---

## Payoff cells (where the framework is genuinely at risk — the findings live here)

Cells whose outcome cannot be confidently predicted, OR where the predicted result would contradict a metric's
own paper:

1. **S3 × LLM-judges → predicted BLIND.** If judges catch relocated citations, the prediction is wrong.
   Highest-stakes cell.
2. **S4 × LLM-judges → "?".** If judges DROP on grounded-but-wrong, they are leaking parametric knowledge
   (a different failure) — reframing "judges are better" as "judges ignore the provided context."
3. **S1/S2 × NLI metrics (AutoAIS/AlignScore/SummaC/HHEM) → "?".** Conventional wisdom says NLI catches negation;
   prior work says it quietly fails on entity/number. A failure here contradicts these metrics' validity claims.
4. **S5 cherry-pick → broad BLIND.** If even the best metrics fail partial-support, that is a clean cross-family
   result; if some catch it, the taxonomy needs the distinction.
5. **I1 paraphrase × AlignScore/SummaC → predicted CONFOUNDED**, contradicting their "robust alignment"
   self-description.
6. **I2 content-free padding × decomposition/classifier metrics → "?".** Judge gameability is known; whether the
   non-judge metrics are equally gameable is open.

## Predicted N/A rate (so "full grid" is honest, not asserted)
Citation operator (S3) applies only to attribution metrics + judges; judge-protocol operators (I4/I5/I6) only to
LLM-judges; the threshold row only to NLI classifiers. The applicable-cell count and N/A rate are tallied in the
results file; the grid is "complete over applicable cells," not "every cell filled."

## Pass/fail thresholds (analysis hygiene)
- Sensitivity verdict: CATCHES if mean score drop ≥ δ with 95% CI excluding 0; BLIND if CI includes 0 / drop < δ.
- Invariance verdict: STABLE if |mean delta| < ε with CI inside ±ε; CONFOUNDED otherwise.
- δ, ε set per metric on a normalized [0,1] scale before running; paired test (perturbed vs original), bootstrap
  CIs, Holm correction across the grid. ≥ N items per applicable cell (N + power note fixed at run start).

## Human-label codebook (the inter-rater-reliability artifact)
Two coders independently label a sample of flagged cells with: (a) which failure mode, (b) which validity
threat, (c) whether the operator is valid (meaning-altering vs meaning-preserving as intended). Codebook with
definitions + examples is written before coding; Cohen's κ is reported; the minimum trusted κ is fixed in
advance.

## Sequence
1. Commit THIS file (the timestamp). 2. Lock δ/ε/N + codebook (small follow-up commit, still pre-run). 3. Build
operators on a shared item set; human-validate operators. 4. Run; record measured deltas + CIs + verdicts in
`results.md`, noting every prediction match/mismatch. 5. κ coding pass. 6. Mismatches + payoff-cell outcomes
become the findings and open-problems sections.
