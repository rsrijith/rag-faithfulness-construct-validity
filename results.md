# Results (running log — honest, partial)

Status: early-scale, partial metric coverage. Predictions in `predictions.md` are immutable; mismatches are
recorded here. CIs are 95% bootstrap (2000 resamples). EPS=0.05 meaningful-movement band on [0,1].

## VitaminC groundedness grid (real, n=80 SUPPORTS items)

| metric | S1 negation | S2 number-swap | I2 supported-padding |
|---|---|---|---|
| HHEM-2.1-Open | −0.338 [−0.448,−0.227] CATCHES | −0.237 [−0.343,−0.145] CATCHES | −0.016 [−0.052,+0.020] STABLE |
| BERTScore (baseline) | −0.043 [−0.058,−0.030] weak/ambiguous | −0.014 [−0.022,−0.008] BLIND | **+0.419 [+0.369,+0.469] CONFOUNDED** |

Clean findings (hold up with CIs):
- The NLI classifier (HHEM) drops hard on broken faithfulness; the overlap baseline barely moves (and its
  negation "drop" is lexical, not semantic). The metric-family contrast is real.
- **BERTScore inflates +0.42 on supported padding** — a large, robust discriminant-validity confound (append a
  supported sentence, overlap-F1 jumps). Matches the prediction.
- HHEM is NOT length-confounded (padding STABLE). Matches prediction.

## M6 citation-relocation × judge framing (real HotpotQA, gpt-4o-mini, n=40)

| framing | delta | 95% CI | verdict | predicted |
|---|---|---|---|---|
| groundedness | −0.590 | [−0.725,−0.465] | CATCHES | **BLIND** (MISMATCH) |
| attribution | −0.975 | [−1.000,−0.938] | CATCHES | CATCHES ✓ |

**The headline prediction did NOT hold at scale.** The n=4 toy pilot showed the groundedness-framed judge BLIND
(delta 0.00); on real HotpotQA it CATCHES (delta −0.59). This contradicts the committed prediction and the
GroundLM blindness result. Diagnosis (why, and why it is not yet a refutation):
1. **Relocation target is too obvious.** HotpotQA distractor passages are on unrelated topics, so relocating a
   citation to one makes the mismatch glaring. GroundLM's blindness used content-plausible relocations.
2. **Explicit citation tags leak attribution-awareness.** The cited answer is rendered as "(cites D3) <claim>".
   Even under a groundedness prompt, the judge sees the tag and penalizes the obvious mismatch. GroundLM's
   blindness used inline citations with a content-only prompt that did not foreground the tag.
So the current operationalization makes relocation too easy to catch; it does not faithfully replicate the
GroundLM conditions. OPEN: re-run with (a) content-plausible relocation targets (same-topic passages), (b) a
content-only groundedness prompt that does not flag the citation tag, before concluding anything about M6.

## M6 FAITHFUL re-run (plausible swap + content-only prompt + inline markers, HotpotQA, gpt-4o-mini, n=40)

| framing | delta | 95% CI | verdict | predicted |
|---|---|---|---|---|
| content_inline | −0.065 | [−0.140,−0.002] | near-blind (weak) | BLIND ✓ (≈) |
| attribution_inline | −0.775 | [−0.900,−0.625] | CATCHES | CATCHES ✓ |

**The blindness reproduces under faithful conditions.** Fixing the two flaws from the first M6 run (plausible
relocation AMONG on-topic supporting passages instead of unrelated distractors; inline [Dx] markers + a
content-only prompt that does not foreground citations) drops the content-framed judge from −0.59 (CATCHES) to
−0.065 (near-blind) — ~12x weaker than the attribution framing (−0.775). The content delta is not exactly 0
(CI just excludes 0), so the honest claim is "near-blind," not "perfectly blind." Net result: the
citation-attribution blindness is REAL but OPERATIONALIZATION-DEPENDENT — it appears only when relocations are
content-plausible and the judge is content-prompted; obvious relocations or attribution prompts both defeat it.
That conditionality is itself the contribution (characterizing WHEN judges are blind), and the
content-vs-attribution contrast (−0.065 vs −0.775, same model + input) is the headline.

## Deferred / blocked
- **SummaC**: crashes on longer inputs (transformers-4.57 tokenizer kwarg conflict in summac's code). Works at
  tiny scale only. Needs a patch or an isolated env.
- **MiniCheck, AlignScore, AutoAIS/ALCE**: not yet wired (git-install / finicky / T5-11B heavy). The metric
  zoo's conflicting transformers pins point toward per-metric isolated envs.
- Operators S4/S5/S6 (constructed) and I1/I3 (LLM) not yet built. S2 still mis-targets title/year numbers.

## Honest read of the evidence so far
Real, clean failures exist (BERTScore confound; the overlap-vs-NLI contrast). But the marquee
citation-blindness result is fragile under faithful operationalization, and the "easy" probes mostly show
metrics behaving correctly. The interesting paper depends on the harder, under-documented cells
(cherry-pick, multi-hop, parametric leakage, judge biases) producing robust failures — not yet tested.
