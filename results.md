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

## M6 across judge families (faithful: plausible swap + inline, HotpotQA, n=40)

| judge | content framing delta | attribution framing delta |
|---|---|---|
| gpt-4o-mini | −0.077 [−0.15,−0.01] near-blind | −0.753 [−0.88,−0.61] CATCHES |
| gpt-4o | **+0.013 [+0.00,+0.04] BLIND** | −0.775 [−0.90,−0.65] CATCHES |
| claude-haiku-4-5 | −0.139 [−0.22,−0.07] partial-catch | −0.550 [−0.70,−0.38] CATCHES |
| claude-sonnet-4-6 | −0.037 [−0.08,−0.01] near-blind | −0.885 [−0.93,−0.83] CATCHES |
| gemini-2.5-flash | −0.025 [−0.07,+0.00] near-blind | −0.909 [−1.00,−0.73] CATCHES (n=11, transient 503) |

STRONG, MODEL-GENERAL result. The prompt-framing contrast (attribution catches hard, content framing far
weaker) holds across both families (OpenAI, Anthropic) and both sizes. Attribution framing CATCHES on all 4
models (−0.55 to −0.89). Content-framing blindness severity is MODEL-DEPENDENT, and notably **gpt-4o is the
MOST blind** (+0.013, no drop at all) — more blind than gpt-4o-mini — so the blindness is NOT a weak-model
artifact and capability does not fix it. Haiku is the partial exception (catches somewhat even under content
framing). (Gemini blocked: key quota-exhausted; a 3rd family would strengthen it.)
NOTE: a Sonnet parse bug ("Sentence 1" -> constant 0.01) initially faked a +0.000; fixed by parsing the LAST
0-100 integer. Verified Sonnet genuinely catches (scores 0 on swapped) under attribution.

## M7 parametric-knowledge leakage (mismatched evidence, VitaminC, n=60) — NULL so far

| scorer | delta | verdict |
|---|---|---|
| HHEM-2.1-Open | −0.524 [−0.600,−0.443] | DROPS (no leak) |
| judge gpt-4o-mini | −0.774 [−0.870,−0.666] | DROPS (no leak) |
| judge gpt-4o | −0.800 [−0.900,−0.700] | DROPS (no leak) |

No leakage detected. When a world-true claim is paired with unrelated evidence, both the NLI metric and the
LLM-judges correctly drop (judges drop MORE than the NLI metric). The Ramprasad & Wallace leakage finding does
NOT reproduce with this simple swap — likely because unrelated evidence makes non-support obvious. A faithful
leakage test needs a subtler setup (famous fact + topically-related-but-non-stating context). HONEST READ: not
every under-documented cell yields a failure; this one shows robustness. Either build the subtler probe or
report M7 as "robust to obvious evidence mismatch."

## Multi-hop (S6): sentence-level NLI fails claims entailed only by combining passages (HotpotQA, n=30)

Validated multi-hop claims (gpt-4o-mini generated + validated as supported-by-both-and-requires-both):

| scorer | mean | miss-rate (<0.5) |
|---|---|---|
| SentNLI (sentence-level NLI, roberta-large-mnli) | 0.194 | **93%** |
| HHEM (full-context cross-encoder) | 0.776 | 13% |
| judge gpt-4o-mini | 0.900 | 10% |

CLEAN SECOND FINDING. Sentence-level NLI metrics (the SummaC / per-sentence family) score fully-supported
multi-hop claims as UNsupported 93% of the time, while full-context metrics and judges handle them. The failure
is attributable to GRANULARITY (no single context sentence entails a multi-hop claim), not to bad claims —
HHEM, which is independent of the gpt-4o-mini that built the claims, scores them 0.78 (confirming support).
Caveats: one MNLI model stands in for the sentence-level family (the mechanism generalizes; SummaC/AutoAIS-per-
sentence would behave the same); n=30; some passages >512 tokens get truncated (minor). Also banks SentNLI as a
working NLI metric replacing broken SummaC.

## Where the findings stand (2 robust + grid contrasts)
1. M6 citation-attribution blindness: prompt-dependent, model-general, capability-resistant (gpt-4o most blind).
2. Multi-hop granularity blindness of sentence-level NLI (93% miss) vs full-context metrics.
3. Grid contrasts: NLI catches faithfulness breaks / overlap blind; BERTScore +0.42 padding confound.
M7 leakage = null (robust). The framework IS paying off beyond the headline (unlike the M7 worry).

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
