# RAG Faithfulness — Construct-Validity Stress-Test Battery

A reusable perturbation battery for meta-evaluating RAG faithfulness / attribution / LLM-judge metrics,
organized by construct-validity threat. Each deployed metric is probed with construct-altering perturbations
(a valid metric must respond) and construct-preserving perturbations (a valid metric must stay invariant), and
we report which metric fails which probe.

`predictions.md` is a **predicted-movement grid committed before the battery is run**. The commit date of that
file establishes that the predictions existed prior to the measured results — the diff between this commit and
the later results commits is the evidence that the framework predicted metric behavior rather than explaining
it after the fact.

Status: predictions committed; battery and results to follow.
