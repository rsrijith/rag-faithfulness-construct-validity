"""M6 faithful, across judge families: is the citation-blindness a property of LLM-judges generally, or one
cheap model? Runs the content-vs-attribution contrast (plausible swap + inline markers) on several judges.

For each judge: content_inline delta (predicted near-BLIND) and attribution_inline delta (predicted CATCHES),
each with a 95% bootstrap CI. A consistent pattern across families = the headline is model-general.
"""

import sys
import time
import numpy as np
from battery import operators as ops
from data.load import load_hotpot_citation
from metrics import judge as J

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
EPS, N_BOOT = 0.05, 2000
MODELS = [
    ("openai", "gpt-4o-mini"),
    ("openai", "gpt-4o"),
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("anthropic", "claude-sonnet-4-6"),
    # ("gemini", "gemini-2.5-flash"),  # key quota-exhausted (429); re-enable when refreshed
]


def call_retry(item, task, provider, model, tries=4):
    for k in range(tries):
        try:
            return J.judge_call(item, task, provider, model)
        except Exception as e:
            if "429" in repr(e) or "rate" in repr(e).lower():
                time.sleep(2 ** k)
                continue
            raise
    return None, None, None


def boot(d, rng):
    d = np.array(d)
    ms = [rng.choice(d, len(d), replace=True).mean() for _ in range(N_BOOT)]
    return float(d.mean()), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def run_framing(items, framing, provider, model):
    deltas, fails = [], 0
    for it in items:
        res = ops.s3b_citation_swap_plausible(it)
        if not res["applied"]:
            continue
        try:
            sb, _b, _ub = call_retry(it, framing, provider, model)
            sr, _r, _ur = call_retry(res["item"], framing, provider, model)
        except Exception as e:
            fails += 1
            if fails >= 5:
                print(f"    ! {provider}/{model} {framing}: aborting after {fails} errors ({repr(e)[:60]})")
                return deltas or None
            continue
        if sb is not None and sr is not None:
            deltas.append(sr - sb)
    return deltas


def main():
    items = load_hotpot_citation(n=N)
    rng = np.random.default_rng(0)
    print(f"\nM6 across judge families  (n={len(items)} HotpotQA, plausible swap + inline)\n")
    print(f"  {'judge':34s} | {'content delta (near-BLIND?)':30s} | attribution delta (CATCHES?)")
    print("  " + "-" * 96)
    for provider, model in MODELS:
        cd = run_framing(items, "content_inline", provider, model)
        ad = run_framing(items, "attribution_inline", provider, model)
        if not cd or not ad:
            continue
        cm, clo, chi = boot(cd, rng)
        am, alo, ahi = boot(ad, rng)
        cverd = "BLIND" if clo >= -EPS else ("CATCHES" if chi < -EPS else "weak")
        averd = "CATCHES" if ahi < -EPS else ("BLIND" if alo >= -EPS else "weak")
        print(f"  {model:34s} | {cm:+.3f} [{clo:+.2f},{chi:+.2f}] {cverd:7s}    | {am:+.3f} [{alo:+.2f},{ahi:+.2f}] {averd}")
    print(f"\n  Pattern: content framing near-blind, attribution framing catches = model-general blindness.")


if __name__ == "__main__":
    main()
