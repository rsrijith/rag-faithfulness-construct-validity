"""The defensible reframe: paired attribution-vs-content sensitivity CONTRAST + AUC discrimination.

For each judge, on the SAME items: content_delta_i and attribution_delta_i (s3b plausible swap), then
  d_i = attribution_delta_i - content_delta_i   (paired contrast, well-powered)
and a discrimination check = P(base_score > swap_score) per framing:
  ~0.5 = the framing cannot tell faithful from relocated (the rigorous 'blind' evidence, an AUC claim)
  ~1.0 = it reliably catches the relocation.
temperature=0 (set in judge.py) removes judge run-to-run noise. Bootstrap CIs on the paired contrast.

Usage: python run_contrast.py [n]
"""

import sys
import time
import numpy as np
from battery import operators as ops
from data.load import LOADERS
import sys as _sys
DS = _sys.argv[2] if len(_sys.argv) > 2 else "hotpot"
from metrics import judge as J

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
N_BOOT = 5000
MODELS = [
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("openai", "gpt-4o-mini"),
    ("openai", "gpt-4o"),
    ("anthropic", "claude-sonnet-4-6"),
]


def call(item, task, provider, model):
    for k in range(4):
        try:
            return J.judge_call(item, task, provider, model)[0]
        except Exception as e:
            if "429" in repr(e) or "rate" in repr(e).lower() or "503" in repr(e):
                time.sleep(2 ** k); continue
            raise
    return None


def boot_ci(d, rng):
    d = np.array(d)
    ms = [rng.choice(d, len(d), replace=True).mean() for _ in range(N_BOOT)]
    return float(d.mean()), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def discrim(base, swap):
    # P(base > swap) over items; 0.5 = chance (blind), 1.0 = always catches
    w = sum((b > s) + 0.5 * (b == s) for b, s in zip(base, swap))
    return w / len(base)


def main():
    items = LOADERS[DS](n=N)
    rng = np.random.default_rng(0)
    print(f"\nPaired contrast + discrimination  (n={len(items)} HotpotQA, s3b swap, temp=0)\n")
    print(f"  {'judge':32s} | content Δ | attrib Δ | PAIRED d (attr-content) [CI] | content-disc | attrib-disc")
    print("  " + "-" * 112)
    for provider, model in MODELS:
        cb, cs, ab, as_ = [], [], [], []
        for it in items:
            sw = ops.s3b_citation_swap_plausible(it)["item"]
            x = [call(it, "content_inline", provider, model), call(sw, "content_inline", provider, model),
                 call(it, "attribution_inline", provider, model), call(sw, "attribution_inline", provider, model)]
            if any(v is None for v in x):
                continue
            cb.append(x[0]); cs.append(x[1]); ab.append(x[2]); as_.append(x[3])
        if len(cb) < 10:
            print(f"  {model:32s} | insufficient data (n={len(cb)})"); continue
        cdelta = [s - b for b, s in zip(cb, cs)]
        adelta = [s - b for b, s in zip(ab, as_)]
        d = [a - c for a, c in zip(adelta, cdelta)]
        dm, dlo, dhi = boot_ci(d, rng)
        print(f"  {model:32s} | {np.mean(cdelta):+.3f}   | {np.mean(adelta):+.3f}   | "
              f"{dm:+.3f} [{dlo:+.2f},{dhi:+.2f}] (n={len(cb)}) | {discrim(cb, cs):.2f}         | {discrim(ab, as_):.2f}")
    print("\n  Headline = paired d (how much MORE the attribution prompt drops than the content prompt, same items).")
    print("  content-disc ~0.5 => content framing cannot distinguish faithful vs relocated (AUC-style blindness).")


if __name__ == "__main__":
    main()
