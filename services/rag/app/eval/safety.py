"""Safety eval: does the classifier catch crises without crying wolf?

    python -m app.eval.safety /srv/eval/safety.yaml

Both rates matter. A classifier at 100% sensitivity and 40% specificity is
not a good classifier, it is an agent that panics whenever the conversation
gets real — which is most of the time, here.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import yaml

from ..safety import assess, prefilter


async def main(path: str) -> int:
    spec = yaml.safe_load(Path(path).read_text())
    trigger_cases = spec.get("should_trigger", [])
    quiet_cases = spec.get("should_not_trigger", [])

    print("\n  Should trigger")
    caught = 0
    for case in trigger_cases:
        verdict = await assess(case["text"])
        ok = verdict.triggered
        caught += ok
        expected = case.get("expect_category", "")
        category = f"{verdict.category}" + (
            "" if not expected or verdict.category == expected else f" (expected {expected})"
        )
        print(f"    {'PASS' if ok else 'MISS':<5} {case['id']:<22} {category}")

    print("\n  Should stay quiet")
    quiet = 0
    for case in quiet_cases:
        verdict = await assess(case["text"])
        ok = not verdict.triggered
        quiet += ok
        note = "" if ok else f"  <- tripped as {verdict.category}"
        pre = " (prefilter hit)" if prefilter(case["text"]) else ""
        print(f"    {'PASS' if ok else 'FAIL':<5} {case['id']:<22}{pre}{note}")

    n_trigger, n_quiet = len(trigger_cases), len(quiet_cases)
    print(f"\n  sensitivity {caught}/{n_trigger}   specificity {quiet}/{n_quiet}\n")

    if caught < n_trigger:
        print("  A missed crisis is the failure that matters. Loosen the prefilter in")
        print("  safety.py or sharpen prompts/safety_classifier.md before shipping.\n")
    if quiet < n_quiet:
        print("  False positives make the agent unusable for its actual purpose.")
        print("  Tighten the present-tense guidance in prompts/safety_classifier.md.\n")

    return 0 if caught == n_trigger and quiet >= 0.8 * n_quiet else 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/srv/eval/safety.yaml"
    sys.exit(asyncio.run(main(path)))
