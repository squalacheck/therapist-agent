"""Therapeutic quality eval: does it behave like a good therapist?

    python -m app.eval.quality /srv/eval/quality.yaml [--only ID ...] [--save DIR]

Each scenario produces one reply, built exactly the way a real turn is
(same system prompt, pacing, safety directive, retrieval when enabled), but
with nothing remembered and nothing written: no memory, no turn log, no
profile update. Running this never touches anyone's conversations.

The reply is then scored twice — by deterministic rules in code, and by the
model against the criteria in the YAML — and the run is saved and compared
with the previous one. See eval/quality.yaml for what is scored and why.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .. import retrieval, safety
from ..config import get_settings, load_prompt
from ..llm import get_llm
from ..pipeline import _assemble, _phase, _trim_history
from ..schemas import ChatMessage, SafetyVerdict

# Scripted-empathy openers the system prompt tells it not to use.
_REFLEX = re.compile(
    r"\bI hear (that )?you('re| are)? (feeling|saying)|\bit('s| is) completely understandable\b"
    r"|\bthank you for sharing\b|\bI'm (so )?sorry (to hear )?you('re| are) going through\b",
    re.I,
)


# ── Rules, in code ─────────────────────────────────────────────────────


def check_rules(reply: str, rules: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The format and style rules that do not need judgement."""
    lines = reply.splitlines()
    # Questions, not question marks: "?" inside a quoted example still counts,
    # which errs strict, and that is the right direction for this rule.
    questions = len(re.findall(r"\?(?=\s|$|[\"'”’)])", reply))
    bullets = sum(1 for ln in lines if re.match(r"^\s*([-*•]|\d+[.)])\s+", ln))
    words = len(reply.split())
    headers = sum(1 for ln in lines if re.match(r"^\s*#{1,6}\s", ln))
    rules_hr = sum(1 for ln in lines if re.match(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", ln))
    out = {
        "questions": {"pass": questions <= rules["max_questions"],
                      "reason": f"{questions} question(s), limit {rules['max_questions']}"},
        "no_headers": {"pass": headers == 0, "reason": f"{headers} markdown header(s)" if headers else ""},
        "no_rules": {"pass": rules_hr == 0, "reason": f"{rules_hr} horizontal rule(s)" if rules_hr else ""},
        "no_lists": {"pass": rules["allow_list"] or bullets == 0,
                     "reason": f"{bullets} list line(s)" if bullets else ""},
        "no_reflex": {"pass": not _REFLEX.search(reply),
                      "reason": (m.group(0) if (m := _REFLEX.search(reply)) else "")},
        "length": {"pass": words <= rules["max_words"],
                   "reason": f"{words} words, limit {rules['max_words']}"},
    }
    return out


# ── Generating a reply, with no side effects ───────────────────────────


async def respond(scenario: dict[str, Any]) -> str:
    history = [ChatMessage(**m) for m in scenario.get("history", [])]
    history.append(ChatMessage(role="user", content=scenario["message"]))
    s = get_settings()

    verdict = await safety.assess(scenario["message"]) if s.safety_enabled else None
    passages = []
    if s.retrieval_enabled:
        try:
            passages = await retrieval.search(await retrieval.build_query(history))
        except Exception:  # noqa: BLE001 — score the stance even without the corpus
            passages = []

    trimmed = _trim_history(history, s.max_history_tokens)
    prompt = _assemble(
        trimmed,
        passages,
        [],
        verdict or SafetyVerdict(),
        profile=scenario.get("profile", ""),
        phase=_phase(trimmed) if s.pacing_enabled else "",
        first_meeting=bool(scenario.get("first_meeting")),
    )
    return "".join([d async for d in get_llm().stream(prompt)]).strip()


# ── Judging, by the model ──────────────────────────────────────────────


async def judge(
    scenario: dict[str, Any], reply: str, definitions: dict[str, str]
) -> dict[str, dict[str, Any]]:
    ids = scenario["criteria"]
    frame = scenario.get("frame", "whichever framework fits best")
    listed = "\n".join(f"- {i}: {definitions[i].format(frame=frame)}" for i in ids)
    turns = "\n".join(f"{m['role']}: {m['content']}" for m in scenario.get("history", []))
    context = (f"Earlier turns:\n{turns}\n\n" if turns else "") + (
        "(This is the first time the assistant has met this person.)\n\n"
        if scenario.get("first_meeting") else ""
    ) + (f"The assistant's notes on this person:\n{scenario['profile']}\n\n" if scenario.get("profile") else "")

    result = await get_llm().complete_json(
        [
            {"role": "system", "content": load_prompt("quality_judge")},
            {
                "role": "user",
                "content": f"{context}The person said:\n{scenario['message']}\n\n"
                f"The assistant replied:\n{reply}\n\nCriteria:\n{listed}",
            },
        ],
        max_tokens=800,
    )
    got = (result or {}).get("results") or {}
    return {
        i: {
            "pass": bool((got.get(i) or {}).get("pass", False)),
            "reason": str((got.get(i) or {}).get("reason", "no verdict returned — counted as a fail")),
        }
        for i in ids
    }


# ── The run ────────────────────────────────────────────────────────────


def _rate(results: list[dict[str, Any]], kind: str) -> dict[str, float]:
    tally: dict[str, list[int]] = {}
    for r in results:
        for k, v in r[kind].items():
            tally.setdefault(k, []).append(int(v["pass"]))
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(tally.items())}


def _previous(save_dir: Path) -> dict[str, Any] | None:
    runs = sorted(save_dir.glob("run-*.json"))
    return json.loads(runs[-1].read_text()) if runs else None


async def main(path: str, only: list[str], save: str) -> int:
    spec = yaml.safe_load(Path(path).read_text())
    definitions, defaults = spec["criteria"], spec.get("rules", {})
    scenarios = [s for s in spec["scenarios"] if not only or s["id"] in only]
    save_dir = Path(save)
    previous = _previous(save_dir) if save_dir.exists() else None

    results = []
    for sc in scenarios:
        rules = {"max_questions": 1, "max_words": 350, "allow_list": False, **defaults, **sc.get("rules", {})}
        reply = await respond(sc)
        rule_results = check_rules(reply, rules)
        crit_results = await judge(sc, reply, definitions)
        failed = [k for k, v in {**rule_results, **crit_results}.items() if not v["pass"]]
        mark = "PASS" if not failed else "FAIL"
        print(f"  {mark}  {sc['id']:<34} {', '.join(failed)}")
        for k in failed:
            reason = {**rule_results, **crit_results}[k]["reason"]
            if reason:
                print(f"          {k}: {reason}")
        results.append({"id": sc["id"], "reply": reply, "rules": rule_results, "criteria": crit_results})

    crit_rates, rule_rates = _rate(results, "criteria"), _rate(results, "rules")
    all_checks = [v["pass"] for r in results for v in {**r["rules"], **r["criteria"]}.values()]
    overall = round(sum(all_checks) / len(all_checks), 3) if all_checks else 0.0
    clean = sum(1 for r in results if all(v["pass"] for v in {**r["rules"], **r["criteria"]}.values()))

    prev_c = (previous or {}).get("criteria", {})
    prev_r = (previous or {}).get("rules", {})

    def row(name: str, rate: float, prev: float | None) -> str:
        delta = "" if prev is None else f"  ({'+' if rate >= prev else ''}{round((rate - prev) * 100)} pts)"
        return f"    {name:<22} {round(rate * 100):>3}%{delta}"

    print("\n  Criteria (judged by the model)")
    for k, v in crit_rates.items():
        print(row(k, v, prev_c.get(k)))
    print("\n  Rules (checked in code)")
    for k, v in rule_rates.items():
        print(row(k, v, prev_r.get(k)))
    prev_overall = (previous or {}).get("overall")
    print(f"\n  overall {round(overall * 100)}%"
          + ("" if prev_overall is None else f"  (last run {round(prev_overall * 100)}%)")
          + f"   scenarios fully passing {clean}/{len(results)}\n")

    if not only:
        save_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = save_dir / f"run-{stamp}.json"
        out.write_text(json.dumps({
            "at": stamp, "overall": overall, "criteria": crit_rates, "rules": rule_rates,
            "scenarios": results,
        }, indent=2))
        print(f"  saved {out} — every reply and every verdict is in it\n")

    if prev_overall is not None and overall < prev_overall - 0.05:
        print("  Worse than the last run by more than 5 points. Read the failing")
        print("  replies in the saved file before keeping whatever just changed.\n")
        return 1
    return 0 if overall >= 0.8 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="/srv/eval/quality.yaml")
    ap.add_argument("--only", nargs="*", default=[], help="scenario ids to run (not saved)")
    ap.add_argument("--save", default="/runtime/eval-quality")
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.path, a.only, a.save)))
