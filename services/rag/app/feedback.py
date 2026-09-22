"""How the session was for them, asked at the end of it.

The therapy research is fairly consistent that the relationship between
client and therapist predicts outcome better than the choice of method,
and that asking about it — briefly, every session — catches ruptures a
therapist would otherwise miss. People rarely volunteer "that didn't land";
they just stop coming back. This is the smallest version of that habit.

Four questions, 0–10, asked by the deterministic close-out rather than the
model, so the wording never drifts:

    heard     did you feel heard and understood
    focus     did we work on what you wanted to work on
    approach  did the way I went about it fit you
    overall   was this worth your time

The answer shapes the next session, quietly: a low score becomes a short
note in the next system prompt (see `prompt_block`), and the full history
is at GET /v1/feedback and `make feedback`.

Flow. Saying "let's end for now" saves everything, then asks. The shutdown
flag is only written once they answer or say skip — the machine going down
before they can reply would make the question pointless. If they carry on
talking instead, the request lapses and the conversation continues.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from sqlalchemy import text as sql

from . import memory
from .config import get_settings

log = structlog.get_logger(__name__)

ITEMS: tuple[tuple[str, str], ...] = (
    ("heard", "Did you feel heard and understood?"),
    ("focus", "Did we work on what you wanted to work on?"),
    ("approach", "Did the way I went about it fit you?"),
    ("overall", "Overall, was this worth your time?"),
)
KEYS = tuple(k for k, _ in ITEMS)

# At or below this, a score is worth carrying into the next session.
LOW = 6

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS session_feedback (
        id         BIGSERIAL PRIMARY KEY,
        scope      TEXT        NOT NULL,
        status     TEXT        NOT NULL DEFAULT 'pending',   -- pending | rated | skipped | lapsed
        exchanges  INTEGER     NOT NULL DEFAULT 0,
        heard      SMALLINT,
        focus      SMALLINT,
        approach   SMALLINT,
        overall    SMALLINT,
        comment    TEXT        NOT NULL DEFAULT '',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        answered_at TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS session_feedback_scope_idx ON session_feedback (scope, created_at DESC)",
)


async def init_schema() -> None:
    """Created at startup rather than by an init script, so an install that
    predates this feature gets the table without rebuilding its database."""
    async with memory.get_sessionmaker()() as session:
        for statement in _DDL:
            await session.execute(sql(statement))
        await session.commit()
    log.info("feedback.schema_ready")


def enabled_for(remember: bool) -> bool:
    """Never on the fresh model: that one promises to store nothing."""
    s = get_settings()
    return s.feedback_enabled and s.memory_enabled and remember


# ── Parsing the answer ─────────────────────────────────────────────────

_SKIP = re.compile(r"^\s*(skip|pass|no(pe)?|no\s+thanks|not\s+now|later|n/?a)\s*[.!]?\s*$", re.I)
_SCORE = r"(10|[0-9])(?:\s*/\s*10)?"
_FOUR = re.compile(
    rf"^\s*{_SCORE}[\s,;/-]+{_SCORE}[\s,;/-]+{_SCORE}[\s,;/-]+{_SCORE}\b[\s.,;:!—–-]*(.*)$", re.S
)
# A lone number counts only on its own or followed by punctuation: "8" or
# "8 — good", never "3 things happened today", which is conversation.
_ONE = re.compile(rf"^\s*{_SCORE}(?:\s+out\s+of\s+10)?\s*(?:[.,;:!—–-]\s*(.*))?$", re.S | re.I)


def parse(text: str) -> dict[str, Any] | None:
    """Read a reply to the four questions, deterministically.

    Accepts "8 7 9 8", "8, 7, 9, 8 — felt rushed at the end", a single
    number (taken as overall), or skip. Anything else returns None: they
    have gone back to talking, and that is not an answer to be guessed at.
    """
    if _SKIP.match(text):
        return {"status": "skipped"}
    if m := _FOUR.match(text):
        scores = [int(g) for g in m.groups()[:4]]
        return {"status": "rated", **dict(zip(KEYS, scores)), "comment": m.group(5).strip()}
    if m := _ONE.match(text):
        return {"status": "rated", "overall": int(m.group(1)), "comment": (m.group(2) or "").strip()}
    return None


# ── Storing ────────────────────────────────────────────────────────────


async def open_request(scope: str, exchanges: int) -> None:
    async with memory.get_sessionmaker()() as session:
        # One outstanding question at a time.
        await session.execute(
            sql("UPDATE session_feedback SET status = 'lapsed' WHERE scope = :scope AND status = 'pending'"),
            {"scope": scope},
        )
        await session.execute(
            sql("INSERT INTO session_feedback (scope, exchanges) VALUES (:scope, :n)"),
            {"scope": scope, "n": exchanges},
        )
        await session.commit()


async def pending(scope: str) -> int | None:
    """The id of an unanswered request from the last few hours, if any."""
    try:
        async with memory.get_sessionmaker()() as session:
            return (
                await session.execute(
                    sql(
                        "SELECT id FROM session_feedback "
                        " WHERE scope = :scope AND status = 'pending' "
                        "   AND created_at > now() - interval '6 hours' "
                        " ORDER BY created_at DESC LIMIT 1"
                    ),
                    {"scope": scope},
                )
            ).scalar_one_or_none()
    except Exception as exc:  # noqa: BLE001
        log.warning("feedback.pending_failed", error=str(exc))
        return None


async def resolve(request_id: int, answer: dict[str, Any] | None) -> None:
    """Record the answer, or mark the request lapsed when there was none."""
    status = answer["status"] if answer else "lapsed"
    values = {k: (answer or {}).get(k) for k in KEYS}
    async with memory.get_sessionmaker()() as session:
        await session.execute(
            sql(
                "UPDATE session_feedback SET status = :status, heard = :heard, focus = :focus, "
                "  approach = :approach, overall = :overall, comment = :comment, answered_at = now() "
                "WHERE id = :id"
            ),
            {"id": request_id, "status": status, "comment": (answer or {}).get("comment", "")[:1000], **values},
        )
        await session.commit()
    log.info("feedback.resolved", status=status, **{k: v for k, v in values.items() if v is not None})


async def latest(scope: str) -> dict[str, Any] | None:
    """The most recent answered request, if it is recent enough to matter.

    Only the latest one counts. If they skipped last time, there is no
    note next time — an older score is about an older conversation.
    """
    try:
        async with memory.get_sessionmaker()() as session:
            row = (
                await session.execute(
                    sql(
                        "SELECT status, heard, focus, approach, overall, comment "
                        "  FROM session_feedback "
                        " WHERE scope = :scope AND status IN ('rated', 'skipped') "
                        "   AND created_at > now() - interval '30 days' "
                        " ORDER BY created_at DESC LIMIT 1"
                    ),
                    {"scope": scope},
                )
            ).mappings().one_or_none()
    except Exception as exc:  # noqa: BLE001
        log.warning("feedback.latest_failed", error=str(exc))
        return None
    return dict(row) if row and row["status"] == "rated" else None


async def history(scope: str, limit: int = 30) -> list[dict[str, Any]]:
    async with memory.get_sessionmaker()() as session:
        rows = (
            await session.execute(
                sql(
                    "SELECT created_at::date AS day, status, exchanges, heard, focus, approach, "
                    "       overall, comment "
                    "  FROM session_feedback WHERE scope = :scope AND status IN ('rated', 'skipped') "
                    " ORDER BY created_at DESC LIMIT :n"
                ),
                {"scope": scope, "n": limit},
            )
        ).mappings().all()
    return [{**dict(r), "day": r["day"].isoformat()} for r in rows]


# ── What the person sees ───────────────────────────────────────────────


def ask(summary: dict[str, Any]) -> str:
    """The close-out message when feedback is wanted. Deterministic."""
    exchanges = summary.get("exchanges") or 0
    opening = f"Alright — that's {exchanges} exchanges saved." if exchanges > 1 else "Alright — saved."
    questions = "\n".join(f"{i}. {q}" for i, (_, q) in enumerate(ITEMS, 1))
    return (
        f"{opening} Your notes are up to date.\n\n"
        "Before I close down — how was this one for you? Four scores out of 10:\n\n"
        f"{questions}\n\n"
        "Something like `8 7 9 8`, with a sentence after if something was off. "
        "Or just say **skip**. Honest low scores are the useful ones: they change "
        "how I go about the next conversation.\n\n"
        "_I'll close down as soon as you answer._"
    )


def thanks(answer: dict[str, Any], shutdown_requested: bool) -> str:
    if answer.get("status") == "skipped":
        line = "No problem. See you next time."
    elif any((answer.get(k) is not None and answer[k] <= LOW) for k in KEYS) or answer.get("comment"):
        line = "Thank you for being straight about it. I'll take that into next time."
    else:
        line = "Thank you — noted."
    tail = (
        "_Shutting the stack down now. Start it again with_ `make up`."
        if shutdown_requested
        else "_Run_ `make end` _when you want to stop the stack._"
    )
    return f"{line}\n\n{tail}"


def prompt_block(last: dict[str, Any] | None) -> str:
    """A note for the next session, only when last time did not go well.

    High scores add nothing to the prompt: "they liked it" gives the model
    nothing to do differently, and it would only encourage it to coast.
    """
    if not last:
        return ""
    low = [k for k in KEYS if last.get(k) is not None and last[k] <= LOW]
    comment = (last.get("comment") or "").strip()
    if not low and not comment:
        return ""

    scores = ", ".join(f"{k} {last[k]}/10" for k in KEYS if last.get(k) is not None)
    advice = {
        "heard": "Slow down. Reflect back what they actually said, in their terms, and check "
        "you have it right before offering anything.",
        "focus": "Early on, ask what they want from today, and keep coming back to it rather "
        "than following whatever seems most interesting to you.",
        "approach": "Ask, once and plainly, what would suit them better — more direct, more "
        "practical, more space, less theory — and then do that.",
        "overall": "Make sure this conversation leaves them with something: a clearer view, "
        "one concrete thing, or simply the sense of having been properly met.",
    }
    lines = [
        "## How the last conversation landed",
        "",
        f"At the end of your last conversation they rated it: {scores}.",
    ]
    if comment:
        lines.append(f'They added: "{comment[:400]}"')
    lines += ["", "Adjust accordingly:"]
    lines += [f"- {advice[k]}" for k in low]
    lines += [
        "",
        "Do not bring the rating up yourself, and do not apologise for last time unless "
        "they raise it. Just be better at the thing they said was missing.",
    ]
    return "\n".join(lines)
