"""Time, and the gap since last time.

The stack had timestamps everywhere and told the model none of them. Facts
carried a created_at, the profile carried an updated_at, the `turns` table
was created in 001_schema.sql and then never written to — and the agent
went into every conversation with no idea what day it was or when it had
last spoken to anyone. It could say "you first mentioned that 3 days ago"
about a fact, because format_memory() renders fact age, and still could not
tell the difference between you coming back after lunch and coming back
after a fortnight.

That difference is most of what makes a returning conversation feel like a
returning conversation. "How have things been since Sunday?" is only
available to something that knows it is Wednesday.

Two things live here:

- **Turn logging.** Every user message and every reply, timestamped, in the
  `turns` table that has been sitting empty since the schema was written.
- **The gap.** How long since the last exchange, in words a person would
  use, plus the rules about when to mention it — which mostly means not.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import structlog
from sqlalchemy import text as sql

from . import memory
from .config import get_settings
from .schemas import ChatMessage

log = structlog.get_logger(__name__)


async def log_turn(
    scope: str,
    role: str,
    content: str,
    *,
    retrieval_query: str | None = None,
    safety_category: str | None = None,
) -> None:
    """Append one timestamped turn. Never raises.

    A failure to write the audit log must not cost someone their reply, so
    this swallows everything and complains in the logs instead.
    """
    if not get_settings().time_awareness or not content.strip():
        return
    try:
        async with memory.get_sessionmaker()() as session:
            await session.execute(
                sql(
                    "INSERT INTO turns (id, scope, role, content, retrieval_query, safety_category) "
                    "VALUES (:id, :scope, :role, :content, :q, :safety)"
                ),
                {
                    "id": str(uuid.uuid4()),
                    "scope": scope,
                    "role": role,
                    "content": content[:20000],
                    "q": retrieval_query,
                    "safety": safety_category,
                },
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("session.log_turn_failed", error=str(exc))


async def last_turn_at(scope: str) -> datetime | None:
    """When the previous exchange happened, or None on a first ever turn.

    Must be read BEFORE the current turn is logged, or the gap is always
    zero and the whole thing quietly does nothing.
    """
    if not get_settings().time_awareness:
        return None
    try:
        async with memory.get_sessionmaker()() as session:
            return (
                await session.execute(
                    sql(
                        "SELECT created_at FROM turns WHERE scope = :scope "
                        "ORDER BY created_at DESC LIMIT 1"
                    ),
                    {"scope": scope},
                )
            ).scalar_one_or_none()
    except Exception as exc:  # noqa: BLE001
        log.warning("session.last_turn_failed", error=str(exc))
        return None


def _local(when: datetime | None) -> datetime | None:
    """Normalise to an aware datetime in the local zone.

    Postgres hands back TIMESTAMPTZ as timezone-AWARE, and datetime.now() is
    naive, and subtracting one from the other raises
    "can't subtract offset-naive and offset-aware datetimes" — which is how
    this first reached a running stack. Everything below goes through here,
    so callers may pass either kind.
    """
    if when is None:
        return None
    if when.tzinfo is None:
        return when.astimezone()
    return when.astimezone()


def describe_gap(previous: datetime | None, now: datetime) -> tuple[str, float]:
    """The gap in the words a person would use, plus its size in hours.

    Returns ("", 0.0) when there is no previous turn. The caller uses the
    hours to decide whether the gap is worth mentioning at all; the string
    is only ever for the prompt.

    Deliberately vague at the long end. "It has been 23 days" is the register
    of a subscription reminder, and being greeted that way by something you
    talk to about your marriage is faintly awful.
    """
    previous, now = _local(previous), _local(now)
    if previous is None or now is None:
        return "", 0.0

    delta = now - previous
    hours = delta.total_seconds() / 3600.0
    if hours < 0:
        return "", 0.0

    minutes = delta.total_seconds() / 60.0
    if minutes < 15:
        return "a few minutes ago — this is a continuation of the same conversation", hours
    if minutes < 90:
        return "under an hour ago, in the same sitting", hours
    if hours < 6 and previous.date() == now.date():
        return f"earlier today, about {int(round(hours))} hours ago", hours

    days = (now.date() - previous.date()).days
    if days == 0:
        return f"earlier today, about {int(round(hours))} hours ago", hours
    if days == 1:
        return f"yesterday {_part_of_day(previous)}", hours
    if days < 7:
        return f"{days} days ago, on {previous.strftime('%A')} {_part_of_day(previous)}", hours
    if days < 14:
        return "about a week ago", hours
    if days < 45:
        return f"about {max(2, round(days / 7))} weeks ago", hours
    return "a long time ago — months, not weeks", hours


def _part_of_day(when: datetime) -> str:
    hour = when.hour
    if hour < 5:
        return "in the small hours"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 22:
        return "evening"
    return "late in the evening"


def now_block(previous: datetime | None, now: datetime | None = None) -> str:
    """The system-prompt section giving the model a clock and a memory of when.

    The guidance matters as much as the facts. A model handed a timestamp
    will use it, and what that looks like unprompted is opening every reply
    with "Good evening! I see it has been 3 days since our last session,"
    which is the voice of a hotel concierge.
    """
    if not get_settings().time_awareness:
        return ""

    now = _local(now) or datetime.now().astimezone()
    lines = [
        "## Now",
        "",
        f"It is {now.strftime('%A %-d %B %Y')}, {now.strftime('%-I:%M%p').lower()}.",
    ]

    phrase, hours = describe_gap(previous, now)
    if not phrase:
        lines.append("This is the first time you have spoken with them.")
    else:
        lines.append(f"You last spoke with them {phrase}.")

    lines += [
        "",
        "Use this the way a person uses it, which is mostly silently. It tells you "
        "whether to pick up mid-thought or to ask how things have been.",
        "",
        "- Under an hour: say nothing about time at all. You are mid-conversation.",
        "- Later the same day: at most a light acknowledgement, if it fits.",
        "- A day or more: it is natural to ask how things have gone since — once, "
        "near the start, in your own words, and then let them steer. If something "
        "was live last time, that is the thing to ask about, not the gap itself.",
        "- Weeks or months: do not remark on the absence, and never imply they "
        "should have come back sooner. Ask what has changed, not where they have been.",
        "",
        "Never open with the date or the elapsed time as a statement. "
        "\"It has been 3 days since our last session\" is the voice of a billing "
        "system. \"How's it been since Sunday?\" is the voice of a person.",
    ]
    return "\n".join(lines)


# ── Ending a session ───────────────────────────────────────────────────

# Matched on the raw user turn, before the model sees it. Deterministic on
# purpose: this triggers a real shutdown, and "did the model think you meant
# to stop?" is not a question worth having in that path.
import re  # noqa: E402

_END_PHRASES = re.compile(
    r"""^\s*(
          let'?s\s+end\s+(for\s+now|here|it\s+here)
        | end\s+(the\s+)?session
        | end\s+for\s+now
        | (i'?m\s+)?done\s+for\s+(now|today|tonight)
        | (that'?s\s+)?(all\s+)?for\s+(now|today|tonight)
        | close\s+(this\s+)?(out|down)
        | shut\s+(it\s+)?down
    )\s*[.!]?\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def is_end_request(text: str) -> bool:
    """Whether this turn is asking to close the session down.

    Anchored to the whole message. "I'm done for now" ends the session;
    "I'm done for now with trying to explain myself to them" very much does
    not, and shutting the machine down mid-sentence on someone would be a
    memorable way to get that wrong.
    """
    return bool(_END_PHRASES.match(text.strip()))


async def session_summary(scope: str, hours: int = 12) -> dict:
    """What happened in this sitting, for the close-out message."""
    try:
        async with memory.get_sessionmaker()() as session:
            row = (
                await session.execute(
                    sql(
                        "SELECT count(*) FILTER (WHERE role = 'user') AS exchanges, "
                        "       min(created_at) AS started "
                        "  FROM turns WHERE scope = :scope AND created_at > :since"
                    ),
                    {
                        "scope": scope,
                        "since": datetime.now().astimezone() - timedelta(hours=hours),
                    },
                )
            ).mappings().one()
        return {"exchanges": row["exchanges"] or 0, "started": row["started"]}
    except Exception as exc:  # noqa: BLE001
        log.warning("session.summary_failed", error=str(exc))
        return {"exchanges": 0, "started": None}


async def close_out(messages: list[ChatMessage], scope: str) -> dict:
    """Finish the session properly, then ask the host to stop the stack.

    Order is the whole point. Everything durable is written and awaited
    BEFORE the shutdown flag appears on disk, because the watcher acts on
    that file within seconds and a `docker compose stop` landing mid-write
    is exactly the way to lose the session you were trying to preserve.

    Worth being clear about what this does and does not rescue: memory
    extraction already runs after every single turn, so nothing is sitting
    unsaved when you say goodbye. What genuinely needs doing here is the
    profile, which is batched every few exchanges and would otherwise end
    the session up to two exchanges out of date.
    """
    s = get_settings()
    summary = await session_summary(scope)
    reply_stub = "(session closed)"

    if s.memory_enabled:
        try:
            await memory.extract_and_store(messages, reply_stub, scope)
        except Exception as exc:  # noqa: BLE001
            log.warning("session.close_extract_failed", error=str(exc))
        if s.profile_enabled:
            try:
                await memory.update_profile(messages, reply_stub, scope, force=True)
            except Exception as exc:  # noqa: BLE001
                log.warning("session.close_profile_failed", error=str(exc))

    await log_turn(scope, "system", "session ended by request")

    shutdown_requested = False
    if s.shutdown_on_end:
        shutdown_requested = _request_shutdown()

    log.info(
        "session.closed",
        exchanges=summary["exchanges"],
        shutdown_requested=shutdown_requested,
        scope=scope,
    )
    return {**summary, "shutdown_requested": shutdown_requested}


def _request_shutdown() -> bool:
    """Leave the flag the host-side watcher is looking for.

    A file, not a Docker call. Stopping the stack means talking to the
    Docker daemon, and the only way to do that from in here is to mount
    /var/run/docker.sock — which would hand root-equivalent control of the
    machine to the one container that also holds every word of these
    conversations. For a stack whose backend network is `internal: true`
    specifically so that content cannot leave, that is the wrong trade.

    So the container writes a file and the host decides what to do about
    it. See `make watch`.
    """
    from pathlib import Path

    try:
        flag = Path(get_settings().shutdown_flag_path)
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text(
            datetime.now().astimezone().isoformat(timespec="seconds") + "\n", encoding="utf-8"
        )
        return True
    except Exception as exc:  # noqa: BLE001
        # Not fatal. The session is already saved; only the automatic stop
        # is lost, and `make end` does the same job by hand.
        log.warning("session.shutdown_flag_failed", error=str(exc))
        return False


def farewell(summary: dict) -> str:
    """The closing message. Deterministic — the model is not consulted.

    It is making a factual claim about what was saved and whether the
    machine is about to stop, and neither of those should be subject to a
    27B's mood. It also has to be instant: the stack is about to go down,
    and waiting a minute to generate a goodbye is a poor last impression.
    """
    exchanges = summary.get("exchanges") or 0
    if exchanges > 1:
        opening = f"Alright — that's {exchanges} exchanges saved."
    else:
        opening = "Alright — saved."

    lines = [
        opening,
        "",
        "Your notes are up to date, and anything worth carrying is in memory. "
        "When you come back I'll know how long it's been.",
    ]
    if summary.get("shutdown_requested"):
        lines += ["", "_Shutting the stack down now. Start it again with_ `make up`."]
    else:
        lines += ["", "_Run_ `make end` _when you want to stop the stack._"]
    return "\n".join(lines)
