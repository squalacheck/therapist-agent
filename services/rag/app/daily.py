"""Daily practice — the between-session layer.

A conversation you have and then close is not therapy; it is a good talk.
What makes the difference over months is what happens between sittings, and
that is the thing this stack had no representation of at all. Memory
remembers what was said. This remembers what was *agreed to*, checks whether
it happened, and lets the next day's suggestions respond to the answer.

Three or four items a day, no more. A list long enough to feel like a
programme is a list that gets skipped entirely, and a skipped list teaches
the person that this tool asks for things they do not do.

Design notes worth keeping:

- Generation is a blocking JSON call, and rendering is deterministic from
  what was stored. The alternative — streaming the plan as prose and parsing
  it back — puts the format on the model, which drifts. Here the model
  supplies judgement and this file supplies shape.
- Items are never invented fresh each morning. Yesterday's outcomes are in
  the prompt, so "you said you'd bring up the weekend plans and didn't"
  is available to it, and repeating an untouched item four days running is
  a decision it makes on purpose rather than by amnesia.
- Nothing here may require the other person to cooperate for the item to
  count. "Have a calm conversation about money" is not an assignment, it is
  a wish; "say the one sentence you rehearsed, then stop talking" is.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, timedelta
from typing import Any

import structlog
from sqlalchemy import text as sql

from . import memory
from .config import get_settings, load_prompt
from .llm import get_llm
from .schemas import ChatMessage

log = structlog.get_logger(__name__)

DOMAINS = ("self", "relationship", "communication")
STATUSES = ("open", "done", "skipped")

# The DDL lives here rather than in infra/postgres/init because those files
# run exactly once, when the Postgres volume is first created. This feature
# arrived after the database already had months of conversation in it, so a
# file there would never have executed. Idempotent, applied at startup.
_DDL = (
    """
    CREATE TABLE IF NOT EXISTS daily_plan (
        id         UUID PRIMARY KEY,
        scope      TEXT        NOT NULL DEFAULT 'global',
        plan_date  DATE        NOT NULL,
        intro      TEXT        NOT NULL DEFAULT '',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (scope, plan_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_item (
        id         UUID PRIMARY KEY,
        plan_id    UUID NOT NULL REFERENCES daily_plan(id) ON DELETE CASCADE,
        position   INTEGER     NOT NULL DEFAULT 0,
        domain     TEXT        NOT NULL DEFAULT 'self',
        title      TEXT        NOT NULL,
        detail     TEXT        NOT NULL DEFAULT '',
        frame      TEXT        NOT NULL DEFAULT '',
        status     TEXT        NOT NULL DEFAULT 'open',
        outcome    TEXT        NOT NULL DEFAULT '',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS daily_item_plan_idx ON daily_item (plan_id, position)",
    "CREATE INDEX IF NOT EXISTS daily_plan_scope_date_idx ON daily_plan (scope, plan_date DESC)",
)


async def init_schema() -> None:
    if not get_settings().daily_enabled:
        return
    async with memory.get_sessionmaker()() as session:
        for statement in _DDL:
            await session.execute(sql(statement))
        await session.commit()
    log.info("daily.schema_ready")


def today() -> date:
    """Local date, not UTC.

    The rag container runs with TZ set from .env. Without it a plan
    generated in the evening can land on tomorrow's date, and the morning
    check-in finds yesterday's list already marked as today's.
    """
    return date.today()


# ── Reading ────────────────────────────────────────────────────────────


async def get_plan(scope: str, day: date | None = None) -> dict[str, Any] | None:
    """One day's plan with its items, or None if it was never generated."""
    day = day or today()
    async with memory.get_sessionmaker()() as session:
        plan = (
            await session.execute(
                sql(
                    "SELECT id, plan_date, intro FROM daily_plan "
                    "WHERE scope = :scope AND plan_date = :day"
                ),
                {"scope": scope, "day": day},
            )
        ).mappings().one_or_none()
        if not plan:
            return None
        items = (
            await session.execute(
                sql(
                    "SELECT id, position, domain, title, detail, frame, status, outcome "
                    "FROM daily_item WHERE plan_id = :pid ORDER BY position"
                ),
                {"pid": plan["id"]},
            )
        ).mappings().all()

    return {
        "id": str(plan["id"]),
        "date": plan["plan_date"].isoformat(),
        "intro": plan["intro"],
        "items": [
            {
                "id": str(i["id"]),
                "position": i["position"],
                "domain": i["domain"],
                "title": i["title"],
                "detail": i["detail"],
                "frame": i["frame"],
                "status": i["status"],
                "outcome": i["outcome"],
            }
            for i in items
        ],
    }


async def recent_plans(scope: str, days: int = 5, before: date | None = None) -> list[dict]:
    """The last few days, for the generator to react to."""
    before = before or today()
    async with memory.get_sessionmaker()() as session:
        rows = (
            await session.execute(
                sql(
                    """
                    SELECT p.plan_date, i.domain, i.title, i.status, i.outcome
                      FROM daily_plan p JOIN daily_item i ON i.plan_id = p.id
                     WHERE p.scope = :scope
                       AND p.plan_date < :before
                       AND p.plan_date >= :since
                     ORDER BY p.plan_date DESC, i.position
                    """
                ),
                {"scope": scope, "before": before, "since": before - timedelta(days=days)},
            )
        ).mappings().all()
    return [dict(r) | {"plan_date": r["plan_date"].isoformat()} for r in rows]


async def open_items(scope: str, day: date | None = None) -> list[dict]:
    """Today's still-open items — what the main agent should know about.

    This is the bit that makes the two halves one tool rather than two.
    Without it you can commit to something in the daily list and then have
    an hour-long conversation with an agent that has no idea you did.
    """
    plan = await get_plan(scope, day)
    if not plan:
        return []
    return [i for i in plan["items"] if i["status"] == "open"]


# ── Writing ────────────────────────────────────────────────────────────


async def set_status(item_id: str, status: str, outcome: str = "") -> bool:
    if status not in STATUSES:
        return False
    async with memory.get_sessionmaker()() as session:
        result = await session.execute(
            sql(
                "UPDATE daily_item SET status = :status, "
                "       outcome = CASE WHEN :outcome = '' THEN outcome ELSE :outcome END, "
                "       updated_at = now() "
                " WHERE id = :id"
            ),
            {"id": item_id, "status": status, "outcome": outcome},
        )
        await session.commit()
    return result.rowcount > 0


async def _store(scope: str, day: date, intro: str, items: list[dict]) -> dict[str, Any]:
    plan_id = str(uuid.uuid4())
    async with memory.get_sessionmaker()() as session:
        # Regenerating a day replaces it. ON DELETE CASCADE takes the items.
        await session.execute(
            sql("DELETE FROM daily_plan WHERE scope = :scope AND plan_date = :day"),
            {"scope": scope, "day": day},
        )
        await session.execute(
            sql(
                "INSERT INTO daily_plan (id, scope, plan_date, intro) "
                "VALUES (:id, :scope, :day, :intro)"
            ),
            {"id": plan_id, "scope": scope, "day": day, "intro": intro},
        )
        for position, item in enumerate(items):
            await session.execute(
                sql(
                    "INSERT INTO daily_item "
                    "  (id, plan_id, position, domain, title, detail, frame) "
                    "VALUES (:id, :plan_id, :position, :domain, :title, :detail, :frame)"
                ),
                {
                    "id": str(uuid.uuid4()),
                    "plan_id": plan_id,
                    "position": position,
                    "domain": item["domain"],
                    "title": item["title"],
                    "detail": item["detail"],
                    "frame": item["frame"],
                },
            )
        await session.commit()
    log.info("daily.plan_written", day=day.isoformat(), items=len(items), scope=scope)
    return await get_plan(scope, day)


def _clean_items(raw: Any, limit: int) -> list[dict]:
    """Coerce whatever the model returned into storable items.

    Anything without a title is dropped rather than stored blank: an empty
    row in a to-do list is worse than a shorter list.
    """
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        if len(title) < 4:
            continue
        domain = str(entry.get("domain") or "self").strip().lower()
        out.append(
            {
                "domain": domain if domain in DOMAINS else "self",
                "title": title,
                "detail": str(entry.get("detail") or "").strip(),
                "frame": str(entry.get("frame") or "").strip(),
            }
        )
        if len(out) >= limit:
            break
    return out


async def generate(scope: str, day: date | None = None) -> dict[str, Any] | None:
    """Build a day's plan from the profile, memory, history and research."""
    s = get_settings()
    day = day or today()

    profile = await memory.get_profile(scope) if s.memory_enabled else ""
    history = await recent_plans(scope, days=5, before=day)

    # Facts, biased toward the categories that actually generate actions.
    # A recalled fact about someone's job title does not suggest anything
    # to do; a goal or a recurring pattern does.
    facts = []
    if s.memory_enabled:
        seed = "goals, recurring patterns, and the relationship"
        if s.partner_name:
            seed += f" with {s.partner_name}"
        facts = await memory.recall(seed, scope)

    passages = []
    if s.retrieval_enabled:
        try:
            from . import retrieval

            query = "concrete practices for repair, bids, and communication in couples"
            if profile:
                query += " — " + profile[:400]
            passages = (await retrieval.search(query))[: s.daily_passages]
        except Exception as exc:  # noqa: BLE001
            log.warning("daily.retrieval_failed", error=str(exc))

    context = [f"Today is {day.strftime('%A, %-d %B %Y')}."]
    if s.partner_name:
        context.append(f"Their partner is called {s.partner_name}.")
    if profile:
        context.append(f"Running notes on this person:\n{profile}")
    else:
        context.append("Running notes on this person: none yet — you are only getting to know them.")
    if facts:
        context.append(f"Things known about them:\n{memory.format_memory(facts)}")
    if history:
        lines = [
            f"- {h['plan_date']} [{h['domain']}] {h['title']} — {h['status']}"
            + (f" ({h['outcome']})" if h["outcome"] else "")
            for h in history
        ]
        context.append(
            "What was suggested on previous days, and what came of it:\n" + "\n".join(lines)
        )
    else:
        context.append("There is no previous day to react to — this is the first list.")
    if passages:
        from . import retrieval

        context.append(
            "Relevant research passages, if any of them genuinely fit:\n"
            + retrieval.format_passages(passages)
        )

    try:
        result = await get_llm().complete_json(
            [
                {"role": "system", "content": load_prompt("daily_plan")},
                {"role": "user", "content": "\n\n".join(context)},
            ],
            max_tokens=s.daily_max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("daily.generate_failed", error=str(exc))
        return None

    if not result:
        log.warning("daily.generate_unparseable")
        return None

    items = _clean_items(result.get("items"), s.daily_max_items)
    if not items:
        log.warning("daily.generate_no_items")
        return None

    return await _store(scope, day, str(result.get("intro") or "").strip(), items)


async def ensure_plan(scope: str, day: date | None = None) -> tuple[dict | None, bool]:
    """Today's plan, generating it if this is the first ask of the day.

    Returns (plan, was_generated) so the caller can tell a fresh list from
    one being looked at again.
    """
    day = day or today()
    if existing := await get_plan(scope, day):
        return existing, False
    return await generate(scope, day), True


# ── Follow-through ─────────────────────────────────────────────────────


async def review_exchange(messages: list[ChatMessage], reply: str, scope: str) -> None:
    """Update item statuses from what was just said. Runs after streaming.

    Deliberately conservative: the model is told to return nothing unless
    the person actually said what happened. Marking something done because
    the conversation drifted near it would make the record worthless, and
    the record is the only reason any of this beats a notes app.
    """
    plan = await get_plan(scope)
    if not plan or not plan["items"]:
        return

    last_user = next((m.text() for m in reversed(messages) if m.role == "user"), "")
    if not last_user:
        return

    roster = "\n".join(
        f"{i['position']}. [{i['status']}] {i['title']}" for i in plan["items"]
    )
    try:
        result = await get_llm().complete_json(
            [
                {"role": "system", "content": load_prompt("daily_review")},
                {
                    "role": "user",
                    "content": (
                        f"Today's items:\n{roster}\n\n"
                        f"They said:\n{last_user}\n\n"
                        f"The reply was:\n{reply[:1200]}"
                    ),
                },
            ],
            max_tokens=400,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("daily.review_failed", error=str(exc))
        return

    updates = (result or {}).get("updates")
    if not isinstance(updates, list) or not updates:
        return

    by_position = {i["position"]: i for i in plan["items"]}
    changed = 0
    for update in updates[: len(plan["items"])]:
        if not isinstance(update, dict):
            continue
        try:
            position = int(update.get("position"))
        except (TypeError, ValueError):
            continue
        item = by_position.get(position)
        status = str(update.get("status") or "").strip().lower()
        if not item or status not in ("done", "skipped"):
            continue
        if await set_status(item["id"], status, str(update.get("outcome") or "").strip()):
            changed += 1

    if changed:
        log.info("daily.reviewed", updated=changed, scope=scope)


# ── Rendering ──────────────────────────────────────────────────────────

_MARK = {"open": "☐", "done": "☑", "skipped": "—"}
_LABEL = {"self": "you", "relationship": "the relationship", "communication": "how you say it"}


def render(plan: dict[str, Any], *, heading: str | None = None) -> str:
    """Markdown for the chat.

    The one place in this stack that is allowed a list, because here the
    content genuinely is enumerable and the whole point is that it can be
    read at a glance and worked through.
    """
    if not plan or not plan["items"]:
        return "I couldn't put a list together just now. Try asking again in a moment."

    lines: list[str] = []
    if heading:
        lines.append(heading)
    if plan.get("intro"):
        lines.append(plan["intro"])
    lines.append("")

    for item in plan["items"]:
        mark = _MARK.get(item["status"], "☐")
        where = _LABEL.get(item["domain"], item["domain"])
        lines.append(f"{mark} **{item['title']}**  ·  _{where}_")
        if item["detail"]:
            lines.append(f"   {item['detail']}")
        if item["frame"]:
            lines.append(f"   _{item['frame']}_")
        if item["outcome"]:
            lines.append(f"   → {item['outcome']}")
        lines.append("")

    return "\n".join(lines).rstrip()


def format_open_items(items: list[dict]) -> str:
    """Compact form for injection into the main agent's system prompt."""
    return "\n".join(f"- [{i['domain']}] {i['title']}" for i in items)
