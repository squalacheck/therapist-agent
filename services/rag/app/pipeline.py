"""Per-turn orchestration.

Order matters:

  1. safety      first, on the raw turn, before anything else spends time
  2. retrieval   query rewrite -> hybrid search -> rerank
  3. memory      recall durable facts about this person
  4. assemble    system prompt + memory + passages + budgeted history
  5. stream      the response
  6. extract     new memory, after the fact, off the critical path

Steps 2 and 3 both need the rewritten query, so they run concurrently once
it exists. Every stage degrades to a no-op on failure — a broken reranker
should cost you citations, not the conversation.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime

import structlog

from . import daily, feedback, memory, retrieval, safety, session
from .config import get_settings, load_prompt
from .llm import get_llm
from .schemas import ChatMessage, Passage, SafetyVerdict

log = structlog.get_logger(__name__)

# Rough enough. Used only for trimming history to a budget, where being
# 15% out costs nothing and importing a tokeniser costs a dependency.
_CHARS_PER_TOKEN = 3.6


def _approx_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN)


def _trim_history(messages: list[ChatMessage], budget: int) -> list[ChatMessage]:
    """Keep the most recent turns that fit, oldest dropped first.

    The current user turn is always kept, however long it is — truncating
    what someone just said is the one thing that is never acceptable here.
    """
    turns = [m for m in messages if m.role in ("user", "assistant")]
    if not turns:
        return []

    kept: list[ChatMessage] = [turns[-1]]
    used = _approx_tokens(turns[-1].text())
    for message in reversed(turns[:-1]):
        cost = _approx_tokens(message.text())
        if used + cost > budget:
            break
        kept.insert(0, message)
        used += cost
    return kept


def _phase(history: list[ChatMessage]) -> str:
    """Where this conversation is: 'opening' or 'working'.

    A turn count, not a clock — a session is however long it is, and wall
    time says nothing about whether the picture is complete. Crude on
    purpose: the aim is to stop the model reaching for a fix on turn one,
    and a counter does that without pretending to a judgement it cannot
    make. It resets with each new chat, which is the correct behaviour;
    what carries across conversations is the profile, not the pacing.
    """
    user_turns = sum(1 for m in history if m.role == "user")
    return "opening" if user_turns <= get_settings().pacing_opening_turns else "working"


def _assemble(
    history: list[ChatMessage],
    passages: list[Passage],
    facts: list[memory.MemoryFact],
    verdict: SafetyVerdict,
    profile: str = "",
    phase: str = "",
    commitments: list[dict] | None = None,
    now_block: str = "",
    first_meeting: bool = False,
    feedback_note: str = "",
) -> list[dict[str, str]]:
    system_parts = [load_prompt("system")]

    # Early, and before the stance detail. What day it is and when you last
    # spoke frames everything that follows it.
    if now_block:
        system_parts.append(now_block)

    if directive := safety.directive(verdict):
        system_parts.append(directive)

    # Pacing is suppressed on a safety turn. The directive above is already
    # telling it exactly how to handle this turn, and "spend longer
    # understanding before you act" is the wrong instruction to add when
    # someone has just said they are not safe.
    if phase and not verdict.triggered:
        system_parts.append(load_prompt(f"pacing_{phase}"))

    # Nothing remembered at all: a fresh install, or memory that has been
    # wiped. Say so plainly, or the model fills the gap by acting as if it
    # already knows them — which, with no notes, means inventing someone.
    if first_meeting:
        system_parts.append(load_prompt("first_meeting"))

    if profile:
        system_parts.append(
            "## Where things stand with the person you are talking with\n\n"
            "Your running notes, carried across every conversation you have had "
            "with them. This is context, not a script: do not recite it back, do "
            "not open by summarising it, and treat anything they say now as more "
            "current than anything written here.\n\n" + profile
        )

    # How they rated the last conversation — only present when it went
    # badly somewhere. After the notes, so it reads as "and here is what to
    # do better with this person", not as the first thing about them.
    if feedback_note:
        system_parts.append(feedback_note)

    if facts:
        system_parts.append(
            "## What you already know about the person you are talking with\n\n"
            "Carried over from earlier conversations. Use it where it changes what "
            "you say; do not recite it back at them, and do not treat it as more "
            "current than what they tell you now.\n\n" + memory.format_memory(facts)
        )

    if commitments:
        system_parts.append(
            "## What they said they would do today\n\n"
            "From today's practice list, still outstanding. Do not open by "
            "reciting these or asking them to report on each one — that turns a "
            "conversation into a stand-up meeting. Use them the way you would "
            "use anything else you remember: if what they are talking about "
            "touches one, you already know it is there.\n\n"
            + daily.format_open_items(commitments)
        )

    if passages:
        system_parts.append(
            "## Retrieved research\n\n"
            "Passages from the corpus, most relevant first. When you draw on one, "
            "cite it inline as [1], [2] and so on. If these do not actually address "
            "the question, say so plainly rather than stretching them to fit — and "
            "never attribute a claim to a source that does not make it.\n\n"
            + retrieval.format_passages(passages)
        )

    prompt: list[dict[str, str]] = [{"role": "system", "content": "\n\n".join(system_parts)}]
    prompt.extend({"role": m.role, "content": m.text()} for m in history)
    return prompt


async def run(
    messages: list[ChatMessage],
    *,
    scope: str = memory.GLOBAL_SCOPE,
    remember: bool = True,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> AsyncIterator[str]:
    """Run one turn, yielding response text as it is generated."""
    s = get_settings()
    last_user = next((m.text() for m in reversed(messages) if m.role == "user"), "")

    # 0. When did we last speak? Read BEFORE this turn is logged, or the
    # gap is always zero and the whole feature quietly does nothing.
    previous = await session.last_turn_at(scope) if remember else None
    now_block = session.now_block(previous) if remember else ""

    # 1. Safety, first and on the raw text.
    verdict = await safety.assess(last_user)

    # 2 + 3. Retrieval and memory, concurrently.
    passages: list[Passage] = []
    facts: list[memory.MemoryFact] = []
    profile = ""

    # The running notes need no query, so they are fetched regardless of
    # what was said — including on a first turn of "hey", which is exactly
    # the case semantic recall cannot serve.
    if s.memory_enabled and s.profile_enabled and remember:
        profile = await memory.get_profile(scope)

    feedback_note = ""
    if feedback.enabled_for(remember):
        feedback_note = feedback.prompt_block(await feedback.latest(scope))

    # What is still open on today's practice list. Cheap, and it is the
    # thing that makes the two halves of this one tool: without it you can
    # commit to something in the morning list and then talk for an hour to
    # an agent with no idea you did.
    commitments: list[dict] = []
    if s.daily_enabled and s.memory_enabled and remember:
        try:
            commitments = await daily.open_items(scope)
        except Exception as exc:  # noqa: BLE001
            log.warning("daily.open_items_failed", error=str(exc))

    if s.retrieval_enabled or (s.memory_enabled and remember):
        query = await retrieval.build_query(messages) if s.retrieval_enabled else last_user

        tasks = []
        tasks.append(retrieval.search(query) if s.retrieval_enabled else _empty())
        tasks.append(
            memory.recall(query, scope) if (s.memory_enabled and remember) else _empty()
        )
        passages, facts = await asyncio.gather(*tasks)

    # 4. Assemble.
    history = _trim_history(messages, s.max_history_tokens)
    phase = _phase(history) if s.pacing_enabled else ""
    first_meeting = s.memory_enabled and remember and not profile and not facts
    prompt = _assemble(
        history, passages, facts, verdict, profile, phase, commitments, now_block,
        first_meeting=first_meeting,
        feedback_note=feedback_note,
    )

    if remember:
        await session.log_turn(
            scope,
            "user",
            last_user,
            safety_category=verdict.category if verdict.triggered else None,
        )

    log.info(
        "turn.start",
        history_turns=len(history),
        passages=len(passages),
        facts=len(facts),
        profile_chars=len(profile),
        commitments=len(commitments),
        phase=phase or None,
        first_meeting=first_meeting,
        feedback_note=bool(feedback_note),
        since_last=session.describe_gap(previous, datetime.now().astimezone())[0] or None,
        remembering=remember,
        safety=verdict.category if verdict.triggered else None,
    )

    # 5. Stream.
    collected: list[str] = []
    # Rolling window for the output screen. Screening only the newest delta
    # would miss any phrase that straddles two of them, and vLLM emits one
    # token at a time — so essentially every phrase straddles. 600 chars is
    # comfortably longer than the longest pattern in safety._OUTPUT_BLOCK.
    window = ""
    blocked = False

    async for delta in get_llm().stream(
        prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p
    ):
        window = (window + delta)[-600:]
        if safety.screen_output(window):
            # Deterministic override: stop generating, replace rather than
            # append, and let the model have no say in it. This is the one
            # place in the pipeline where code overrules the model outright.
            blocked = True
            collected.append(safety.OUTPUT_BLOCKED_MESSAGE)
            yield safety.OUTPUT_BLOCKED_MESSAGE
            break
        collected.append(delta)
        yield delta

    reply = "".join(collected)

    if blocked:
        log.error("turn.output_blocked", scope=scope)
        # Resources go out regardless of what the input classifier decided.
        # If the reply drifted here, the conversation is somewhere that
        # warrants them whatever the opening turn looked like.
        if not verdict.triggered:
            verdict = SafetyVerdict(
                triggered=True, category="output_screen", confidence=1.0
            )

    # Citations and resources are appended after the model finishes, not
    # left to the model to remember. It forgets; this does not.
    if passages:
        tail = retrieval.format_sources(passages)
        collected.append(tail)
        yield tail

    if resources := safety.resources_block(verdict):
        collected.append(resources)
        yield resources

    # 6. Memory extraction and profile update, off the critical path.
    if s.memory_enabled and remember and reply.strip():
        async def _remember() -> None:
            await session.log_turn(scope, "assistant", reply)
            await memory.extract_and_store(messages, reply, scope)
            if s.profile_enabled:
                await memory.update_profile(messages, reply, scope)
            if s.daily_enabled:
                # Did they just report doing one of today's items? Same
                # place as memory extraction, for the same reason: it needs
                # the finished reply and must not cost the user latency.
                await daily.review_exchange(messages, reply, scope)

        if s.memory_extract_async:
            # Held in a set: asyncio only keeps a weak reference to a bare
            # create_task, so it can be garbage-collected mid-flight and the
            # write silently never happens.
            task = asyncio.create_task(_remember())
            _background.add(task)
            task.add_done_callback(_on_background_done)
        else:
            await _remember()


async def _empty() -> list:
    return []


# Strong references to in-flight background writes. See the comment above.
_background: set[asyncio.Task] = set()


def _on_background_done(task: asyncio.Task) -> None:
    """Log what a background write did, including when it blew up.

    Without this an exception in memory extraction is stored on the Task
    object and never retrieved, so it vanishes: no traceback, no log line,
    and memory just quietly stops being written while the chat carries on
    looking perfectly healthy.
    """
    _background.discard(task)
    if task.cancelled():
        log.warning("memory.background_cancelled")
        return
    if exc := task.exception():
        log.error("memory.background_failed", error=repr(exc), exc_info=exc)


# ── The daily practice turn ────────────────────────────────────────────


def _daily_heading(plan: dict, generated: bool) -> str:
    from datetime import date

    when = date.fromisoformat(plan["date"])
    if generated:
        return f"Here's today — {when.strftime('%A %-d %B')}."
    done = sum(1 for i in plan["items"] if i["status"] == "done")
    total = len(plan["items"])
    if done == total:
        return f"All {total} done today. Worth noticing."
    if done:
        return f"Today so far — {done} of {total} done."
    return f"Today — {when.strftime('%A %-d %B')}."


async def run_daily(
    messages: list[ChatMessage],
    *,
    scope: str = memory.GLOBAL_SCOPE,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> AsyncIterator[str]:
    """The `therapist-daily` turn.

    Opening the chat shows the list. Everything after that is an ordinary
    conversation that happens to have the list in front of it — which is
    what makes "I tried the second one and it went badly" a usable thing to
    say, rather than something you have to file somewhere.

    Rendering is deterministic from what was stored rather than streamed as
    prose. The model supplies the judgement about what belongs on the list;
    this file supplies the shape, so the format cannot drift.
    """
    s = get_settings()
    user_turns = sum(1 for m in messages if m.role == "user")

    plan, generated = await daily.ensure_plan(scope)
    if not plan:
        yield (
            "I couldn't put today's list together — that usually means the model "
            "is still loading. Nothing is lost; ask again in a minute."
        )
        return

    # Opening the list, rather than talking about it.
    if generated or user_turns <= 1:
        yield daily.render(plan, heading=_daily_heading(plan, generated))
        return

    # Talking about it. Safety still runs first, on the raw turn.
    last_user = next((m.text() for m in reversed(messages) if m.role == "user"), "")
    verdict = await safety.assess(last_user)

    profile = await memory.get_profile(scope) if s.memory_enabled else ""
    history = _trim_history(messages, s.max_history_tokens)

    previous = await session.last_turn_at(scope)
    prompt = _assemble(
        history, [], [], verdict, profile, "", None, session.now_block(previous)
    )
    prompt[0]["content"] += (
        "\n\n## Today's practice list\n\n"
        "This is what today's list actually says. Talk about it the way you would "
        "talk about anything else — do not re-print it unless they ask, and do not "
        "ask them to account for every line. If they say something went badly, that "
        "is the interesting part of the day, not a failure to be smoothed over.\n\n"
        + daily.render(plan)
    )

    await session.log_turn(scope, "user", last_user)
    log.info("daily.turn", items=len(plan["items"]), history_turns=len(history))

    collected: list[str] = []
    window = ""
    async for delta in get_llm().stream(
        prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p
    ):
        window = (window + delta)[-600:]
        if safety.screen_output(window):
            collected.append(safety.OUTPUT_BLOCKED_MESSAGE)
            yield safety.OUTPUT_BLOCKED_MESSAGE
            verdict = SafetyVerdict(triggered=True, category="output_screen", confidence=1.0)
            break
        collected.append(delta)
        yield delta

    if resources := safety.resources_block(verdict):
        yield resources

    reply = "".join(collected)
    if reply.strip():
        async def _followup() -> None:
            await session.log_turn(scope, "assistant", reply)
            await daily.review_exchange(messages, reply, scope)
            if s.memory_enabled:
                await memory.extract_and_store(messages, reply, scope)

        task = asyncio.create_task(_followup())
        _background.add(task)
        task.add_done_callback(_on_background_done)
