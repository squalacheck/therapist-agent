"""Unit tests for the parts that do not need a model or a database."""

from __future__ import annotations

from app.pipeline import _approx_tokens, _trim_history
from app.retrieval import format_passages, format_sources
from app.safety import prefilter
from app.schemas import ChatMessage, Passage


# ── History trimming ───────────────────────────────────────────────


def test_trim_keeps_current_turn_even_when_oversized():
    long_turn = ChatMessage(role="user", content="x" * 100_000)
    kept = _trim_history([long_turn], budget=100)
    assert kept == [long_turn], "the current user turn must never be dropped"


def test_trim_drops_oldest_first():
    messages = [
        ChatMessage(role="user", content="a" * 4000),
        ChatMessage(role="assistant", content="b" * 4000),
        ChatMessage(role="user", content="c" * 400),
    ]
    kept = _trim_history(messages, budget=_approx_tokens("c" * 400) + 50)
    assert len(kept) == 1
    assert kept[0].content.startswith("c")


def test_trim_ignores_system_messages():
    messages = [
        ChatMessage(role="system", content="injected by the frontend"),
        ChatMessage(role="user", content="hello"),
    ]
    assert all(m.role != "system" for m in _trim_history(messages, budget=1000))


# ── Safety prefilter ───────────────────────────────────────────────


def test_prefilter_catches_crisis_language():
    assert prefilter("I've been thinking about killing myself")
    assert prefilter("my partner choked me an hour ago")
    assert prefilter("I started cutting myself again")


def test_prefilter_is_broad_by_design():
    """The prefilter decides whether a classifier call is worth making, and
    nothing else.

    This test used to assert the opposite — that phrases like "I can't do
    this anymore" stay clear of it. That was the original contract, and it
    scored 1/6 on eval/safety.yaml, because five of six real crisis cases
    never reached the classifier at all. The filter was widened on purpose
    and specificity moved to the classifier, which is the layer that can
    tell "I can't do this anymore" the exhausted from "I can't do this
    anymore" the goodbye. Regex cannot, and asking it to is what produced
    the misses.

    So these SHOULD trip the prefilter. Specificity is asserted end to end
    by `make eval-safety`, against a real model.
    """
    for text in (
        "I can't do this anymore, every visit home ends the same way",
        "my grandmother died last autumn and some days I don't see the point",
    ):
        assert prefilter(text), f"prefilter deliberately catches: {text}"


def test_prefilter_still_ignores_plain_conflict_talk():
    """Cheapness is the point — most turns must cost nothing."""
    for text in (
        "this deadline is killing me",
        "when I bring up the holidays they say I'm attacking them",
        "we argued about the dishes again and I shut down",
    ):
        assert not prefilter(text), f"prefilter should ignore: {text}"


# ── Message flattening ─────────────────────────────────────────────


def test_message_text_flattens_multipart_content():
    message = ChatMessage(
        role="user",
        content=[
            {"type": "text", "text": "first"},
            {"type": "image_url", "image_url": {"url": "..."}},
            {"type": "text", "text": "second"},
        ],
    )
    assert message.text() == "first\nsecond"


def test_message_text_handles_none():
    assert ChatMessage(role="assistant", content=None).text() == ""


# ── Citations ──────────────────────────────────────────────────────


def _passage(**kwargs) -> Passage:
    return Passage(chunk_id="c1", text="Contempt predicted dissolution.", score=0.9, **kwargs)


def test_citation_includes_doi_when_present():
    citation = _passage(
        authors="Gottman, John; Levenson, Robert",
        year=1992,
        title="Marital processes predictive of later dissolution",
        journal="J Pers Soc Psychol",
        doi="10.1037/0022-3514.63.2.221",
    ).citation()
    assert "Gottman" in citation
    assert "(1992)" in citation
    assert "10.1037/0022-3514.63.2.221" in citation


def test_citation_degrades_to_chunk_id():
    assert Passage(chunk_id="abc123", text="t", score=0.1).citation() == "abc123"


def test_passages_are_numbered_for_citation():
    rendered = format_passages([_passage(title="One"), _passage(title="Two")])
    assert rendered.startswith("[1]")
    assert "[2]" in rendered


def test_empty_passages_render_to_nothing():
    assert format_passages([]) == ""
    assert format_sources([]) == ""


# ── Output screening ───────────────────────────────────────────────
#
# Both directions matter and the second one matters more. A screen that
# fires on ordinary trauma conversation gets switched off within a
# fortnight, and a switched-off screen protects nothing.


def test_output_screen_blocks_method_and_lethality():
    from app.safety import screen_output

    for text in (
        "The lethal dose of that is around",
        "Taking 40 tablets would be fatal in most cases",
        "the most effective way to kill yourself is",
    ):
        assert screen_output(text), f"should have blocked: {text!r}"


def test_output_screen_passes_ordinary_therapeutic_prose():
    from app.safety import screen_output

    for text in (
        "You said you overdosed years ago and have never told anyone.",
        "It sounds like you don't want to be here anymore. Can we stay with that?",
        "They grabbed you by the throat. That is violence, whatever else was going on.",
        "They said they'd kill themselves if you left. That is a terrible thing to be handed.",
        "Contempt is the strongest single predictor of divorce in Gottman's work.",
    ):
        assert not screen_output(text), f"false positive on: {text!r}"


# ── Session pacing ─────────────────────────────────────────────────


def test_phase_opens_before_it_works():
    from app.pipeline import _phase

    opening = [ChatMessage(role="user", content="hi")]
    assert _phase(opening) == "opening"

    long_session = []
    for _ in range(6):
        long_session.append(ChatMessage(role="user", content="..."))
        long_session.append(ChatMessage(role="assistant", content="..."))
    assert _phase(long_session) == "working"


# ── Daily practice ─────────────────────────────────────────────────


def test_daily_items_are_cleaned_and_capped():
    from app.daily import _clean_items

    raw = [
        {"domain": "relationship", "title": "Ask how their week actually went", "detail": "d", "frame": "f"},
        {"domain": "nonsense", "title": "Walk before breakfast"},
        {"title": "no"},           # too short, dropped
        "not a dict",              # wrong type, dropped
        {"domain": "self", "title": "Phone in the other room at dinner"},
        {"domain": "self", "title": "One more that exceeds the cap"},
    ]
    items = _clean_items(raw, limit=3)
    assert len(items) == 3, "the cap must hold"
    assert items[1]["domain"] == "self", "an unknown domain falls back rather than storing junk"
    assert all(len(i["title"]) >= 4 for i in items)
    assert all(set(i) == {"domain", "title", "detail", "frame"} for i in items)


def test_daily_render_marks_status_and_survives_empty():
    from app.daily import render

    plan = {
        "date": "2026-09-09",
        "intro": "Short one today.",
        "items": [
            {"domain": "self", "title": "Walk", "detail": "Before coffee.",
             "frame": "", "status": "done", "outcome": "did it"},
            {"domain": "communication", "title": "Say the one sentence", "detail": "",
             "frame": "DEAR MAN", "status": "open", "outcome": ""},
        ],
    }
    out = render(plan, heading="Today.")
    assert "☑" in out and "☐" in out
    assert "Say the one sentence" in out
    assert "did it" in out

    assert "couldn't put a list together" in render({"items": []})


# ── Passage diversity ──────────────────────────────────────────────


def _p(chunk: str, doc: str, score: float) -> Passage:
    return Passage(chunk_id=chunk, doc_id=doc, text="…", score=score, rerank_score=score)


def test_diversify_caps_one_document_from_taking_every_slot():
    from app.retrieval import _diversify

    # The real failure: one paper reranked into seven of eight slots.
    ranked = [_p(f"c{i}", "gottman-pdf", 9 - i) for i in range(7)] + [
        _p("other", "deylami-2021", 1.0)
    ]
    out = _diversify(ranked, per_doc=2, limit=8)
    from collections import Counter

    counts = Counter(p.doc_id for p in out)
    assert counts["gottman-pdf"] == 2, "one document must not dominate the set"
    assert "deylami-2021" in counts, "a lower-ranked distinct source must get in"


def test_diversify_returns_short_rather_than_padding_with_one_source():
    """The cap is hard.

    Backfilling to fill the budget is what the first version did, and it
    reproduced the bug: a set dominated by one document, arrived at by a
    different route.
    """
    from app.retrieval import _diversify

    ranked = [_p(f"c{i}", "only-paper", 9 - i) for i in range(6)]
    out = _diversify(ranked, per_doc=2, limit=5)
    assert len(out) == 2, "two passages from the only paper, not five copies of it"
    assert [p.chunk_id for p in out] == ["c0", "c1"], "rank order is preserved"


def test_diversify_preserves_rank_order_within_the_cap():
    from app.retrieval import _diversify

    ranked = [_p("a1", "A", 9), _p("b1", "B", 8), _p("a2", "A", 7), _p("c1", "C", 6)]
    out = _diversify(ranked, per_doc=1, limit=3)
    assert [p.chunk_id for p in out] == ["a1", "b1", "c1"]


# ── Time and the gap since last time ───────────────────────────────


def test_gap_descriptions_scale_with_the_actual_gap():
    from datetime import datetime

    from app.session import describe_gap

    now = datetime(2026, 9, 9, 19, 0)
    cases = [
        (datetime(2026, 9, 9, 18, 55), "minutes"),
        (datetime(2026, 9, 9, 18, 10), "sitting"),
        (datetime(2026, 9, 9, 13, 0), "today"),
        (datetime(2026, 9, 8, 20, 0), "yesterday"),
        (datetime(2026, 9, 6, 15, 0), "3 days ago"),
        (datetime(2026, 8, 30, 15, 0), "week"),
        (datetime(2026, 5, 1, 15, 0), "months"),
    ]
    for previous, expected in cases:
        phrase, _ = describe_gap(previous, now)
        assert expected in phrase, f"{previous} -> {phrase!r}, wanted {expected!r}"


def test_gap_is_empty_on_a_first_ever_turn():
    from datetime import datetime

    from app.session import describe_gap

    assert describe_gap(None, datetime(2026, 9, 9)) == ("", 0.0)


def test_gap_does_not_go_negative_on_clock_skew():
    from datetime import datetime

    from app.session import describe_gap

    future = datetime(2026, 9, 10, 12, 0)
    phrase, hours = describe_gap(future, datetime(2026, 9, 9, 12, 0))
    assert (phrase, hours) == ("", 0.0), "a timestamp from the future is not a gap"


# ── Ending a session ───────────────────────────────────────────────


def test_end_request_matches_the_intended_phrases():
    from app.session import is_end_request

    for text in (
        "let's end for now",
        "Let's end for now.",
        "end session",
        "I'm done for today",
        "that's all for tonight",
        "shut it down",
    ):
        assert is_end_request(text), f"should end: {text!r}"


def test_end_request_ignores_the_same_words_inside_a_sentence():
    """This triggers a real shutdown, so the bar is the whole message.

    Cutting the machine off mid-sentence because someone said "I'm done"
    about their relationship would be a memorable way to lose their trust.
    """
    from app.session import is_end_request

    for text in (
        "I'm done for now with trying to explain myself to them",
        "my brother said let's end for now and walked out",
        "I feel like shutting it down whenever anyone raises their voice",
        "we should end the session early next time we try this",
    ):
        assert not is_end_request(text), f"should NOT end: {text!r}"


def test_gap_handles_the_aware_timestamps_postgres_actually_returns():
    """The regression that reached a running stack.

    Every test above passes naive datetimes on both sides, which is not what
    happens in production: `turns.created_at` is TIMESTAMPTZ and comes back
    timezone-aware, while datetime.now() is naive. Subtracting them raises
    "can't subtract offset-naive and offset-aware datetimes", and the whole
    turn 502s.
    """
    from datetime import datetime, timedelta, timezone

    from app.session import describe_gap, now_block

    aware_previous = datetime(2026, 9, 6, 15, 0, tzinfo=timezone.utc)
    naive_now = datetime(2026, 9, 9, 19, 0)

    phrase, hours = describe_gap(aware_previous, naive_now)
    assert phrase, "an aware previous must not blow up against a naive now"
    assert hours > 0

    # Mixed the other way round, and both-aware, must also work.
    assert describe_gap(naive_now - timedelta(days=1), datetime.now().astimezone())[0]
    assert "Now" in now_block(aware_previous)


# ── A fresh install knows nobody ───────────────────────────────────


def test_first_meeting_is_announced_only_when_nothing_is_remembered():
    """A new install has no notes. Without saying so, the model fills the
    gap by acting as if it already knows the person — which, with nothing
    to go on, means inventing them."""
    from app.pipeline import _assemble
    from app.schemas import SafetyVerdict

    history = [ChatMessage(role="user", content="hi")]
    fresh = _assemble(history, [], [], SafetyVerdict(), first_meeting=True)[0]["content"]
    assert "meeting this person for the first time" in fresh
    assert "what they would like to be called" in fresh

    known = _assemble(history, [], [], SafetyVerdict(), profile="Goes by Sam.")[0]["content"]
    assert "meeting this person for the first time" not in known


# ── Session feedback ───────────────────────────────────────────────


def test_feedback_parses_the_ways_people_actually_answer():
    from app.feedback import parse

    assert parse("8 7 9 8") == {"status": "rated", "heard": 8, "focus": 7, "approach": 9,
                                "overall": 8, "comment": ""}
    got = parse("8, 7, 10, 6 — felt a bit rushed at the end")
    assert (got["approach"], got["overall"]) == (10, 6)
    assert got["comment"] == "felt a bit rushed at the end"
    assert parse("7/10 8/10 8/10 9/10")["heard"] == 7
    assert parse("9") == {"status": "rated", "overall": 9, "comment": ""}
    assert parse("6 - good but I wanted more practical stuff")["comment"].startswith("good")
    for skip in ("skip", "Skip.", "no thanks", "not now", "pass"):
        assert parse(skip) == {"status": "skipped"}, skip


def test_feedback_does_not_mistake_conversation_for_a_score():
    """A pending request lapses when they keep talking. Reading "3 things
    happened today" as a rating of 3 would record a score they never gave
    and shut the machine down on them mid-thought."""
    from app.feedback import parse

    for text in (
        "3 things happened today that I want to tell you about",
        "actually wait, one more thing before we stop",
        "10 years ago this would never have bothered me",
        "no, I think it was more that I felt unheard",
    ):
        assert parse(text) is None, text


def test_feedback_note_only_when_something_went_badly():
    from app.feedback import prompt_block

    assert prompt_block(None) == ""
    assert prompt_block({"heard": 9, "focus": 8, "approach": 9, "overall": 9, "comment": ""}) == ""

    note = prompt_block({"heard": 4, "focus": 8, "approach": 9, "overall": 7, "comment": ""})
    assert "heard 4/10" in note and "Slow down" in note
    assert "Ask, once and plainly" not in note, "only the low items get advice"

    note = prompt_block({"heard": 9, "focus": 9, "approach": 9, "overall": 9,
                         "comment": "too much theory"})
    assert "too much theory" in note, "a comment is worth carrying even with high scores"


def test_feedback_ask_lists_all_four_questions():
    from app.feedback import ITEMS, ask

    text = ask({"exchanges": 6})
    assert "6 exchanges saved" in text
    assert all(q in text for _, q in ITEMS)
    assert "skip" in text


def test_last_session_feedback_reaches_the_prompt():
    from app.pipeline import _assemble
    from app.schemas import SafetyVerdict

    history = [ChatMessage(role="user", content="hi")]
    note = "## How the last conversation landed\n\nheard 4/10"
    assert note in _assemble(history, [], [], SafetyVerdict(), feedback_note=note)[0]["content"]


# ── Quality eval rules ─────────────────────────────────────────────


def test_quality_rules_catch_what_the_system_prompt_forbids():
    from app.eval.quality import check_rules

    rules = {"max_questions": 1, "max_words": 350, "allow_list": False}
    good = "That sounds exhausting. What happened right before it started?"
    assert all(v["pass"] for v in check_rules(good, rules).values())

    bad = (
        "## What I notice\n\nI hear that you're feeling frustrated.\n\n---\n\n"
        "- Try this\n- And this\n\nDoes that help? Or would something else?"
    )
    r = check_rules(bad, rules)
    assert not r["no_headers"]["pass"] and not r["no_rules"]["pass"]
    assert not r["no_lists"]["pass"] and not r["no_reflex"]["pass"]
    assert not r["questions"]["pass"]
    assert check_rules(bad, {**rules, "allow_list": True})["no_lists"]["pass"]


def test_quality_scenarios_are_well_formed():
    """Every criterion a scenario asks for must be defined, or the judge is
    asked about something it has no definition for and fails it silently."""
    from pathlib import Path

    import yaml

    spec = yaml.safe_load(Path("/srv/eval/quality.yaml").read_text())
    ids = [s["id"] for s in spec["scenarios"]]
    assert len(ids) == len(set(ids)), "scenario ids must be unique"
    for s in spec["scenarios"]:
        assert s["criteria"], s["id"]
        missing = set(s["criteria"]) - set(spec["criteria"])
        assert not missing, f"{s['id']} uses undefined criteria {missing}"
        if "names_frame" in s["criteria"]:
            assert s.get("frame"), f"{s['id']} judges names_frame without saying which frame"
