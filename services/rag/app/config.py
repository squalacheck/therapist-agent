"""Runtime configuration, read from the environment."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    # ── Inference ──────────────────────────────────────────────────
    vllm_base_url: str = "http://vllm:8000/v1"
    vllm_model: str = "therapist-base"
    vllm_timeout_s: float = 600.0

    # The name Open WebUI shows in its model picker.
    public_model_name: str = "therapist"

    # Thinking off for the user-facing turn, and this is a latency decision,
    # not a quality one. With --reasoning-parser qwen3, reasoning goes to
    # `reasoning_content`; llm.stream() only forwards `content`, so every
    # reasoning token is generated, paid for, and thrown away. At the ~4 tok/s
    # this 27B does in eager mode that is minutes of dead air before a word
    # appears. Measured: "reply with exactly: ok" took 61s thinking, 0.55s not.
    # Turn it on only if you also surface reasoning_content in the UI.
    enable_thinking: bool = False

    # ── Retrieval ──────────────────────────────────────────────────
    retrieval_enabled: bool = False
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "corpus"
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    rerank_model: str = "Qwen/Qwen3-Reranker-0.6B"
    embed_device: str = "cuda"

    # Cast wide, then let the reranker cut. The gap between these two
    # numbers is where retrieval quality actually comes from.
    retrieve_top_k: int = 40
    rerank_top_k: int = 8
    # Below this the passage is noise and hurts more than it helps.
    # NOT calibrated yet — 0.15 assumes a 0..1 score, and Qwen3-Reranker's
    # LogitScore head with an Identity activation does not necessarily give
    # one. Run a query with retrieval on and read the retrieval.rerank_scores
    # line in `make logs-rag` before trusting this number.
    min_rerank_score: float = 0.15

    # At most this many chunks from any one document in the final set.
    # Without a cap the reranker is free to fill all eight slots from the
    # single most on-topic paper — which is exactly what it did on the
    # Four Horsemen question: seven of eight passages were two near-identical
    # copies of the same Gottman Institute PDF. Technically the best eight
    # passages; in practice one source wearing eight hats, and it makes
    # "follow the citations" — the check that catches invented references —
    # impossible to actually perform.
    max_passages_per_doc: int = 2

    # ── Memory ─────────────────────────────────────────────────────
    memory_enabled: bool = False
    database_url: str = ""
    memory_recall_k: int = 8

    # The running profile — the "session notes" layer. Injected on every
    # turn, unconditionally, so a new chat that opens with "hey" still
    # knows who it is talking to. Rewritten every N exchanges rather than
    # every turn, which would double the model calls per exchange to
    # maintain a summary that does not change that fast.
    profile_enabled: bool = True
    profile_update_every: int = 3
    profile_max_tokens: int = 420

    # Name for a second entry in the model picker that reads and writes no
    # memory at all. For thinking about someone else's situation, or a
    # research question that should not become part of your own history.
    fresh_model_name: str = "therapist-fresh"
    # Extraction runs after the response streams, so it never adds latency.
    memory_extract_async: bool = True

    # ── Daily practice ─────────────────────────────────────────────
    # The between-session layer: what was agreed to, whether it happened,
    # and what that means for tomorrow. See daily.py.
    daily_enabled: bool = True
    daily_model_name: str = "therapist-daily"
    # Three or four. A list that reads as a programme gets abandoned, and an
    # abandoned list is worse than none — it teaches that this tool asks for
    # things you do not do.
    daily_max_items: int = 4
    daily_max_tokens: int = 900
    daily_passages: int = 4
    # Named explicitly rather than left to be inferred from the notes, so
    # the relationship and communication items are reliably about the right
    # person even on a day the profile happens not to mention them.
    partner_name: str = ""

    # ── Time and sessions ──────────────────────────────────────────
    # Log every turn with a timestamp, and tell the model what time it is
    # and how long since the last exchange. Without this it cannot tell
    # coming back after lunch from coming back after a fortnight.
    time_awareness: bool = True
    # "let's end for now" closes the session out and asks the host to stop
    # the stack. The container writes a file; a watcher on the host acts on
    # it. Deliberately NOT the Docker socket — see session._request_shutdown.
    shutdown_on_end: bool = True
    shutdown_flag_path: str = "/runtime/shutdown-requested"

    # ── Safety ─────────────────────────────────────────────────────
    safety_enabled: bool = True
    # Screen the agent's own output, not just the user's input. The input
    # classifier cannot catch a reply that drifts somewhere it should not
    # go, and that is the direction with the worse failure mode.
    safety_output_check: bool = True

    # ── Session pacing ─────────────────────────────────────────────
    # A raw model reaches for the fix. Real sessions open by understanding
    # and only then move to what to do about it. This nudges the stance by
    # where the conversation is, rather than leaving every turn identical.
    pacing_enabled: bool = True
    # Exchanges spent understanding before action-planning is encouraged.
    pacing_opening_turns: int = 3

    # ── Prompt budget ──────────────────────────────────────────────
    # Must track MAX_MODEL_LEN in .env — compose passes it in as
    # MAX_CONTEXT_TOKENS. 32768 is what the card actually affords; see
    # docs/gpu-notes.md. The three sub-budgets below sum to exactly this.
    max_context_tokens: int = 32768
    reserve_output_tokens: int = 4096
    max_history_tokens: int = 12288
    max_passage_tokens: int = 16384

    # ── Misc ───────────────────────────────────────────────────────
    log_level: str = "INFO"
    prompts_dir: Path = Path(__file__).parent / "prompts"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def load_prompt(name: str) -> str:
    """Read a versioned prompt file.

    Prompts live on disk rather than inline in the source because the
    therapeutic stance gets iterated on constantly, and being able to
    diff it is how you tell whether a change actually helped.
    """
    path = get_settings().prompts_dir / f"{name}.md"
    return path.read_text(encoding="utf-8").strip()
