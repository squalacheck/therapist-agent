"""Retrieval eval: recall@k, with and without the reranker.

    python -m app.eval.retrieval /srv/eval/retrieval.yaml

The interesting output is the gap between the two columns. If reranking is
not moving recall, either the corpus is too thin for the questions or the
reranker is not loading, and either way it should not be costing you 1.2GB.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import yaml

from ..config import get_settings
from ..embedding import embed_query, rerank, sparse_query
from ..retrieval import get_qdrant


async def _candidates(question: str, limit: int) -> list[str]:
    from qdrant_client import models

    s = get_settings()
    vector = await embed_query(question)
    result = await get_qdrant().query_points(
        collection_name=s.qdrant_collection,
        prefetch=[
            models.Prefetch(query=vector, using="dense", limit=limit),
            models.Prefetch(
                query=sparse_query(question),
                using="sparse",
                limit=limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return [p.payload.get("text", "") for p in result.points if p.payload]


def _hit(passages: list[str], terms: list[str]) -> bool:
    blob = " ".join(passages).lower()
    return any(term.lower() in blob for term in terms)


async def main(path: str) -> int:
    s = get_settings()
    cases = yaml.safe_load(Path(path).read_text())["cases"]

    print(f"\n  {len(cases)} cases · retrieve top {s.retrieve_top_k} · keep top {s.rerank_top_k}\n")
    print(f"  {'case':<24} {'hybrid':>8} {'reranked':>10}")
    print(f"  {'-' * 24} {'-' * 8:>8} {'-' * 10:>10}")

    hybrid_hits = reranked_hits = 0
    failures: list[str] = []

    for case in cases:
        question, terms = case["question"], case["must_match_any"]
        candidates = await _candidates(question, s.retrieve_top_k)

        baseline = _hit(candidates[: s.rerank_top_k], terms)

        try:
            scores = await rerank(question, candidates)
            ordered = [
                text
                for text, _ in sorted(
                    zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True
                )
            ]
            improved = _hit(ordered[: s.rerank_top_k], terms)
        except Exception as exc:  # noqa: BLE001
            print(f"  reranker failed: {exc}")
            improved = baseline

        hybrid_hits += baseline
        reranked_hits += improved
        if not improved:
            failures.append(case["id"])

        mark = lambda ok: "  hit" if ok else " miss"  # noqa: E731
        print(f"  {case['id']:<24} {mark(baseline):>8} {mark(improved):>10}")

    n = len(cases)
    print(f"\n  recall@{s.rerank_top_k}   hybrid {hybrid_hits}/{n} ({hybrid_hits / n:.0%})"
          f"   reranked {reranked_hits}/{n} ({reranked_hits / n:.0%})")

    if failures:
        print(f"\n  misses: {', '.join(failures)}")
        print("  A miss usually means the corpus lacks coverage, not that retrieval is")
        print("  broken. Check `make corpus-stats`, then widen the manifest queries.\n")
    else:
        print()

    return 0 if reranked_hits >= 0.7 * n else 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/srv/eval/retrieval.yaml"
    sys.exit(asyncio.run(main(path)))
