"""Corpus ingestion CLI.

    python -m ingest.cli run [--dry-run] [--source pmc]
    python -m ingest.cli reindex
    python -m ingest.cli stats

Harvesting is idempotent: documents already in the index are skipped, so
re-running picks up only what is new. Raw harvested metadata is written to
data/processed/documents.jsonl so `reindex` can rebuild the vector store
without going back out to the network.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import structlog
import yaml

from . import index
from .harvesters.pmc import PMCHarvester
from .harvesters.simple import (
    DOAJHarvester,
    DirectHarvester,
    DropFolderHarvester,
    EuropePMCHarvester,
    OSFHarvester,
)
from .models import Document

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    ),
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(),
    ],
)
log = structlog.get_logger("ingest")

DATA = Path("/data")
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
DOCUMENTS_FILE = PROCESSED / "documents.jsonl"
# Append-only record of documents whose chunks are ALL in the index.
# Qdrant cannot answer this: a doc_id appears there as soon as the
# first chunk lands, so an interrupted run leaves documents that look
# indexed and are not. This file is the only thing that knows.
INDEXED_FILE = PROCESSED / "indexed.jsonl"
MANIFEST = Path(__file__).parent / "manifest.yaml"


async def harvest(manifest: dict, only_source: str | None) -> list[Document]:
    defaults = manifest.get("defaults", {})
    max_results = defaults.get("max_results", 60)
    min_year = defaults.get("min_year", 1990)

    docs: list[Document] = []

    for source in manifest.get("sources", []):
        if not source.get("enabled", True):
            continue
        if only_source and source["name"] != only_source:
            continue

        kind = source["type"]
        log.info("harvest.source", name=source["name"], type=kind)

        try:
            if kind == "pmc":
                harvester = PMCHarvester()
                try:
                    allowed = set(source.get("license_filter") or [])
                    for query in source["queries"]:
                        ids = await harvester.search(query["terms"], max_results, min_year)
                        found = await harvester.fetch(ids, query.get("tags", []))
                        if allowed:
                            found = [
                                d for d in found if not d.license or d.license in allowed
                                or d.license == "open-access"
                            ]
                        docs.extend(found)
                finally:
                    await harvester.aclose()

            elif kind == "europepmc":
                harvester = EuropePMCHarvester()
                try:
                    for query in source["queries"]:
                        docs.extend(
                            await harvester.harvest(
                                query["terms"], max_results, query.get("tags", [])
                            )
                        )
                finally:
                    await harvester.aclose()

            elif kind == "osf":
                harvester = OSFHarvester(source.get("provider", "psyarxiv"))
                try:
                    for query in source["queries"]:
                        docs.extend(
                            await harvester.harvest(
                                query["terms"], max_results, query.get("tags", [])
                            )
                        )
                finally:
                    await harvester.aclose()

            elif kind == "doaj":
                harvester = DOAJHarvester()
                try:
                    for query in source["queries"]:
                        docs.extend(
                            await harvester.harvest(
                                query["terms"], max_results, query.get("tags", [])
                            )
                        )
                finally:
                    await harvester.aclose()

            elif kind == "direct":
                harvester = DirectHarvester(RAW)
                try:
                    for spec in source.get("documents", []):
                        if doc := await harvester.harvest(spec, source.get("license", "unknown")):
                            docs.append(doc)
                finally:
                    await harvester.aclose()

            elif kind == "dropfolder":
                docs.extend(
                    await DropFolderHarvester().harvest(
                        Path(source.get("path", RAW / "dropfolder")),
                        source.get("license", "local"),
                        source.get("tags", []),
                    )
                )

            else:
                log.warning("harvest.unknown_type", type=kind)

        except Exception as exc:  # noqa: BLE001
            # One bad source must not abort the whole harvest.
            log.error("harvest.source_failed", name=source["name"], error=str(exc))

    # Dedup across sources. PMC and Europe PMC overlap heavily by design.
    unique: dict[str, Document] = {}
    for doc in docs:
        key = doc.doc_id()
        # Prefer whichever copy actually has full text.
        if key not in unique or (doc.full_text and not unique[key].full_text):
            unique[key] = doc

    log.info("harvest.complete", harvested=len(docs), unique=len(unique))
    return list(unique.values())


def write_documents(docs: list[Document]) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with DOCUMENTS_FILE.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    log.info("documents.written", path=str(DOCUMENTS_FILE), count=len(docs))


def read_documents() -> list[Document]:
    if not DOCUMENTS_FILE.exists():
        return []
    with DOCUMENTS_FILE.open(encoding="utf-8") as fh:
        return [Document(**json.loads(line)) for line in fh if line.strip()]


def read_ledger() -> set[str]:
    """doc_ids known to be completely indexed."""
    if not INDEXED_FILE.exists():
        return set()
    done: set[str] = set()
    with INDEXED_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["doc_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _ledger_writer():
    """Append to the ledger and flush per document.

    Flushed and fsynced every time, not buffered. The whole point of the
    ledger is to survive the run being killed, and a write sitting in a
    userspace buffer does not.
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)
    fh = INDEXED_FILE.open("a", encoding="utf-8")

    def record(doc: Document, chunks: int) -> None:
        fh.write(json.dumps({"doc_id": doc.doc_id(), "chunks": chunks}) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    return fh, record


def build_and_index(docs: list[Document], manifest: dict, *, resume: bool) -> None:
    collection = os.getenv("QDRANT_COLLECTION", "corpus")
    defaults = manifest.get("defaults", {})

    already = read_ledger() if resume else set()
    pending = [d for d in docs if d.doc_id() not in already]
    if already:
        log.info("index.resuming", already_complete=len(docs) - len(pending))

    full_text = sum(1 for d in pending if d.full_text)
    log.info(
        "index.prepared",
        documents=len(pending),
        with_full_text=full_text,
        abstract_only=len(pending) - full_text,
    )
    if not pending:
        log.info("index.nothing_to_do")
        return

    fh, record = _ledger_writer()
    try:
        index.index_documents(
            pending,
            collection,
            batch_size=int(os.getenv("EMBED_BATCH_SIZE", "32")),
            max_tokens=defaults.get("chunk_tokens", 800),
            overlap=defaults.get("chunk_overlap", 0.15),
            on_complete=record,
        )
    finally:
        fh.close()


async def cmd_run(args: argparse.Namespace) -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())
    docs = await harvest(manifest, args.source)
    if not docs:
        log.warning("run.no_documents")
        return
    write_documents(docs)

    if args.dry_run:
        by_source: dict[str, int] = {}
        for doc in docs:
            by_source[doc.source] = by_source.get(doc.source, 0) + 1
        log.info("run.dry_run_complete", by_source=by_source)
        print(f"\n  {len(docs)} documents harvested (metadata written, nothing indexed)")
        for source, count in sorted(by_source.items(), key=lambda kv: -kv[1]):
            print(f"    {source:<14} {count}")
        print(f"\n  Review {DOCUMENTS_FILE}, then run `make ingest` to index.")
        return

    build_and_index(docs, manifest, resume=True)


def cmd_reindex(args: argparse.Namespace) -> None:
    docs = read_documents()
    if not docs:
        log.error("reindex.no_documents", hint="run `make ingest` first")
        return
    manifest = yaml.safe_load(MANIFEST.read_text())
    build_and_index(docs, manifest, resume=args.resume)


def cmd_stats(_: argparse.Namespace) -> None:
    collection = os.getenv("QDRANT_COLLECTION", "corpus")
    info = index.stats(collection)
    if not info["exists"]:
        print(f"  Collection '{collection}' does not exist yet. Run `make ingest`.")
        return
    print(f"\n  Collection  {collection}")
    print(f"  Documents   {info['documents']}")
    print(f"  Chunks      {info['chunks']}")

    docs = read_documents()
    if docs:
        by_source: dict[str, int] = {}
        by_tag: dict[str, int] = {}
        for doc in docs:
            by_source[doc.source] = by_source.get(doc.source, 0) + 1
            for tag in doc.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1
        print("\n  By source")
        for source, count in sorted(by_source.items(), key=lambda kv: -kv[1]):
            print(f"    {source:<14} {count}")
        print("\n  By topic")
        for tag, count in sorted(by_tag.items(), key=lambda kv: -kv[1]):
            print(f"    {tag:<14} {count}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="ingest")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="harvest and index")
    run.add_argument("--dry-run", action="store_true", help="harvest metadata only")
    run.add_argument("--source", help="limit to one manifest source")
    run.set_defaults(func=lambda a: asyncio.run(cmd_run(a)))

    reindex = sub.add_parser("reindex", help="re-embed documents already on disk")
    reindex.add_argument(
        "--resume",
        action="store_true",
        help="skip documents the ledger records as fully indexed",
    )
    reindex.set_defaults(func=cmd_reindex)
    sub.add_parser("stats", help="show corpus contents").set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
