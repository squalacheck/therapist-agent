"""Europe PMC, OSF/PsyArXiv, DOAJ, direct URLs, and the drop folder.

These are grouped because they are all thin: one REST call, or one download.
PMC is the one that earns its own module.
"""

from __future__ import annotations

import re
from pathlib import Path

import structlog

from ..models import Document
from .base import HttpClient

log = structlog.get_logger(__name__)


def _clean(text: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", text or "").split())


class EuropePMCHarvester:
    """Europe PMC REST. Overlaps PMC; dedup by DOI makes that harmless."""

    name = "europepmc"
    BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self) -> None:
        self._http = HttpClient(per_second=4.0)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def harvest(self, terms: str, max_results: int, tags: list[str]) -> list[Document]:
        response = await self._http.get(
            self.BASE,
            params={
                "query": terms,
                "format": "json",
                "pageSize": min(max_results, 100),
                "resultType": "core",
            },
        )
        if response is None:
            return []
        try:
            results = response.json()["resultList"]["result"]
        except (KeyError, ValueError):
            return []

        docs = []
        for r in results:
            abstract = _clean(r.get("abstractText", ""))
            if not abstract:
                continue
            authors = [
                a.get("fullName", "")
                for a in (r.get("authorList") or {}).get("author", [])
                if a.get("fullName")
            ]
            year = r.get("pubYear")
            docs.append(
                Document(
                    doi=r.get("doi"),
                    pmcid=r.get("pmcid"),
                    pmid=r.get("pmid"),
                    url=f"https://europepmc.org/article/{r.get('source', 'MED')}/{r.get('id')}",
                    title=_clean(r.get("title", "")),
                    authors=authors,
                    year=int(year) if year and str(year).isdigit() else None,
                    journal=(r.get("journalInfo") or {}).get("journal", {}).get("title"),
                    abstract=abstract,
                    license=r.get("license") or "open-access",
                    source="europepmc",
                    tags=tags,
                    # Abstract only. Europe PMC full text needs a second call
                    # per article; PMC already covers most of the same papers
                    # with structure, so it is not worth the request budget.
                    full_text=False,
                )
            )
        log.info("europepmc.harvest", terms=terms[:60], docs=len(docs))
        return docs


class OSFHarvester:
    """OSF Preprints, which is where PsyArXiv lives."""

    name = "osf"
    BASE = "https://api.osf.io/v2/preprints/"

    def __init__(self, provider: str = "psyarxiv") -> None:
        self._http = HttpClient(per_second=2.0)
        self._provider = provider

    async def aclose(self) -> None:
        await self._http.aclose()

    async def harvest(self, terms: str, max_results: int, tags: list[str]) -> list[Document]:
        # filter[q] is NOT a field on this endpoint — OSF answers 400 to every
        # request using it, so this whole source silently contributed zero
        # documents to the first harvest. The working form is a per-field
        # `contains` filter, run over title and abstract and merged, because
        # neither alone finds much: "DARVO" appears in five PsyArXiv abstracts
        # and no titles, "coercive" in both.
        entries: list[dict] = []
        seen_ids: set[str] = set()
        for field in ("title", "description"):
            response = await self._http.get(
                self.BASE,
                params={
                    "filter[provider]": self._provider,
                    f"filter[{field}][contains]": terms,
                    "page[size]": min(max_results, 100),
                },
            )
            if response is None:
                continue
            try:
                found = response.json().get("data", [])
            except ValueError:
                continue
            for entry in found:
                if (eid := entry.get("id")) and eid not in seen_ids:
                    seen_ids.add(eid)
                    entries.append(entry)

        if not entries:
            log.info("osf.no_results", terms=terms[:60], provider=self._provider)
            return []
        log.info("osf.search", terms=terms[:60], hits=len(entries))

        docs = []
        for entry in entries:
            attrs = entry.get("attributes", {})
            abstract = _clean(attrs.get("description", ""))
            if len(abstract) < 200:
                continue
            published = attrs.get("date_published") or ""
            docs.append(
                Document(
                    doi=attrs.get("doi"),
                    url=f"https://osf.io/preprints/{self._provider}/{entry.get('id')}",
                    title=_clean(attrs.get("title", "")),
                    year=int(published[:4]) if published[:4].isdigit() else None,
                    journal=f"{self._provider} (preprint)",
                    abstract=abstract,
                    license="preprint",
                    source=self._provider,
                    tags=tags,
                    full_text=False,
                )
            )
        log.info("osf.harvest", provider=self._provider, terms=terms[:60], docs=len(docs))
        return docs


class DOAJHarvester:
    name = "doaj"
    BASE = "https://doaj.org/api/search/articles"

    def __init__(self) -> None:
        self._http = HttpClient(per_second=1.5)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def harvest(self, terms: str, max_results: int, tags: list[str]) -> list[Document]:
        response = await self._http.get(
            f"{self.BASE}/{terms}", params={"pageSize": min(max_results, 100)}
        )
        if response is None:
            return []
        try:
            results = response.json().get("results", [])
        except ValueError:
            return []

        docs = []
        for r in results:
            bib = r.get("bibjson", {})
            abstract = _clean(bib.get("abstract", ""))
            if len(abstract) < 200:
                continue
            doi = next(
                (i.get("id") for i in bib.get("identifier", []) if i.get("type") == "doi"), None
            )
            url = next((link.get("url") for link in bib.get("link", []) if link.get("url")), None)
            year = bib.get("year")
            docs.append(
                Document(
                    doi=doi,
                    url=url,
                    title=_clean(bib.get("title", "")),
                    authors=[a.get("name", "") for a in bib.get("author", []) if a.get("name")],
                    year=int(year) if year and str(year).isdigit() else None,
                    journal=(bib.get("journal") or {}).get("title"),
                    abstract=abstract,
                    license="open-access",
                    source="doaj",
                    tags=tags,
                    full_text=False,
                )
            )
        log.info("doaj.harvest", terms=terms[:60], docs=len(docs))
        return docs


class DirectHarvester:
    """Named documents at known URLs — the public-domain agency material
    and the openly posted primary sources."""

    name = "direct"

    def __init__(self, raw_dir: Path) -> None:
        self._http = HttpClient(per_second=1.0, timeout=180.0)
        self._raw = raw_dir / "direct"
        self._raw.mkdir(parents=True, exist_ok=True)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def harvest(self, spec: dict, license_id: str) -> Document | None:
        url = spec["url"]
        filename = re.sub(r"[^A-Za-z0-9._-]", "_", url.split("/")[-1]) or "document.pdf"
        path = self._raw / filename

        if not path.exists():
            response = await self._http.get(url)
            if response is None:
                log.warning("direct.download_failed", url=url)
                return None

            # Verify it is actually a PDF before writing it. Several .gov
            # hosts answer a dead PDF path with 200 and an HTML page, and
            # PyMuPDF will happily parse that HTML into "sections". The first
            # harvest ingested the VA site's navigation chrome as a clinical
            # document, with sections titled "Cemetery Locations" and
            # "Summer Sports Clinic". Nothing downstream would ever catch it:
            # it embeds cleanly and retrieves like any other passage.
            body = response.content
            content_type = response.headers.get("content-type", "").lower()
            if not body.startswith(b"%PDF") or "html" in content_type:
                log.warning(
                    "direct.not_a_pdf",
                    url=url,
                    content_type=content_type or "unknown",
                    bytes=len(body),
                    hint="the path is probably dead, or the host is blocking "
                    "the harvester's User-Agent — fetch it by hand into "
                    "data/raw/dropfolder/ instead",
                )
                return None

            path.write_bytes(body)
            log.info("direct.downloaded", url=url, bytes=len(body))

        from ..parse import parse_pdf

        sections = parse_pdf(path)
        if not sections:
            log.warning("direct.parse_empty", path=str(path))
            return None

        return Document(
            url=url,
            title=spec.get("title", filename),
            authors=[spec["publisher"]] if spec.get("publisher") else [],
            year=spec.get("year"),
            journal=spec.get("publisher"),
            sections=sections,
            license=license_id,
            source="direct",
            tags=spec.get("tags", []),
            full_text=True,
        )


class DropFolderHarvester:
    """Whatever you put in data/raw/dropfolder/. Empty by default."""

    name = "dropfolder"

    async def harvest(self, path: Path, license_id: str, tags: list[str]) -> list[Document]:
        if not path.exists():
            return []

        from ..parse import parse_pdf, parse_text

        docs = []
        for file in sorted(path.rglob("*")):
            if not file.is_file():
                continue
            suffix = file.suffix.lower()
            if suffix == ".pdf":
                sections = parse_pdf(file)
            elif suffix in (".txt", ".md"):
                sections = parse_text(file)
            else:
                continue
            if not sections:
                continue
            docs.append(
                Document(
                    url=f"file://{file}",
                    title=file.stem.replace("_", " ").replace("-", " "),
                    sections=sections,
                    license=license_id,
                    source="dropfolder",
                    tags=tags,
                    full_text=True,
                )
            )
        if docs:
            log.info("dropfolder.harvest", docs=len(docs))
        return docs
