"""PubMed Central open-access subset, via NCBI E-utilities.

The two-step shape matters: esearch against the `pmc` database with an
`open access[filter]` restriction gives PMCIDs, then efetch returns full
JATS XML. Parsing JATS structurally is why this source produces much cleaner
chunks than anything that goes through a PDF.
"""

from __future__ import annotations

import os
import re
from xml.etree import ElementTree as ET

import structlog

from ..models import Document
from .base import HttpClient

log = structlog.get_logger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Boilerplate that adds nothing and dilutes retrieval.
SKIP_SECTIONS = re.compile(
    r"^(acknowledg|funding|conflict|competing interest|author contribution|"
    r"supplementary|abbreviation|data availability|ethics|references)",
    re.IGNORECASE,
)

LICENSE_HINTS = {
    "cc-by": ["creativecommons.org/licenses/by/", "cc by"],
    "cc-by-sa": ["licenses/by-sa"],
    "cc-by-nc": ["licenses/by-nc"],
    "cc0": ["publicdomain/zero", "cc0"],
    "public-domain": ["public domain", "u.s. government work"],
}


class PMCHarvester:
    name = "pmc"

    def __init__(self) -> None:
        api_key = os.getenv("NCBI_API_KEY", "").strip()
        # NCBI allows 3 req/s without a key, 10 with one.
        self._http = HttpClient(per_second=9.0 if api_key else 2.5)
        self._api_key = api_key

    async def aclose(self) -> None:
        await self._http.aclose()

    def _params(self, extra: dict) -> dict:
        params = {"tool": "therapist-agent", "email": os.getenv("CONTACT_EMAIL", ""), **extra}
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    async def search(self, terms: str, max_results: int, min_year: int) -> list[str]:
        query = f'({terms}) AND "open access"[filter] AND {min_year}:3000[dp]'
        response = await self._http.get(
            f"{EUTILS}/esearch.fcgi",
            params=self._params(
                {"db": "pmc", "term": query, "retmax": max_results, "retmode": "json"}
            ),
        )
        if response is None:
            return []
        try:
            ids = response.json()["esearchresult"]["idlist"]
        except (KeyError, ValueError):
            log.warning("pmc.search_parse_failed", terms=terms[:80])
            return []
        log.info("pmc.search", terms=terms[:60], hits=len(ids))
        return ids

    async def fetch(self, pmcids: list[str], tags: list[str]) -> list[Document]:
        docs: list[Document] = []
        # efetch handles batches; 20 keeps responses a sane size.
        for i in range(0, len(pmcids), 20):
            batch = pmcids[i : i + 20]
            response = await self._http.get(
                f"{EUTILS}/efetch.fcgi",
                params=self._params({"db": "pmc", "id": ",".join(batch), "retmode": "xml"}),
            )
            if response is None:
                continue
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as exc:
                log.warning("pmc.xml_parse_failed", error=str(exc))
                continue
            for article in root.findall(".//article"):
                if doc := self._parse_article(article, tags):
                    docs.append(doc)
        return docs

    def _parse_article(self, article: ET.Element, tags: list[str]) -> Document | None:
        def ids(id_type: str) -> str | None:
            node = article.find(f'.//article-id[@pub-id-type="{id_type}"]')
            return node.text.strip() if node is not None and node.text else None

        title_node = article.find(".//article-title")
        title = "".join(title_node.itertext()).strip() if title_node is not None else ""
        if not title:
            return None

        authors: list[str] = []
        for contrib in article.findall('.//contrib[@contrib-type="author"]'):
            surname = contrib.findtext(".//surname", "").strip()
            given = contrib.findtext(".//given-names", "").strip()
            if surname:
                authors.append(f"{surname}, {given}".strip(", "))

        year = None
        for node in article.findall(".//pub-date/year"):
            if node.text and node.text.strip().isdigit():
                year = int(node.text.strip())
                break

        journal = article.findtext(".//journal-title", "").strip() or None

        abstract_node = article.find(".//abstract")
        abstract = (
            " ".join("".join(abstract_node.itertext()).split()) if abstract_node is not None else ""
        )

        sections: dict[str, str] = {}
        body = article.find(".//body")
        if body is not None:
            for sec in body.findall(".//sec"):
                heading = (sec.findtext("title") or "Body").strip()
                if SKIP_SECTIONS.match(heading):
                    continue
                paragraphs = [
                    " ".join("".join(p.itertext()).split()) for p in sec.findall("./p")
                ]
                text = "\n\n".join(t for t in paragraphs if len(t) > 60)
                if len(text) > 200:
                    # Repeated headings ("Results" in a multi-study paper) merge.
                    sections[heading] = (sections.get(heading, "") + "\n\n" + text).strip()

        license_text = ""
        if (permissions := article.find(".//permissions")) is not None:
            license_text = " ".join("".join(permissions.itertext()).split()).lower()
            for node in permissions.iter():
                for value in node.attrib.values():
                    license_text += " " + str(value).lower()

        license_id = next(
            (name for name, hints in LICENSE_HINTS.items() if any(h in license_text for h in hints)),
            "open-access",
        )

        pmcid = ids("pmcid") or ids("pmc")
        return Document(
            doi=ids("doi"),
            pmcid=f"PMC{pmcid}" if pmcid and not pmcid.startswith("PMC") else pmcid,
            pmid=ids("pmid"),
            url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/" if pmcid else None,
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            abstract=abstract,
            sections=sections,
            license=license_id,
            source="pmc",
            tags=tags,
            full_text=bool(sections),
        )
