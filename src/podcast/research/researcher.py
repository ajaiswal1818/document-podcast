"""Research operations for the agentic podcast engine.

Retrieves scholarly context for a source document: identifies the paper the
document comes from, searches Europe PMC / Crossref, and pulls open-access
full text so the pipeline can ground the episode in the complete paper.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree

EUROPE_PMC_SEARCH = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    "?query={query}&format=json&pageSize=5&resultType=core"
)
EUROPE_PMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
CROSSREF_SEARCH = "https://api.crossref.org/works?query.bibliographic={query}&rows=5"


def _http_get(url: str, timeout: int = 15) -> bytes:
    request = Request(url, headers={"User-Agent": "document-podcast/0.1 (local research module)"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _strip_markup(text: str) -> str:
    """Remove HTML/JATS tags and collapse whitespace."""
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


class ResearchAgent:
    """Scholarly research layer with open-access full-text retrieval."""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm
        self.knowledge_pool: list[dict[str, Any]] = []

    def derive_query(self, document_text: str) -> str:
        """Build the best bibliographic search query for the paper this document comes from."""
        text = (document_text or "").strip()
        if self.llm is not None and getattr(self.llm, "available", False):
            try:
                payload = self.llm.generate_json(
                    "You extract bibliographic search queries from document excerpts.",
                    "Identify the published paper this document excerpt most likely comes from.\n\n"
                    f"EXCERPT:\n{text[:4000]}\n\n"
                    'Return JSON: {"title": "<exact paper title if stated, else best reconstruction>", '
                    '"query": "<search query with the key clinical/scientific terms>"}',
                    max_tokens=2000,
                )
                query = str(payload.get("title") or payload.get("query") or "").strip()
                if query:
                    return query[:250]
            except Exception:
                pass

        for line in text.splitlines():
            if len(line.split()) >= 4:
                return line.strip()[:250]
        return text[:250]

    _STOP_WORDS = {
        "a", "an", "the", "of", "in", "on", "as", "and", "or", "with", "for", "to", "by",
        "case", "report", "study", "presenting", "presentation", "patient", "initial",
    }
    _MINIMUM_VERIFIED_TITLE_OVERLAP = 0.75

    def _europe_pmc_search(self, query: str) -> list[dict[str, Any]]:
        url = EUROPE_PMC_SEARCH.format(query=quote_plus(query))
        payload = json.loads(_http_get(url).decode("utf-8"))
        results = payload.get("resultList", {}).get("result", [])
        return results if isinstance(results, list) else []

    def _significant_terms(self, query: str) -> list[str]:
        words = re.findall(r"[A-Za-z][A-Za-z-]{2,}", query.lower())
        terms = [word for word in words if word not in self._STOP_WORDS]
        # Longest words first as a cheap proxy for distinctiveness.
        return sorted(dict.fromkeys(terms), key=len, reverse=True)

    def _title_overlap(self, query: str, title: str) -> float:
        query_terms = set(self._significant_terms(query))
        title_terms = set(self._significant_terms(title))
        if not query_terms:
            return 0.0
        return len(query_terms & title_terms) / len(query_terms)

    def _search_ranked(self, query: str) -> list[dict[str, Any]]:
        """Search Europe PMC by title terms first, then plain query, ranked by title overlap."""
        results: list[dict[str, Any]] = []
        terms = self._significant_terms(query)[:4]
        if terms:
            title_query = " AND ".join(f'TITLE:"{term}"' for term in terms)
            try:
                results = self._europe_pmc_search(title_query)
            except Exception:
                results = []
        if not results:
            try:
                results = self._europe_pmc_search(query)
            except Exception:
                results = []

        scored = [
            (self._title_overlap(query, str(result.get("title", ""))), result)
            for result in results
            if isinstance(result, dict)
        ]
        scored.sort(key=lambda pair: (pair[0], pair[1].get("isOpenAccess") == "Y"), reverse=True)
        return [result for score, result in scored if score >= 0.3]

    def _europe_pmc_fulltext(self, pmcid: str) -> str:
        """Fetch and flatten the open-access full-text body for a PMC article."""
        raw = _http_get(EUROPE_PMC_FULLTEXT.format(pmcid=pmcid))
        root = ElementTree.fromstring(raw)
        body = root.find(".//body")
        node = body if body is not None else root
        text = " ".join(fragment.strip() for fragment in node.itertext() if fragment.strip())
        return re.sub(r"\s+", " ", text).strip()

    def _crossref_search(self, query: str) -> list[dict[str, Any]]:
        url = CROSSREF_SEARCH.format(query=quote_plus(query))
        payload = json.loads(_http_get(url).decode("utf-8"))
        items = payload.get("message", {}).get("items", [])
        return items if isinstance(items, list) else []

    def research_document(self, document_text: str, *, max_sources: int = 3) -> dict[str, Any]:
        """Retrieve scholarly sources, abstracts, and open-access full text for a document."""
        query = self.derive_query(document_text)
        sources: list[dict[str, Any]] = []
        full_texts: list[dict[str, str]] = []

        results = self._search_ranked(query)

        rejected_source_count = 0
        for result in results:
            if len(sources) >= max_sources:
                break
            if not isinstance(result, dict):
                continue
            title = _strip_markup(str(result.get("title", "")))
            title_overlap = self._title_overlap(query, title)
            # Research adds external claims to the podcast. Do not admit a paper
            # merely because it shares a broad disease name with the source.
            if title_overlap < self._MINIMUM_VERIFIED_TITLE_OVERLAP:
                rejected_source_count += 1
                continue
            entry: dict[str, Any] = {
                "title": title,
                "authors": str(result.get("authorString", "")),
                "journal": str(result.get("journalInfo", {}).get("journal", {}).get("title", "")) if isinstance(result.get("journalInfo"), dict) else "",
                "year": str(result.get("pubYear", "")),
                "doi": str(result.get("doi", "")),
                "pmcid": str(result.get("pmcid", "")),
                "open_access": result.get("isOpenAccess") == "Y",
                "provider": "europe_pmc",
                "title_overlap": round(title_overlap, 3),
                "verified": True,
            }
            abstract = _strip_markup(str(result.get("abstractText", "")))
            if abstract:
                entry["abstract"] = abstract

            if entry["open_access"] and entry["pmcid"]:
                try:
                    body_text = self._europe_pmc_fulltext(entry["pmcid"])
                    word_count = len(body_text.split())
                    # A single article is a few thousand words; enormous bodies are
                    # conference proceedings or supplements matched by accident.
                    if 100 < word_count <= 30000:
                        entry["has_full_text"] = True
                        full_texts.append(
                            {"title": entry["title"], "pmcid": entry["pmcid"], "text": " ".join(body_text.split()[:15000])}
                        )
                except Exception:
                    entry["has_full_text"] = False
            sources.append(entry)

        if not sources:
            try:
                for item in self._crossref_search(query):
                    if len(sources) >= max_sources:
                        break
                    if not isinstance(item, dict):
                        continue
                    titles = item.get("title") or []
                    title = _strip_markup(str(titles[0])) if titles else ""
                    title_overlap = self._title_overlap(query, title)
                    if title_overlap < self._MINIMUM_VERIFIED_TITLE_OVERLAP:
                        rejected_source_count += 1
                        continue
                    sources.append(
                        {
                            "title": title,
                            "authors": ", ".join(
                                str(author.get("family", "")) for author in item.get("author", [])[:8]
                            ),
                            "year": "",
                            "doi": str(item.get("DOI", "")),
                            "abstract": _strip_markup(str(item.get("abstract", ""))),
                            "open_access": False,
                            "provider": "crossref",
                            "title_overlap": round(title_overlap, 3),
                            "verified": True,
                        }
                    )
            except Exception:
                pass

        report = {
            "query": query,
            "sources": sources,
            "full_texts": full_texts,
            "abstracts": [source["abstract"] for source in sources if source.get("abstract")],
            "verification": {
                "minimum_title_overlap": self._MINIMUM_VERIFIED_TITLE_OVERLAP,
                "accepted_sources": len(sources),
                "rejected_source_count": rejected_source_count,
                "all_sources_verified": all(source.get("verified") for source in sources),
            },
        }
        self.knowledge_pool.append(report)
        return report

    def lookup(self, topic: str, material: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return source-backed evidence for a topic, falling back to local material."""
        material = material or {}
        facts = material.get("main_ideas", []) or material.get("facts", []) or []

        findings: list[dict[str, Any]] = []
        try:
            for result in self._europe_pmc_search(topic)[:5]:
                if not isinstance(result, dict):
                    continue
                claim = _strip_markup(str(result.get("abstractText", ""))) or _strip_markup(str(result.get("title", "")))
                if not claim:
                    continue
                doi = str(result.get("doi", ""))
                findings.append(
                    {
                        "claim": claim[:600],
                        "source_title": _strip_markup(str(result.get("title", ""))) or "Europe PMC",
                        "url": f"https://doi.org/{doi}" if doi else "https://europepmc.org/",
                        "source_type": "scholarly_abstract",
                        "relevance": 0.9,
                    }
                )
        except Exception:
            findings = []

        if not findings:
            findings = [
                {
                    "claim": str(facts[0]) if facts else f"The source material suggests the topic is important for {topic}.",
                    "source_title": "Source document",
                    "url": "local://document",
                    "source_type": "internal_document",
                    "relevance": 0.95,
                }
            ]

        sources = [
            {
                "title": finding.get("source_title") or "Source",
                "url": finding.get("url") or "local://document",
                "summary": finding.get("claim") or "No summary provided.",
                "source": finding.get("source_type") or "internal_document",
                "relevance": finding.get("relevance", 0.8),
            }
            for finding in findings
        ]

        entry = {
            "topic": topic,
            "findings": findings,
            "sources": sources,
            "summary": f"Research for {topic} has been retrieved and structured as evidence for the conversation.",
        }
        self.knowledge_pool.append(entry)
        return entry
