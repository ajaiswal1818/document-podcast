"""Research operations for the agentic podcast engine."""

from __future__ import annotations

import json
from urllib.parse import quote_plus
from urllib.request import urlopen
from typing import Any


class ResearchAgent:
    """Research layer that retrieves public evidence when possible and falls back to local material otherwise."""

    def __init__(self) -> None:
        self.knowledge_pool: list[dict[str, Any]] = []

    def _fetch_web_evidence(self, topic: str) -> list[dict[str, Any]]:
        """Query a public DuckDuckGo instant answer endpoint for concrete evidence."""
        query = quote_plus(topic)
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
        try:
            with urlopen(url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []

        findings: list[dict[str, Any]] = []
        abstract = payload.get("AbstractText")
        if abstract:
            findings.append(
                {
                    "claim": abstract,
                    "source_title": payload.get("AbstractSource") or "DuckDuckGo",
                    "url": payload.get("AbstractURL") or "https://duckduckgo.com/",
                    "source_type": "web_summary",
                    "relevance": 0.92,
                }
            )

        for item in payload.get("RelatedTopics", []) or []:
            if isinstance(item, dict):
                text = item.get("Text")
                source_url = item.get("FirstURL")
                if text:
                    findings.append(
                        {
                            "claim": text,
                            "source_title": item.get("Name") or "Related result",
                            "url": source_url or "https://duckduckgo.com/",
                            "source_type": "related_topic",
                            "relevance": 0.76,
                        }
                    )

        return findings[:5]

    def lookup(self, topic: str, material: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return source-backed evidence and local fallback material for a topic."""
        material = material or {}
        facts = material.get("main_ideas", []) or material.get("facts", []) or []
        findings = self._fetch_web_evidence(topic)

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
