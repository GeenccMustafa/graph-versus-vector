"""LLM-based entity & relationship extraction -- the heart of GraphRAG.

For every chunk we ask the model to pull out *entities* (people, places, orgs,
works, ...) and the *relationships* between them. The resulting triples are
what give GraphRAG its ability to connect evidence across passages.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    name: str
    type: str = "ENTITY"
    description: str = ""

    @property
    def normalized(self) -> str:
        return normalize_name(self.name)


@dataclass
class Relationship:
    source: str
    target: str
    description: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (normalize_name(self.source), normalize_name(self.target))


@dataclass
class Extraction:
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)


def normalize_name(name: str) -> str:
    name = name.strip().strip(".")
    name = re.sub(r"\s+", " ", name)
    return name.casefold()


EXTRACTION_SYSTEM = (
    "You are an information-extraction engine that builds a knowledge graph. "
    "Given a passage, identify the important entities and the relationships "
    "between them. Return STRICT JSON only."
)

EXTRACTION_TEMPLATE = """Extract a knowledge graph from the passage.

Rules:
- Entities are real, specific, named things: people, organizations, places, works (films/books/songs), events, dates, and other notable nouns. Avoid generic common nouns.
- Use the entity's surface name as it appears (e.g. "Ed Wood", not "the director").
- "type" is one short uppercase label, e.g. PERSON, ORG, LOCATION, WORK, EVENT, DATE, MISC.
- "description" is a short phrase (<= 15 words) describing that entity *within this passage*.
- Relationships connect two entities extracted above, using their exact names. "description" is a short phrase describing how they are related (<= 15 words).
- Extract at most 10 entities and 12 relationships.
- Only use information stated in the passage. Do not invent facts.

Return JSON with this exact shape:
{{"entities": [{{"name": "...", "type": "PERSON", "description": "..."}}],
  "relationships": [{{"source": "...", "target": "...", "description": "..."}}]}}

Passage:
\"\"\"{passage}\"\"\"
"""


def extract_from_chunk(llm: LLMClient, passage: str, *, model: str | None = None) -> Extraction:
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": EXTRACTION_TEMPLATE.format(passage=passage)},
    ]
    data, _ = llm.chat_json(messages, model=model, max_tokens=1200)
    if not isinstance(data, dict):
        return Extraction()

    entities: list[Entity] = []
    seen: set[str] = set()
    for raw in data.get("entities", []) or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        if not name:
            continue
        norm = normalize_name(name)
        if norm in seen:
            continue
        seen.add(norm)
        entities.append(
            Entity(
                name=name,
                type=str(raw.get("type", "MISC")).strip().upper() or "MISC",
                description=str(raw.get("description", "")).strip(),
            )
        )

    name_lookup = {normalize_name(e.name): e.name for e in entities}
    relationships: list[Relationship] = []
    rel_seen: set[tuple[str, str, str]] = set()
    for raw in data.get("relationships", []) or []:
        if not isinstance(raw, dict):
            continue
        src = normalize_name(str(raw.get("source", "")))
        dst = normalize_name(str(raw.get("target", "")))
        if src not in name_lookup or dst not in name_lookup or src == dst:
            continue
        desc = str(raw.get("description", "")).strip()
        key = (src, dst, desc)
        if key in rel_seen:
            continue
        rel_seen.add(key)
        relationships.append(
            Relationship(name_lookup[src], name_lookup[dst], desc)
        )

    return Extraction(entities=entities, relationships=relationships)
