"""Community detection + LLM-generated community summaries (global search).

Global search answers "big picture" questions by summarising clusters of
densely connected entities, rather than retrieving individual chunks.
"""

from __future__ import annotations

import logging

import networkx as nx

from .extraction import Entity, Relationship, normalize_name
from .llm import LLMClient

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM = (
    "You write concise analytical reports about clusters of entities in a "
    "knowledge graph. Base everything strictly on the provided data."
)

SUMMARY_TEMPLATE = """Write a short report (3-5 sentences) about this community of
entities. Describe the main theme, the most important entities, and how they
are connected. Do not invent facts.

Entities:
{entities}

Relationships:
{relationships}
"""


def detect_communities(
    entities: list[Entity],
    relationships: list[Relationship],
    *,
    resolution: float = 1.0,
) -> list[list[str]]:
    """Return communities as lists of normalised entity names."""
    graph = nx.Graph()
    for e in entities:
        graph.add_node(normalize_name(e.name))
    for r in relationships:
        s, t = r.key
        if s in graph and t in graph and s != t:
            graph.add_edge(s, t)

    if graph.number_of_edges() == 0:
        logger.info("No relationship edges; treating each entity as its own community")
        return [[n] for n in graph.nodes]

    try:
        comms = nx.community.greedy_modularity_communities(graph, resolution=resolution)
        return [sorted(c) for c in comms]
    except Exception:  # pragma: no cover - fall back to a cheaper method
        comms = nx.community.label_propagation_communities(graph)
        return [sorted(c) for c in comms]


def summarize_community(
    llm: LLMClient,
    entity_names: list[str],
    entity_by_name: dict[str, Entity],
    relationships: list[Relationship],
    *,
    model: str | None = None,
) -> str:
    """Return an LLM-written summary report for one community."""
    name_set = set(entity_names)
    ent_lines = []
    for n in entity_names:
        e = entity_by_name.get(n)
        if e:
            ent_lines.append(f"- {e.name} ({e.type}): {e.description}")
        else:
            ent_lines.append(f"- {n}")
    rel_lines = []
    for r in relationships:
        s, t = r.key
        if s in name_set and t in name_set:
            rel_lines.append(f"- {r.source} -- {r.description or 'related to'} --> {r.target}")
    if not rel_lines:
        rel_lines.append("- (no explicit relationships)")

    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": SUMMARY_TEMPLATE.format(
                entities="\n".join(ent_lines),
                relationships="\n".join(rel_lines),
            ),
        },
    ]
    result = llm.chat(messages, model=model, max_tokens=400, temperature=0.0)
    return result.text.strip()
