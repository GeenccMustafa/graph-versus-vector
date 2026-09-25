"""GraphRAG: build a knowledge graph in Neo4j and retrieve via relationships.

Build steps
-----------
1. Embed every chunk (used by a Neo4j vector index for local seeds).
2. Ask the LLM to extract entities + relationships per chunk.
3. Merge entities, write triples and ``MENTIONS`` links into Neo4j.
4. Detect communities of densely-connected entities and summarise them
   (used for "global"/thematic search).

Retrieval modes
---------------
* ``local``  -- vector-seed entities, expand N hops, pull related facts+chunks.
* ``global`` -- vector-match community summaries.
* ``hybrid`` -- both, which is the default.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from neo4j import GraphDatabase
from rich.progress import track

from .communities import detect_communities, summarize_community
from .config import Settings, get_settings
from .extraction import (
    Entity,
    Extraction,
    Relationship,
    extract_from_chunk,
    normalize_name,
)
from .llm import LLMClient
from .schema import RAGResult, RetrievalItem, build_answer_messages
from .text import Chunk, corpus_fingerprint

logger = logging.getLogger(__name__)

CHUNK_INDEX = "chunk_embeddings"
ENTITY_INDEX = "entity_embeddings"
COMMUNITY_INDEX = "community_embeddings"


class GraphRAG:
    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None):
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)
        self.driver = None

    # ------------------------------------------------------------- connection
    def connect(self):
        if self.driver is None:
            self.driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
            )
            self.driver.verify_connectivity()
        return self.driver

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    def _run(self, cypher: str, **params):
        return self.driver.execute_query(
            cypher, params, database_=self.settings.neo4j_database
        )[0]

    def clear(self) -> None:
        self._run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared Neo4j graph")

    def counts(self) -> dict[str, int]:
        rec = self._run(
            """
            RETURN
              COUNT { MATCH (:Document) } AS documents,
              COUNT { MATCH (:Chunk) } AS chunks,
              COUNT { MATCH (:Entity) } AS entities,
              COUNT { MATCH (:Entity)-[:RELATES_TO]->(:Entity) } AS relationships,
              COUNT { MATCH (:Community) } AS communities
            """
        )[0]
        return dict(rec)

    @property
    def is_built(self) -> bool:
        try:
            return self.counts()["entities"] > 0
        except Exception:
            return False

    def stored_fingerprint(self) -> str | None:
        try:
            recs = self._run(
                "MATCH (k:Corpus {id: 'default'}) RETURN k.fingerprint AS fp"
            )
            return recs[0]["fp"] if recs else None
        except Exception:
            return None

    def matches(self, chunks: list[Chunk]) -> bool:
        """True if the graph currently holds exactly this corpus."""
        return self.stored_fingerprint() == corpus_fingerprint(chunks)

    # ------------------------------------------------------------------ build
    def build(
        self,
        chunks: list[Chunk],
        *,
        force: bool = False,
        corpus_key: str | None = None,
    ) -> None:
        self.connect()
        fingerprint = corpus_fingerprint(chunks)
        stored = self.stored_fingerprint()

        # Only skip when the graph provably holds *this* corpus. A built graph
        # with a missing/different marker is rebuilt so switching corpora works.
        if self.is_built and not force:
            if stored == fingerprint:
                logger.info("Graph already built: %s", self.counts())
                return
            logger.info(
                "Graph corpus mismatch (stored=%s, request=%s); rebuilding",
                stored,
                fingerprint,
            )
            force = True
        if force:
            self.clear()

        self._create_indexes()
        chunk_embeddings = self.llm.embed([c.render() for c in chunks])
        self._write_chunks(chunks, chunk_embeddings)
        self._extract_and_write(chunks)
        self._build_communities()
        self._run(
            """
            MERGE (k:Corpus {id: 'default'})
            SET k.fingerprint = $fingerprint, k.corpus_key = $corpus_key,
                k.documents = $documents, k.chunks = $chunks
            """,
            fingerprint=fingerprint,
            corpus_key=corpus_key or "",
            documents=len({c.doc_id for c in chunks}),
            chunks=len(chunks),
        )

    def _create_indexes(self) -> None:
        dim = self.settings.embedding_dim
        for index, label, prop in [
            (CHUNK_INDEX, "Chunk", "embedding"),
            (ENTITY_INDEX, "Entity", "embedding"),
            (COMMUNITY_INDEX, "Community", "embedding"),
        ]:
            self._run(
                f"""
                CREATE VECTOR INDEX {index} IF NOT EXISTS
                FOR (n:{label}) ON (n.{prop})
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
            )
        for label, prop in [("Chunk", "chunk_id"), ("Entity", "name"), ("Community", "community_id")]:
            self._run(
                f"CREATE INDEX {label.lower()}_{prop}_idx IF NOT EXISTS FOR (n:{label}) ON (n.{prop})"
            )

    def _write_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        rows = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "text": c.text,
                "embedding": emb.tolist(),
            }
            for c, emb in zip(chunks, embeddings)
        ]
        self._run(
            """
            UNWIND $rows AS row
            MERGE (d:Document {doc_id: row.doc_id}) SET d.title = row.title
            MERGE (c:Chunk {chunk_id: row.chunk_id})
            SET c.doc_id = row.doc_id, c.title = row.title,
                c.text = row.text, c.embedding = row.embedding
            MERGE (c)-[:PART_OF]->(d)
            """,
            rows=rows,
        )
        logger.info("Wrote %d chunks + embeddings", len(rows))

    def _extract_all(self, chunks: list[Chunk], model: str) -> dict[str, Extraction]:
        """Run entity/relationship extraction concurrently (I/O-bound API calls)."""
        workers = max(1, self.settings.extraction_workers)
        results: dict[str, Extraction] = {}
        if workers == 1:
            for chunk in track(chunks, description="Extracting graph"):
                results[chunk.chunk_id] = extract_from_chunk(
                    self.llm, chunk.render(), model=model
                )
            return results

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    extract_from_chunk, self.llm, chunk.render(), model=model
                ): chunk
                for chunk in chunks
            }
            for future in track(
                as_completed(futures),
                total=len(futures),
                description=f"Extracting graph ({workers} workers)",
            ):
                chunk = futures[future]
                try:
                    results[chunk.chunk_id] = future.result()
                except Exception as exc:  # keep going; a bad chunk shouldn't kill the build
                    logger.warning("Extraction failed for %s: %s", chunk.chunk_id, exc)
                    results[chunk.chunk_id] = Extraction()
        return results

    def _extract_and_write(self, chunks: list[Chunk]) -> None:
        entities: dict[str, Entity] = {}
        descriptions: dict[str, list[str]] = defaultdict(list)
        relationships: dict[tuple[str, str], Relationship] = {}
        rel_weight: dict[tuple[str, str], int] = defaultdict(int)
        mentions: list[dict] = []
        model = self.settings.extractor_model

        extractions = self._extract_all(chunks, model)
        for chunk in track(chunks, description="Aggregating graph"):
            extraction = extractions.get(chunk.chunk_id, Extraction())
            for ent in extraction.entities:
                norm = ent.normalized
                if norm not in entities:
                    entities[norm] = Entity(
                        name=ent.name, type=ent.type, description=ent.description
                    )
                if ent.description and ent.description not in descriptions[norm]:
                    descriptions[norm].append(ent.description)
                mentions.append({"chunk_id": chunk.chunk_id, "entity": norm})

            for rel in extraction.relationships:
                key = rel.key
                rel_weight[key] += 1
                if key not in relationships:
                    relationships[key] = Relationship(
                        source=normalize_name(rel.source),
                        target=normalize_name(rel.target),
                        description=rel.description,
                    )

        for norm, ent in entities.items():
            merged = "; ".join(descriptions.get(norm, [])[:4])
            ent.description = merged or ent.description

        logger.info(
            "Extracted %d entities and %d relationships", len(entities), len(relationships)
        )

        # Embed entity name+description for entity-level vector search.
        entity_rows = []
        if entities:
            ent_list = list(entities.values())
            texts = [f"{e.name}: {e.description}" for e in ent_list]
            ent_embeddings = self.llm.embed(texts)
            for e, emb in zip(ent_list, ent_embeddings):
                entity_rows.append(
                    {
                        "name": e.normalized,
                        "display_name": e.name,
                        "type": e.type,
                        "description": e.description,
                        "embedding": emb.tolist(),
                    }
                )
        self._run(
            """
            UNWIND $rows AS row
            MERGE (e:Entity {name: row.name})
            ON CREATE SET e.display_name = row.display_name
            SET e.type = row.type, e.description = row.description, e.embedding = row.embedding
            """,
            rows=entity_rows,
        )

        rel_rows = [
            {
                "source": rel.source,
                "target": rel.target,
                "key": f"{rel.source}->{rel.target}",
                "description": rel.description,
                "weight": rel_weight[key],
            }
            for key, rel in relationships.items()
        ]
        self._run(
            """
            UNWIND $rows AS row
            MATCH (a:Entity {name: row.source}), (b:Entity {name: row.target})
            MERGE (a)-[r:RELATES_TO {key: row.key}]->(b)
            ON CREATE SET r.description = row.description, r.weight = row.weight
            ON MATCH SET r.weight = r.weight + row.weight
            """,
            rows=rel_rows,
        )

        if mentions:
            self._run(
                """
                UNWIND $rows AS row
                MATCH (c:Chunk {chunk_id: row.chunk_id}), (e:Entity {name: row.entity})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                rows=mentions,
            )
        logger.info("Graph written to Neo4j")

    def _summarize_communities(
        self,
        communities: list[list[str]],
        entity_by_name: dict[str, Entity],
        relationships: list[Relationship],
    ) -> list[dict]:
        model = self.settings.extractor_model
        workers = max(1, self.settings.extraction_workers)
        indexed = [(i, members) for i, members in enumerate(communities) if members]

        def summarize(members: list[str]) -> str:
            return summarize_community(
                self.llm, members, entity_by_name, relationships, model=model
            )

        summaries: dict[int, str] = {}
        if workers == 1:
            for i, members in track(indexed, description="Summarising communities"):
                summaries[i] = summarize(members)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(summarize, members): i for i, members in indexed}
                for future in track(
                    as_completed(futures),
                    total=len(futures),
                    description=f"Summarising communities ({workers} workers)",
                ):
                    i = futures[future]
                    try:
                        summaries[i] = future.result()
                    except Exception as exc:
                        logger.warning("Community summary failed: %s", exc)
                        summaries[i] = ""

        titles: dict[int, str] = {}
        entities: dict[int, list[str]] = {}
        for i, members in indexed:
            titles[i] = (
                ", ".join(
                    entity_by_name[m].name for m in members[:3] if m in entity_by_name
                )
                or f"community::{i}"
            )
            entities[i] = members

        ordered = [i for i, _ in indexed]
        embeddings = self.llm.embed([f"{titles[i]}\n{summaries[i]}" for i in ordered])

        rows = []
        for i, emb in zip(ordered, embeddings):
            rows.append(
                {
                    "community_id": f"community::{i}",
                    "title": titles[i],
                    "summary": summaries[i],
                    "size": len(entities[i]),
                    "embedding": emb.tolist(),
                    "entities": entities[i],
                }
            )
        return rows

    def _build_communities(self) -> None:
        recs = self._run(
            """
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[r:RELATES_TO]->(m:Entity)
            RETURN e.name AS source, m.name AS target, r.description AS description
            """
        )
        entity_names: list[str] = []
        relationships: list[Relationship] = []
        seen_entities: set[str] = set()
        for rec in recs:
            if rec["source"] and rec["source"] not in seen_entities:
                seen_entities.add(rec["source"])
                entity_names.append(rec["source"])
            if rec["target"]:
                relationships.append(
                    Relationship(rec["source"], rec["target"], rec["description"] or "")
                )
        # Sort for deterministic prompts (keeps the LLM cache stable across runs).
        relationships.sort(key=lambda r: (r.source, r.target, r.description))
        entity_names.sort()

        ent_recs = self._run(
            "MATCH (e:Entity) RETURN e.name AS name, e.display_name AS display, e.type AS type, e.description AS description"
        )
        entity_by_name = {
            r["name"]: Entity(r["display"] or r["name"], r["type"] or "MISC", r["description"] or "")
            for r in ent_recs
        }
        entities = [entity_by_name[n] for n in entity_names if n in entity_by_name]

        communities = detect_communities(entities, relationships)
        logger.info("Detected %d communities", len(communities))

        rows = self._summarize_communities(communities, entity_by_name, relationships)

        for row in rows:
            self._run(
                """
                MERGE (cm:Community {community_id: $community_id})
                SET cm.title = $title, cm.summary = $summary, cm.size = $size, cm.embedding = $embedding
                """,
                community_id=row["community_id"],
                title=row["title"],
                summary=row["summary"],
                size=row["size"],
                embedding=row["embedding"],
            )
            self._run(
                """
                UNWIND $entities AS en
                MATCH (e:Entity {name: en}), (cm:Community {community_id: $community_id})
                MERGE (e)-[:IN_COMMUNITY]->(cm)
                """,
                entities=row["entities"],
                community_id=row["community_id"],
            )
        logger.info("Wrote %d community summaries", len(rows))

    # -------------------------------------------------------------- retrieval
    def _vector_search_entities(self, question: str, k: int) -> list[dict]:
        emb = self.llm.embed_one(question).tolist()
        recs = self._run(
            f"""
            CALL db.index.vector.queryNodes('{ENTITY_INDEX}', $k, $embedding)
            YIELD node, score
            RETURN node.name AS name, node.display_name AS display, node.description AS description,
                   node.type AS type, score
            """,
            k=k,
            embedding=emb,
        )
        return [dict(r) for r in recs]

    def _vector_search_chunks(self, question: str, k: int) -> list[dict]:
        emb = self.llm.embed_one(question).tolist()
        recs = self._run(
            f"""
            CALL db.index.vector.queryNodes('{CHUNK_INDEX}', $k, $embedding)
            YIELD node, score
            RETURN node.chunk_id AS chunk_id, node.title AS title, node.text AS text, score
            """,
            k=k,
            embedding=emb,
        )
        return [dict(r) for r in recs]

    def _vector_search_communities(self, question: str, k: int) -> list[dict]:
        emb = self.llm.embed_one(question).tolist()
        recs = self._run(
            f"""
            CALL db.index.vector.queryNodes('{COMMUNITY_INDEX}', $k, $embedding)
            YIELD node, score
            RETURN node.community_id AS community_id, node.title AS title,
                   node.summary AS summary, score
            """,
            k=k,
            embedding=emb,
        )
        return [dict(r) for r in recs]

    def _expand_entities(self, names: list[str], hops: int) -> tuple[list[dict], list[str]]:
        """Return entity rows and relationship fact strings around the seeds."""
        if not names:
            return [], []
        hops = max(1, int(hops))
        recs = self._run(
            f"""
            MATCH (seed:Entity) WHERE seed.name IN $names
            OPTIONAL MATCH (seed)-[:RELATES_TO*1..{hops}]-(other:Entity)
            WITH collect(DISTINCT seed) + collect(DISTINCT other) AS all_nodes
            UNWIND all_nodes AS n
            WITH DISTINCT n WHERE n IS NOT NULL
            OPTIONAL MATCH (n)-[r:RELATES_TO]-(m:Entity)
            RETURN n.name AS source, n.display_name AS source_display,
                   n.description AS source_desc, r.description AS rel,
                   m.name AS target, m.display_name AS target_display,
                   m.description AS target_desc
            """,
            names=names,
        )
        entities: dict[str, dict] = {}
        facts: list[str] = []
        seen_facts: set[str] = set()
        for rec in recs:
            s = rec["source"]
            if s:
                entities.setdefault(
                    s,
                    {"name": s, "display": rec["source_display"] or s, "description": rec["source_desc"] or ""},
                )
            if rec["target"]:
                entities.setdefault(
                    rec["target"],
                    {"name": rec["target"], "display": rec["target_display"] or rec["target"], "description": rec["target_desc"] or ""},
                )
            if rec["source"] and rec["target"] and rec["rel"]:
                fact = f"{rec['source_display'] or rec['source']} -- {rec['rel']} --> {rec['target_display'] or rec['target']}"
                if fact not in seen_facts:
                    seen_facts.add(fact)
                    facts.append(fact)
        return list(entities.values()), facts

    def _chunks_for_entities(self, names: list[str], limit: int) -> list[dict]:
        if not names:
            return []
        recs = self._run(
            """
            MATCH (n:Entity)-[:MENTIONS]-(c:Chunk)
            WHERE n.name IN $names
            RETURN DISTINCT c.chunk_id AS chunk_id, c.title AS title, c.text AS text
            LIMIT $limit
            """,
            names=names,
            limit=limit,
        )
        return [dict(r) for r in recs]

    def retrieve(self, question: str, *, mode: str = "hybrid", k: int | None = None):
        k = k or self.settings.top_k
        items: list[RetrievalItem] = []
        facts: list[str] = []

        if mode in {"local", "hybrid"}:
            seeds = self._vector_search_entities(question, self.settings.graph_entity_seeds)
            seed_names = [s["name"] for s in seeds]
            # Expanded entities are only used to harvest relationship facts;
            # their individual descriptions are dropped to keep context focused.
            _, facts = self._expand_entities(seed_names, self.settings.graph_hops)

            # Direct vector evidence first (highest-signal for the answer).
            chunk_hits = self._vector_search_chunks(question, k)
            seen_chunks = {c["chunk_id"] for c in chunk_hits}
            for c in chunk_hits:
                items.append(
                    RetrievalItem(
                        id=c["chunk_id"],
                        text=f"[{c['title']}] {c['text']}",
                        score=c["score"],
                        source=c["title"],
                        kind="chunk",
                    )
                )
            # Chunks reachable through the graph from the seed entities.
            for c in self._chunks_for_entities(seed_names, k):
                if c["chunk_id"] not in seen_chunks:
                    seen_chunks.add(c["chunk_id"])
                    items.append(
                        RetrievalItem(
                            id=c["chunk_id"],
                            text=f"[{c['title']}] {c['text']}",
                            score=0.0,
                            source=c["title"],
                            kind="chunk_linked",
                        )
                    )
            # Relationship facts: the cross-passage bridges.
            if facts:
                facts = facts[: self.settings.graph_max_facts]
                items.append(
                    RetrievalItem(
                        id="graph-facts",
                        text="Relationships:\n" + "\n".join(f"- {f}" for f in facts),
                        score=1.0,
                        source="graph",
                        kind="relation",
                    )
                )
            # Seed entity descriptions (only the question-matched entities).
            for s in seeds:
                text = f"Entity: {s['display']}"
                if s["description"]:
                    text += f" -- {s['description']}"
                items.append(
                    RetrievalItem(
                        id=s["name"],
                        text=text,
                        score=s["score"],
                        source=s["display"],
                        kind="entity",
                    )
                )

        if mode in {"global", "hybrid"}:
            for cm in self._vector_search_communities(question, max(2, k // 4)):
                items.append(
                    RetrievalItem(
                        id=cm["community_id"],
                        text=f"Community report ({cm['title']}):\n{cm['summary']}",
                        score=cm["score"],
                        source=cm["title"],
                        kind="community",
                    )
                )

        return items, facts

    def answer(self, question: str, *, mode: str = "hybrid", k: int | None = None) -> RAGResult:
        t0 = time.perf_counter()
        with self.llm.tracer.span(
            "graph_rag.answer", input=question, metadata={"mode": mode}
        ) as span:
            items, facts = self.retrieve(question, mode=mode, k=k)
            context = "\n\n".join(
                f"[{i+1}] ({item.kind} | {item.source}) {item.text}"
                for i, item in enumerate(items)
            )
            chat = self.llm.chat(
                build_answer_messages(question, context), max_tokens=64, temperature=0.0
            )
            result = RAGResult(
                question=question,
                answer=chat.text.strip(),
                retrieval=items,
                latency_s=time.perf_counter() - t0,
                prompt_tokens=chat.prompt_tokens,
                completion_tokens=chat.completion_tokens,
                graph_facts=facts,
            )
            self.llm.tracer.update(
                span,
                output=result.answer,
                metadata={
                    "pipeline": "graph_rag",
                    "mode": mode,
                    "context_items": len(items),
                    "graph_facts": len(facts),
                    "latency_s": round(result.latency_s, 3),
                },
            )
        return result
