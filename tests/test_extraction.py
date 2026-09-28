"""Unit tests for entity/relationship names and extraction parsing."""

from __future__ import annotations

from rag_compare.extraction import Entity, Relationship, extract_from_chunk, normalize_name


def test_normalize_name_trims_and_casefolds() -> None:
    assert normalize_name("  Ed Wood.  ") == "ed wood"
    assert normalize_name("Tim   Burton") == "tim burton"


def test_entity_normalized_and_relationship_key() -> None:
    assert Entity(name="Ed Wood").normalized == "ed wood"
    rel = Relationship(source="Tim Burton", target="Ed Wood")
    assert rel.key == ("tim burton", "ed wood")


def test_extract_from_chunk_handles_non_dict_output(fake_llm) -> None:
    extraction = extract_from_chunk(fake_llm, "some passage")
    assert extraction.entities == []
    assert extraction.relationships == []
