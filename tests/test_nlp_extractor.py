"""Tests for nlp_extractor.py, including the as_query_hint() wiring that
lets Stage 1 output actually feed Stage 2's retrieval (previously dead)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel_mesh.nlp_extractor import ExtractedFacts


def test_query_hint_uses_numeric_and_org_terms():
    facts = ExtractedFacts(
        numeric_terms=["30 days"],
        organizations_or_laws=["GDPR"],
        raw_text="A long regulation about data retention that should not be used verbatim.",
    )
    hint = facts.as_query_hint()
    assert "30 days" in hint
    assert "GDPR" in hint


def test_query_hint_falls_back_to_raw_text_when_no_entities_found():
    facts = ExtractedFacts(raw_text="Short text with nothing spaCy would tag.")
    hint = facts.as_query_hint()
    assert hint == facts.raw_text[:80]


# NOTE: extract_compliance_facts() itself requires the spaCy model
# `en_core_web_sm` to be downloaded (`python -m spacy download en_core_web_sm`).
# That integration path is exercised manually / in CI with the model
# installed, not here, to keep the unit test suite fast and dependency-free.
