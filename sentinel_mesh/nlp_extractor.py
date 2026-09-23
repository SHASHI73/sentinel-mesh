"""Stage 1: entity extraction from incoming regulatory text.

Fix vs. the original level2_nlp.py: this now returns a typed, structured
result (ExtractedFacts) instead of loose print statements, so the caller
(agent.py) can actually consume it -- feeding it into the RAG query and the
LLM prompt, instead of extracting it and throwing it away.
"""
from __future__ import annotations

from dataclasses import dataclass, field

try:
    import spacy
except ImportError:  # pragma: no cover - exercised only when spaCy is absent
    spacy = None

_MODEL_CACHE: dict[str, "spacy.language.Language"] = {}


@dataclass
class ExtractedFacts:
    numeric_terms: list[str] = field(default_factory=list)
    organizations_or_laws: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    raw_text: str = ""

    def as_query_hint(self) -> str:
        """Turn extracted facts into a short phrase usable as a RAG query
        hint, so Stage 1's output actually informs Stage 2's retrieval."""
        parts = [*self.numeric_terms, *self.organizations_or_laws]
        return " ".join(parts) if parts else self.raw_text[:80]


def _load_model(model_name: str):
    if spacy is None:
        raise RuntimeError(
            "spaCy is not installed. Run `pip install spacy` and "
            f"`python -m spacy download {model_name}`."
        )
    if model_name not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[model_name] = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not downloaded. Run: "
                f"python -m spacy download {model_name}"
            ) from exc
    return _MODEL_CACHE[model_name]


def extract_compliance_facts(text: str, model_name: str = "en_core_web_sm") -> ExtractedFacts:
    """Extract dates/quantities, org/law names, and locations from text."""
    nlp = _load_model(model_name)
    doc = nlp(text)

    facts = ExtractedFacts(raw_text=text)
    for ent in doc.ents:
        if ent.label_ in {"DATE", "TIME", "CARDINAL", "QUANTITY"}:
            facts.numeric_terms.append(ent.text)
        elif ent.label_ in {"LAW", "ORG"}:
            facts.organizations_or_laws.append(ent.text)
        elif ent.label_ in {"GPE", "LOC"}:
            facts.locations.append(ent.text)
    return facts
