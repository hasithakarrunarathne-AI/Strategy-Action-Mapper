# src/ontology.py
from __future__ import annotations
from typing import Dict, List

# Minimal controlled vocabulary (extend over time)
ONTOLOGY: Dict[str, List[str]] = {
    # Education / curriculum
    "project-based learning": ["pbl", "capstone", "hands-on", "industry case studies", "interdisciplinary"],
    "curriculum redesign": ["program redesign", "course revision", "learning outcomes", "curriculum innovation"],

    # Accreditation
    "accreditation": ["abet", "eur-ace", "self-study report", "program outcomes", "continuous improvement"],

    # Research / grants
    "research output": ["publications", "indexed journals", "citations", "h-index"],
    "grant support": ["grant applications", "proposal writing", "budget planning", "pre-submission review", "external funding"],
}

def expand_with_ontology(text: str, max_terms: int = 20) -> str:
    """
    Query expansion:
    If any ontology concept appears in text, append its synonyms/related terms.
    This helps embedding-based retrieval match strategy <-> action vocabulary.
    """
    t = (text or "").lower()
    expansions: List[str] = []

    for concept, synonyms in ONTOLOGY.items():
        if concept in t:
            for s in synonyms:
                if s not in expansions:
                    expansions.append(s)
        # also allow synonym-triggering (optional)
        else:
            for s in synonyms:
                if s in t and concept not in expansions:
                    expansions.append(concept)

    if not expansions:
        return text

    expansions = expansions[:max_terms]
    return text + "\n\n[Ontology expansions]: " + ", ".join(expansions)
