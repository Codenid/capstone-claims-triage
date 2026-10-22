"""Summary and bibliography of the PB-24 Theorizer runs (models_plan.md §32.4).

Reads the exported theory stores under `.asta/generate-theories/<slug>/data/`
(outside Git) and writes two versioned files: `reports/modeling/theorizer_pb24.json`
(papers, theories, novelty judgments and the citation keys per question) and
`reports/informe/referencias_theorizer.bib` (the papers whose extracted findings
support at least one theory). Keys already present in the hand-written
`reports/informe/referencias.bib` are cited from there and not duplicated.

    uv run --no-sync python scripts/theorizer_bib.py
"""

from __future__ import annotations

import collections
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / ".asta" / "generate-theories"
SUMMARY = ROOT / "reports/modeling/theorizer_pb24.json"
HAND_BIB = ROOT / "reports/informe/referencias.bib"
OUT_BIB = ROOT / "reports/informe/referencias_theorizer.bib"
QUESTIONS = {
    "q1": (
        "2026-10-21-cusum-echo",
        "cusum_echo",
        "Eco de las ráfagas: diseño de la línea base de M11",
        "gen-20261021-c09f9734",
    ),
    "q2": (
        "2026-10-21-pattern-stability",
        "pattern_stability",
        "Estabilidad del patrón: asignación fuera de muestra",
        "gen-20261021-e8cd9fe2",
    ),
    "q3": (
        "2026-10-21-tabpfn-finetune",
        "tabpfn_finetune",
        "T1: cabezal tabular en contexto frente a ajuste fino",
        "gen-20261021-0d334f20",
    ),
    "q4": (
        "2026-10-21-bayesian-counts",
        "bayesian_counts",
        "Conteos bayesianos: memoria corta frente a espacio de estados",
        "gen-20261021-a2026bc1",
    ),
}
DIMENSIONS = (
    "phenomenon_effect",
    "explanatory_mechanistic",
    "unification",
    "generalization_scope_expansion",
    "constraint_limitation",
    "conceptual_reframing_abstraction",
    "empirical_synthesis_meta_regulariry",
)
STOP = {"a", "an", "the", "of", "for", "on", "in", "and", "to", "with", "from", "by"}


def slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def bib_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def protect_title(title: str) -> str:
    """Keep acronyms and model names capitalised in BibTeX."""
    words = []
    for word in title.split():
        core = re.sub(r"[^A-Za-z0-9]", "", word)
        mixed = re.search(r"[a-z][A-Z]|[A-Z].*[A-Z]", core)
        words.append(
            "{" + word + "}" if len(core) > 1 and (core.isupper() or mixed) else word
        )
    return " ".join(words)


def hand_keys() -> set[str]:
    text = HAND_BIB.read_text(encoding="utf-8")
    return set(re.findall(r"^@\w+\{([^,]+),", text, flags=re.M))


def entry(key: str, paper: dict[str, Any], meta: dict[str, Any]) -> str:
    authors = [a["name"] for a in (meta.get("authors") or []) if a.get("name")]
    author = " and ".join(authors[:6]) + (" and others" if len(authors) > 6 else "")
    title = (paper.get("title") or meta.get("title") or "").strip()
    year = paper.get("publication_year") or meta.get("year")
    venue = (meta.get("venue") or "").strip()
    ids = meta.get("externalIds") or {}
    fields = [
        f"  author = {{{bib_escape(author)}}}",
        f"  title = {{{bib_escape(protect_title(title))}}}",
        f"  year = {{{year}}}",
    ]
    journal = venue and not venue.lower().startswith("arxiv")
    if journal:
        fields.append(f"  journal = {{{bib_escape(venue)}}}")
    if ids.get("DOI"):
        fields.append(f"  doi = {{{ids['DOI']}}}")
    elif ids.get("ArXiv"):
        fields.append(f"  eprint = {{{ids['ArXiv']}}}")
        fields.append("  archivePrefix = {arXiv}")
    kind = "article" if journal else "misc"
    return f"@{kind}{{{key},\n" + ",\n".join(fields) + "\n}\n"


def citation_key(
    paper: dict[str, Any], meta: dict[str, Any], used: dict[str, str]
) -> str | None:
    title = (paper.get("title") or meta.get("title") or "").strip()
    year = paper.get("publication_year") or meta.get("year")
    authors = [a["name"] for a in (meta.get("authors") or []) if a.get("name")]
    if not title or not year or not authors:
        return None
    surname = authors[0].split()[-1]
    words = [w for w in re.findall(r"[A-Za-z]+", title) if w.lower() not in STOP]
    key = f"{slug(surname)}{year}{slug(words[0] if words else 'paper')}"
    if key in used and used[key] != title:
        key += slug(words[1]) if len(words) > 1 else "b"
    used.setdefault(key, title)
    return key


def main() -> None:
    existing = hand_keys()
    used: dict[str, str] = {}
    entries: dict[str, str] = {}
    summary: dict[str, Any] = {
        "date": "2026-10-21",
        "tool": "Asta Theorizer via asta generate-theories (piecemeal: find-and-extract, form-theory, evaluate-novelty)",
        "papers_per_question": 30,
        "questions": {},
    }
    totals: collections.Counter[str] = collections.Counter()
    theories_total = 0
    for question, (folder, short, title, run_id) in QUESTIONS.items():
        store = json.loads(
            (EXPORTS / folder / "data" / "theory-store.json").read_text(
                encoding="utf-8"
            )
        )
        papers = store["paperstore"]
        results = store["extraction_results"]
        theories = store["theories"]
        findings = collections.Counter()
        for result in results.values():
            findings[result["paper_id"]] += len(result.get("extracted_data") or [])
        degrees: collections.Counter[str] = collections.Counter()
        for evaluation in store.get("theory_evaluations", {}).values():
            for dimension in DIMENSIONS:
                degree = (evaluation.get(dimension) or {}).get("degree_of_novelty")
                if degree:
                    degrees[degree] += 1
        supporting = {
            results[rid]["paper_id"]
            for theory in theories.values()
            for rid in theory.get("supporting_evidence_ids") or []
            if rid in results
        }
        keys = []
        for pid in sorted(supporting):
            paper = papers.get(pid) or {}
            meta = paper.get("s2_metadata") or {}
            key = citation_key(paper, meta, used)
            if key is None:
                continue
            keys.append(key)
            if key not in existing and key not in entries:
                entries[key] = entry(key, paper, meta)
        totals.update(degrees)
        theories_total += len(theories)
        summary["questions"][question] = {
            "slug": short,
            "title": title,
            "run_id": run_id,
            "papers": len(papers),
            "full_text": sum(
                1 for p in papers.values() if p.get("status") == "full-text"
            ),
            "with_findings": sum(1 for v in findings.values() if v > 0),
            "findings": sum(findings.values()),
            "theories": [
                {
                    "id": tid,
                    "name": t.get("name"),
                    "evidence": len(t.get("supporting_evidence_ids") or []),
                }
                for tid, t in theories.items()
            ],
            "novelty_judgments": dict(degrees),
            "evidence_keys": sorted(set(keys)),
        }
    summary["totals"] = {
        "theories": theories_total,
        "novelty_judgments": dict(totals),
        "judgments": sum(totals.values()),
        "evidence_papers": len(used),
    }
    SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    header = (
        "% Papers whose extracted findings support the Asta Theorizer theories of\n"
        "% PB-24 (models_plan.md §32.4). Generated by scripts/theorizer_bib.py from\n"
        "% the exported theory stores; do not edit by hand. Cited from the closing\n"
        "% chapter of the report.\n\n"
    )
    OUT_BIB.write_text(
        header + "\n".join(entries[k] for k in sorted(entries)),
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"{theories_total} theories, {len(used)} supporting papers, {len(entries)} new entries"
    )
    print(f"-> {SUMMARY.relative_to(ROOT)}, {OUT_BIB.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
