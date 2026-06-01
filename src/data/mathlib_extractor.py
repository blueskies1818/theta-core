"""Extract theorem-proof pairs from Mathlib4 .lean source files.

Parses Lean 4 files to find theorem/lemma declarations with their proofs.
Filters to specified mathematical domains for Phase 1.
"""

import json
import re
from pathlib import Path
from typing import Iterator

THEOREM_PATTERN = re.compile(
    r'(?:theorem|lemma)\s+(\w+(?:\s+\w+)*)\s*(\([^)]*\))?\s*'
    r'(?:\[[^\]]*\])?\s*(?::\s*((?:(?!:=).)+?))?\s*:=\s*by\s*\n'
    r'((?:(?!(?:^[ \t]*(?:theorem|lemma|example|def|inductive|structure|class)\s)).*\n)*)',
    re.MULTILINE,
)

EXAMPLE_PATTERN = re.compile(
    r'example\s*(\([^)]*\))?\s*(?::\s*((?:(?!:=).)+?))?\s*:=\s*by\s*\n'
    r'((?:(?!(?:^[ \t]*(?:theorem|lemma|example|def|inductive|structure|class)\s)).*\n)*)',
    re.MULTILINE,
)

# Mathlib4 domains relevant to differential geometry / GR for Phase 1
DEFAULT_DOMAINS = [
    "Analysis",
    "Geometry/Manifold",
    "Topology",
    "LinearAlgebra",
    "GroupTheory",
    "Algebra",
    "Data/Real",
    "Data/Complex",
]


def extract_theorems_from_file(filepath: Path) -> list[dict]:
    """Extract all theorems and lemmas from a single .lean file."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    results = []

    for match in THEOREM_PATTERN.finditer(content):
        name = match.group(1).strip()
        params = match.group(2) or ""
        type_sig = match.group(3) or ""
        proof = match.group(4).strip()

        if not proof:
            continue

        # Reconstruct the full theorem statement
        statement = f"theorem {name} {params} : {type_sig}".strip()
        if statement.endswith(":"):
            statement = statement[:-1].strip()

        results.append(
            {
                "name": name,
                "statement": statement,
                "proof": proof,
                "source_file": str(filepath),
            }
        )

    for match in EXAMPLE_PATTERN.finditer(content):
        params = match.group(1) or ""
        type_sig = match.group(2) or ""
        proof = match.group(3).strip()

        if not proof or not type_sig:
            continue

        statement = f"example {params} : {type_sig}".strip()

        results.append(
            {
                "name": f"example_{len(results)}",
                "statement": statement,
                "proof": proof,
                "source_file": str(filepath),
            }
        )

    return results


def iter_lean_files(
    base_dir: Path, domains: list[str] | None = None
) -> Iterator[Path]:
    """Iterate over .lean files in specified Mathlib4 domains."""
    if domains is None:
        domains = DEFAULT_DOMAINS

    mathlib_src = base_dir / "Mathlib" if (base_dir / "Mathlib").is_dir() else base_dir

    for domain in domains:
        domain_dir = mathlib_src / domain
        if not domain_dir.is_dir():
            # Try without subdirectory
            parts = domain.split("/")
            domain_dir = mathlib_src
            for part in parts:
                domain_dir = domain_dir / part
        if domain_dir.is_dir():
            yield from domain_dir.rglob("*.lean")


def extract_all_theorems(
    mathlib_dir: Path,
    domains: list[str] | None = None,
    max_theorems: int | None = None,
) -> list[dict]:
    """Extract all theorems from Mathlib4, filtered by domain."""
    all_theorems = []

    for filepath in iter_lean_files(mathlib_dir, domains):
        theorems = extract_theorems_from_file(filepath)
        all_theorems.extend(theorems)
        if max_theorems and len(all_theorems) >= max_theorems:
            break

    return all_theorems[:max_theorems] if max_theorems else all_theorems


def save_theorems_jsonl(theorems: list[dict], output_path: Path) -> None:
    """Save extracted theorems as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for theorem in theorems:
            f.write(json.dumps(theorem) + "\n")
    print(f"Saved {len(theorems)} theorems to {output_path}")


def load_theorems_jsonl(input_path: Path) -> list[dict]:
    """Load theorems from JSONL file."""
    theorems = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                theorems.append(json.loads(line))
    return theorems


def format_theorem_proof(theorem: dict) -> str:
    """Format a theorem dict into a training example string.

    Format: 'Theorem: <STATEMENT>\nProof: <PROOF>'
    """
    return f"Theorem: {theorem['statement']}\nProof: {theorem['proof']}"
