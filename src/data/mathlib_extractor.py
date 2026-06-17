"""Extract theorem-proof pairs from Mathlib4 .lean source files.

Parses Lean 4 files to find theorem/lemma declarations with their proofs.
Handles both ``:= by`` tactic proofs and ``:=`` term-style proofs using
indentation-based block detection.

Filters to specified mathematical domains for Phase 1.
"""

import json
import re
from pathlib import Path
from typing import Iterator

# Keywords that start a new top-level declaration or command.
# When we hit one of these at the base indentation level, the proof block ends.
_STOP_KEYWORDS = re.compile(
    r'^\s*(?:'
    r'theorem\s|lemma\s|example\s|def\s|inductive\s|structure\s|class\s|'
    r'instance\s|axiom\s|opaque\s|'
    r'alias\s|attribute\s|'
    r'variable\s|variables\s|'
    r'section\s|namespace\s|end\s|'
    r'#(?:check|eval|print|guard|guard_msgs|synth|reduce|time)|'
    r'@\[|'
    r'/--|/-!'
    r')',
    re.MULTILINE,
)

# Match a theorem/lemma/example declaration at any indentation
_DECL_START = re.compile(r'^\s*(theorem|lemma|example)\s+')

# Match := (optionally followed by 'by') on a line
_ASSIGN = re.compile(r':=\s*(by)?\s*(.*)')


# Mathlib4 domains — ALL relevant mathematical domains (excluding meta/testing)
# Phase 2+ expansion: include all substantive math domains for full graph coverage
DEFAULT_DOMAINS = [
    "Algebra",
    "AlgebraicGeometry",
    "AlgebraicTopology",
    "Analysis",
    "CategoryTheory",
    "Combinatorics",
    "Computability",
    "Condensed",
    "Data",
    "Dynamics",
    "FieldTheory",
    "Geometry",
    "GroupTheory",
    "InformationTheory",
    "LinearAlgebra",
    "Logic",
    "MeasureTheory",
    "ModelTheory",
    "NumberTheory",
    "Order",
    "Probability",
    "RepresentationTheory",
    "RingTheory",
    "SetTheory",
    "Topology",
]


def _line_indent(line: str) -> int:
    """Return the indentation level (column) of the first non-whitespace char."""
    return len(line) - len(line.lstrip())


def extract_theorems_from_file(filepath: Path) -> list[dict]:
    """Extract all theorems and lemmas from a single .lean file.

    Uses indentation-based block detection to handle both ``:= by``
    tactic proofs and ``:=`` term-style proofs.
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    lines = content.split("\n")
    results = []
    i = 0

    while i < len(lines):
        line = lines[i]
        decl_match = _DECL_START.match(line)
        if not decl_match:
            i += 1
            continue

        decl_kind = decl_match.group(1)  # 'theorem', 'lemma', or 'example'
        base_indent = _line_indent(line)

        # Collect declaration lines until we find :=
        # Check the declaration line itself first (:= on same line is common)
        decl_lines = []
        found_assign = False
        assign_line_idx = i

        if _ASSIGN.search(line):
            # := is on the same line as the theorem/lemma keyword
            found_assign = True
            assign_line_idx = i
            decl_lines.append(line)
        else:
            decl_lines.append(line)
            j = i + 1

            # Look for := within the next 30 lines (declarations with many
            # binders can span several lines)
            while j < len(lines) and (j - i) < 30:
                cur = lines[j]
                decl_lines.append(cur)

                if _ASSIGN.search(cur):
                    found_assign = True
                    assign_line_idx = j
                    break

                # Stop if we hit a new declaration before finding :=
                if _DECL_START.match(cur):
                    break

                # Stop if we return to base indentation (declaration ended
                # without a proof — usually a signature-only axiom/def)
                if cur.strip() and _line_indent(cur) <= base_indent:
                    break

                j += 1

        if not found_assign:
            i = j + 1
            continue

        # Parse the := line to determine proof style
        assign_line = lines[assign_line_idx]
        assign_match = _ASSIGN.search(assign_line)
        if not assign_match:
            i = assign_line_idx + 1
            continue

        is_by = assign_match.group(1) == "by"
        after_assign = assign_match.group(2) or ""

        # Extract the proof body — an indented block following := (by)
        proof_lines = []

        if after_assign.strip():
            # Content on the same line as := (by)
            # For single-line proofs like ':= by rfl' or ':= Iff.rfl'
            proof_lines.append(after_assign.strip())
            # Check if there's more on subsequent indented lines
            k = assign_line_idx + 1
            while k < len(lines):
                cur = lines[k]
                if not cur.strip():
                    proof_lines.append("")
                    k += 1
                    continue
                if _line_indent(cur) <= base_indent:
                    break
                proof_lines.append(cur.strip())
                k += 1
        else:
            # := by on its own line — proof starts on next indented line(s)
            k = assign_line_idx + 1
            while k < len(lines):
                cur = lines[k]
                if not cur.strip():
                    # Empty lines within proof block are fine
                    proof_lines.append("")
                    k += 1
                    continue
                if _line_indent(cur) <= base_indent:
                    break
                proof_lines.append(cur.strip())
                k += 1

        proof = "\n".join(proof_lines).strip()

        if not proof:
            i = assign_line_idx + 1
            continue

        # Filter out trivial / sorry proofs
        if proof.strip() in (".rfl", "rfl", "trivial", "sorry", "?.sorry"):
            i = assign_line_idx + 1
            continue

        # Reconstruct the theorem statement (everything before :=)
        # Take lines from declaration start to the := line,
        # stripping the := and everything after it
        stmt_lines = []
        for k, dl in enumerate(decl_lines):
            if k == len(decl_lines) - 1:
                # Last line contains := — strip it
                stripped = re.sub(r'\s*:=\s*by\s*$', '', dl)
                stripped = re.sub(r'\s*:=\s*(.+)$', '', stripped)  # term-style
                if stripped.strip():
                    stmt_lines.append(stripped)
            else:
                stmt_lines.append(dl)

        statement = "\n".join(stmt_lines).strip()
        # Collapse multi-line statements to single line for cleaner training data
        statement = re.sub(r'\s+', ' ', statement)

        # Extract the name (examples get an auto-generated name)
        name_match = re.search(r'(?:theorem|lemma|example)\s+([\w.]+)?', statement)
        if name_match and name_match.group(1):
            name = name_match.group(1)
        elif decl_kind == "example":
            name = f"example_{len(results)}"
        else:
            name = f"anon_{len(results)}"

        results.append(
            {
                "name": name,
                "statement": statement,
                "proof": proof,
                "source_file": str(filepath),
            }
        )

        # Advance past the proof block
        i = assign_line_idx + 1
        # Skip remaining proof lines (already consumed above)
        while i < len(lines):
            cur = lines[i]
            if not cur.strip():
                i += 1
                continue
            if _line_indent(cur) <= base_indent:
                break
            i += 1

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
