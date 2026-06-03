"""Build the math dependency graph from extracted Mathlib4 theorems.

Parses proof text to extract identifier references (which theorems/lemmas
each proof depends on), then constructs a directed dependency graph.

The resulting graph carries forward to:
- Phase 2.2 (GNN encoder): learns node embeddings from graph structure
- Phase 2.3 (MCTS proof search): uses graph to find relevant lemmas
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from src.explorer.dependency_graph import (
    DependencyGraph,
    DependencyNode,
    EdgeType,
    NodeType,
)

# ---------------------------------------------------------------------------
# Identifier extraction
# ---------------------------------------------------------------------------

# Lean 4 reserved keywords (not theorem references)
_LEAN_KEYWORDS: set[str] = {
    "fun",
    "forall",
    "exists",
    "have",
    "let",
    "show",
    "from",
    "using",
    "with",
    "where",
    "do",
    "match",
    "if",
    "then",
    "else",
    "by",
    "open",
    "set_option",
    "noncomputable",
    "partial",
    "mutual",
    "private",
    "protected",
    "scoped",
    "export",
    "initialize",
    "syntax",
    "macro",
    "elab",
    "deriving",
    "extends",
    "in",
    "at",
    "as",
    "hiding",
    "renaming",
    "def",
    "instance",
    "return",
}

# Built-in tactics (not theorem references)
_LEAN_TACTICS: set[str] = {
    "apply",
    "exact",
    "refine",
    "intro",
    "intros",
    "assumption",
    "rw",
    "rwa",
    "erw",
    "simp",
    "simpa",
    "simp_rw",
    "dsimp",
    "cases",
    "rcases",
    "obtain",
    "induction",
    "case",
    "rename",
    "constructor",
    "left",
    "right",
    "split",
    "ext",
    "infer_instance",
    "dec_trivial",
    "native_decide",
    "omega",
    "positivity",
    "linarith",
    "nlinarith",
    "norm_num",
    "norm_cast",
    "ring",
    "field_simp",
    "gcongr",
    "calc",
    "convert",
    "ac_rfl",
    "tauto",
    "aesop",
    "simp_all",
    "simp_all_arith",
    "apply_rules",
    "solve_by_elim",
    "trivial",
    "rfl",
    "done",
    "skip",
    "admit",
    "abort",
    "first",
    "any_goals",
    "all_goals",
    "try",
    "repeat",
    "focus",
    "rotate",
    "swap",
    "on_goal",
    "conv",
    "conv_lhs",
    "conv_rhs",
    "change",
    "apply_mod_cast",
    "exact_mod_cast",
    "norm_cast",
    "push_cast",
    "field_simp",
    "filter_upwards",
    "specialize",
    "generalize",
    "suffices",
    "revert",
    "clear",
    "subst",
    "injection",
    "contradiction",
    "exfalso",
    "by_contra",
    "push_neg",
    "choose",
    "set",
    "trans",
    "symm",
    "exacts",
    "refine'",
    "fail_if_success",
    "success_if_fail",
    "guard_hyp",
    "guard_target",
    "unfold",
    "delta",
    "trace",
    "trace_state",
    "simp_intro",
    "dsimp_result",
    "simp_result",
}

# Known module prefixes that indicate tactics/builtins, not theorems
_MODULE_PREFIXES: tuple[str, ...] = (
    "Std.Tactic.",
    "Lean.",
    "Lean.Parser.",
    "Lean.Elab.",
    "Init.",
    "Tactic.",
)

# Patterns for local variable names
_LOCAL_VAR_RE = re.compile(
    r"^[a-z][\d']*$|"  # single letter + digits/primes
    r"^h[A-Z]?[\d']*$|"  # hypothesis names: h, hA, h1
    r"^h[₁₂₃₄₅₆₇₈₉₀]+$|"  # unicode subscripts
    r"^ih\d*$|"
    r"^IH\d*$|"
    r"^H\d*$|"
    r"^[a-z]+_\d+$"  # indexed vars like x_1
)

# Regex for Lean identifiers: starts with a letter (including Greek),
# then letters, digits, underscores, dots (namespacing), primes, subscripts
_LEAN_IDENT_RE = re.compile(
    r"[a-zA-Zα-ωΑ-Ωλμπφψθα-ωᴀ-᷿]"
    r"[\w.'₀-₉α-ω]*"
    r"[a-zA-Zα-ωΑ-Ωλμπφψθα-ω₀-₉')]"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_likely_tactic_or_keyword(name: str) -> bool:
    """Check if a name is a Lean tactic/keyword, not a theorem reference."""
    if name in _LEAN_KEYWORDS or name in _LEAN_TACTICS:
        return True
    if name.startswith(_MODULE_PREFIXES):
        return True
    # Parser descriptions like «tactic»...
    if name.startswith("«"):
        return True
    return False


def _is_local_variable(name: str) -> bool:
    """Check if a name matches common local variable patterns."""
    return bool(_LOCAL_VAR_RE.match(name))


def _is_numeric_constant(name: str) -> bool:
    """Check if name is effectively a numeric literal."""
    cleaned = name.replace("_", "").replace(".", "").replace("'", "")
    return cleaned.replace(".", "").isdigit() or len(cleaned) == 0


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------


def extract_references(text: str) -> set[str]:
    """Extract Lean identifier references from proof or statement text.

    Filters out:
    - Lean keywords and tactics
    - Local variable names
    - Numeric constants
    - Very short names (< 3 chars)

    Returns a set of likely theorem/lemma/definition references.
    """
    if not text:
        return set()

    identifiers: set[str] = set()
    for match in _LEAN_IDENT_RE.finditer(text):
        name = match.group(0)

        # Skip very short names
        if len(name) < 3:
            continue
        # Skip known tactics/keywords
        if _is_likely_tactic_or_keyword(name):
            continue
        # Skip local variable patterns
        if _is_local_variable(name):
            continue
        # Skip numeric-looking identifiers
        if _is_numeric_constant(name):
            continue
        # Skip overly long identifiers (likely artifacts)
        if len(name) > 120:
            continue

        identifiers.add(name)

    return identifiers


# ---------------------------------------------------------------------------
# Node type and domain inference
# ---------------------------------------------------------------------------


def _infer_node_type(statement: str) -> NodeType:
    """Infer node type from the Lean declaration keyword."""
    stmt = statement.strip()
    first_word = stmt.split(maxsplit=1)[0] if stmt else ""
    return {
        "theorem": NodeType.THEOREM,
        "lemma": NodeType.LEMMA,
        "example": NodeType.EXAMPLE,
        "def": NodeType.DEFINITION,
        "inductive": NodeType.INDUCTIVE,
        "structure": NodeType.STRUCTURE,
        "class": NodeType.CLASS,
        "axiom": NodeType.AXIOM,
        "opaque": NodeType.AXIOM,
    }.get(first_word, NodeType.LEMMA)


def _infer_domain(source_file: str) -> str:
    """Infer the mathematical domain from the source file path."""
    if "Mathlib/" in source_file:
        rel = source_file.split("Mathlib/", 1)[1]
        parts = rel.split("/")
        if parts:
            return parts[0]  # e.g., "Analysis", "Algebra"
    if "../mathlib4/Mathlib/" in source_file:
        rel = source_file.split("../mathlib4/Mathlib/", 1)[1]
        parts = rel.split("/")
        if parts:
            return parts[0]
    return "Unknown"


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------


def build_dependency_graph(
    theorems_path: Path | str,
    max_theorems: int | None = None,
    min_references: int = 0,
    verbose: bool = True,
) -> DependencyGraph:
    """Build the math dependency graph from extracted theorems.

    Three phases:
    1. Load all theorems as nodes, build a name→node_id index.
    2. For each theorem's proof, extract identifier references.
    3. Match references against known names, creating directed edges.

    Args:
        theorems_path: Path to JSONL file with extracted theorems.
            Each line: {"name": ..., "statement": ..., "proof": ...,
                         "source_file": ...}
        max_theorems: Cap number of theorems loaded (useful for testing).
        min_references: Remove nodes with fewer total (in+out) edges.
            0 keeps all nodes.
        verbose: Print progress information.

    Returns:
        DependencyGraph ready for GNN training or MCTS search.
    """
    start_time = time.time()
    graph = DependencyGraph()

    # ---- Phase 1: Load theorems ----
    if verbose:
        print("Phase 1/4: Loading theorems...")

    theorems: list[dict] = []
    parse_errors = 0
    with open(theorems_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_theorems and i >= max_theorems:
                break
            line = line.strip()
            if not line:
                continue
            try:
                theorems.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1
                continue

    if verbose:
        print(f"  Loaded {len(theorems)} theorems ({parse_errors} parse errors)")

    # ---- Phase 2: Add nodes ----
    if verbose:
        print("Phase 2/4: Adding nodes...")

    # Build name index: all known theorem/lemma names in the dataset
    known_names: set[str] = set()
    name_to_id: dict[str, str] = {}  # canonical name → node id
    name_aliases: dict[str, str] = {}  # short name → canonical name

    for t in theorems:
        name = t["name"]
        node_type = _infer_node_type(t.get("statement", ""))
        domain = _infer_domain(t.get("source_file", ""))

        node = DependencyNode(
            id=name,
            name=name,
            node_type=node_type,
            statement=t.get("statement", ""),
            proof=t.get("proof", ""),
            source_file=t.get("source_file", ""),
            domain=domain,
        )
        graph.add_node(node)
        known_names.add(name)
        name_to_id[name] = name

        # Register short-name aliases for namespaced theorems
        # "Analysis.SumIntegralExpDecay.intervalIntegral_pow_mul_exp_neg_le"
        # → alias "intervalIntegral_pow_mul_exp_neg_le"
        if "." in name:
            short = name.split(".")[-1]
            if short not in name_aliases:
                name_aliases[short] = name
            # Also register two-part suffix for ambiguous short names
            parts = name.split(".")
            if len(parts) >= 2:
                two_part = ".".join(parts[-2:])
                if two_part not in name_aliases:
                    name_aliases[two_part] = name

    if verbose:
        print(f"  Added {graph.num_nodes} nodes")
        print(f"  Known names: {len(known_names)} canonical + "
              f"{len(name_aliases)} aliases")

    # ---- Phase 3: Extract dependencies ----
    if verbose:
        print("Phase 3/4: Extracting dependencies from proofs...")

    total_edges = 0
    nodes_with_deps = 0
    nodes_with_no_deps = 0

    for t in theorems:
        name = t["name"]
        if name not in graph._graph:
            continue

        proof = t.get("proof", "")
        statement = t.get("statement", "")

        # Extract references from proof text
        proof_refs = extract_references(proof) if proof else set()
        # Extract references from statement (usually fewer)
        stmt_refs = extract_references(statement) if statement else set()

        # Resolve references → canonical node IDs
        resolved = set()
        for ref in proof_refs:
            if ref == name:
                continue
            # Direct match
            if ref in known_names:
                resolved.add((ref, EdgeType.USES_IN_PROOF))
            elif ref in name_aliases:
                resolved.add((name_aliases[ref], EdgeType.USES_IN_PROOF))
            elif "." in ref:
                # Try the base name (last component)
                base = ref.split(".")[-1]
                if base in name_aliases:
                    resolved.add((name_aliases[base], EdgeType.USES_IN_PROOF))

        # Check statement references too (avoid duplicates with proof refs)
        for ref in stmt_refs:
            if ref == name:
                continue
            if ref in known_names and (ref, EdgeType.USES_IN_PROOF) not in resolved:
                resolved.add((ref, EdgeType.USES_IN_STATEMENT))
            elif ref in name_aliases:
                canon = name_aliases[ref]
                already = {(r, e) for r, e in resolved if r == canon}
                if not already:
                    resolved.add((canon, EdgeType.USES_IN_STATEMENT))

        # Add edges
        edge_count = 0
        for target_id, etype in resolved:
            if graph.add_edge(name, target_id, etype):
                edge_count += 1

        total_edges += edge_count
        if edge_count > 0:
            nodes_with_deps += 1
        else:
            nodes_with_no_deps += 1

    if verbose:
        pct = 100 * nodes_with_deps / max(graph.num_nodes, 1)
        print(f"  Added {total_edges} edges")
        print(f"  {nodes_with_deps} nodes with deps ({pct:.1f}%), "
              f"{nodes_with_no_deps} isolated")

    # ---- Phase 4: Cleanup ----
    if verbose:
        print("Phase 4/4: Finalizing...")

    if min_references > 0:
        before = graph.num_nodes
        to_remove = [
            n
            for n in graph._graph.nodes()
            if graph._graph.in_degree(n) + graph._graph.out_degree(n) < min_references
        ]
        graph._graph.remove_nodes_from(to_remove)
        if verbose:
            print(f"  Removed {len(to_remove)} low-degree nodes "
                  f"(min_references={min_references})")
        # Clean up name index
        graph._node_index = {
            k: v for k, v in graph._node_index.items() if v in graph._graph
        }

    graph._rebuild_indices()

    elapsed = time.time() - start_time
    if verbose:
        stats = graph.get_statistics()
        print(f"\nBuild complete in {elapsed:.1f}s")
        print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
        print(f"Density: {stats['density']:.6f}")
        print(f"Avg out-degree: {stats['avg_out_degree']:.1f}")
        print(f"Max topological generation: {stats['max_generation']}")
        print(f"Domains: {list(stats.get('nodes_by_domain', {}).keys())[:8]}...")

    return graph


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def build_and_save(
    theorems_path: Path | str,
    output_path: Path | str,
    max_theorems: int | None = None,
    min_references: int = 1,
) -> DependencyGraph:
    """Build the graph and save it to disk in one call.

    Args:
        theorems_path: Path to extracted theorems JSONL.
        output_path: Base path for saved graph files (extensions added).
        max_theorems: Cap on loaded theorems.
        min_references: Minimum degree to keep a node.

    Returns:
        The built DependencyGraph.
    """
    graph = build_dependency_graph(
        theorems_path=theorems_path,
        max_theorems=max_theorems,
        min_references=min_references,
        verbose=True,
    )
    graph.save(Path(output_path))
    return graph
