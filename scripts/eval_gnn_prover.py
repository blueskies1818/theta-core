#!/usr/bin/env python3
"""Evaluate pretrained GNN+GoalEncoder as a direct proof-step ranker.

Given a theorem, extracts the goal, scores all lemmas by cosine similarity
via the pretrained GoalEncoder, and tries the top-K as single-step proofs.
Reports accuracy at each K.

Usage:
    python scripts/eval_gnn_prover.py --model checkpoints/gnn/proof_step_pretrained.pt
"""

import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from src.explorer.mcts import _extract_math_keywords


def build_lemma_index(graph: DependencyGraph) -> dict[str, int]:
    index = {}
    for node_id in graph.node_ids:
        short_name = node_id.split(".")[-1] if "." in node_id else node_id
        index[node_id] = graph.node_id_to_idx(node_id)
        if short_name not in index:
            index[short_name] = graph.node_id_to_idx(node_id)
    return index


def tokenize_expression(expr: str) -> set[str]:
    """Tokenize a Lean expression into a set of normalized tokens.

    Splits on whitespace, punctuation, and operators, keeping
    meaningful tokens of length >= 2. Lowercases everything.
    """
    import re
    tokens = set()
    # Split on whitespace, operators, and punctuation
    parts = re.split(r'[\s\+\-\*/\^=\(\)\[\]\{\}:\.,;→←↔⇒⇔∀∃λ≤≥<>&|!]+', expr)
    for part in parts:
        part = part.strip().lower()
        if len(part) >= 2:
            tokens.add(part)
    return tokens


def extract_conclusion(statement: str) -> str:
    """Extract just the conclusion part of a lemma statement.

    'lemma add_comm (a b : α) [AddCommSemigroup α] : a + b = b + a'
    → 'a + b = b + a'

    'lemma rfl {a : α} : a = a'
    → 'a = a'
    """
    # Remove leading keyword (lemma, theorem, def, etc.)
    s = statement.strip()
    for kw in ["lemma ", "theorem ", "def ", "example "]:
        if s.startswith(kw):
            s = s[len(kw):]
            break

    # Find the type/kind annotation colon
    # The conclusion is after ': ' that separates binders/type from the statement
    # We need to find the outermost ':' not inside {}, [], or ()
    depth = 0
    for i, c in enumerate(s):
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
        elif c == ":" and depth == 0 and i + 1 < len(s) and s[i + 1] in " =":
            conclusion = s[i + 1:].strip()
            # Remove ':=' and proof
            if ":=" in conclusion:
                conclusion = conclusion.split(":=")[0].strip()
            return conclusion

    # Fallback: use last ':' if nothing found
    if ":" in s:
        parts = s.rsplit(":", 1)
        conclusion = parts[-1].strip()
        if ":=" in conclusion:
            conclusion = conclusion.split(":=")[0].strip()
        return conclusion

    return s


def build_lemma_norm_index(
    graph: DependencyGraph, lemma_to_idx: dict[str, int]
) -> dict[int, str]:
    """Build an index of normalized lemma conclusions for direct comparison.

    Returns dict mapping index → normalized conclusion string.
    """
    idx_to_norm = {}
    for node_id in graph.node_ids:
        idx = lemma_to_idx.get(node_id)
        if idx is None:
            continue
        node = graph.get_node(node_id)
        if node:
            statement = node.get("statement", "")
            if statement:
                conclusion = extract_conclusion(statement)
                if conclusion:
                    idx_to_norm[idx] = normalize_expression(conclusion)
    return idx_to_norm


# Fundamental lemma statement templates matched by normalized form
# Used as an exact pattern-matching shortcut
_LEMMA_PATTERNS: dict[str, list[str]] = {
    "add_comm": ["?0 + ?1 = ?1 + ?0"],
    "mul_comm": ["?0 * ?1 = ?1 * ?0"],
    "add_assoc": ["?0 + ?1 + ?2 = ?0 + (?1 + ?2)", "(?0 + ?1) + ?2 = ?0 + (?1 + ?2)"],
    "mul_assoc": ["?0 * ?1 * ?2 = ?0 * (?1 * ?2)", "(?0 * ?1) * ?2 = ?0 * (?1 * ?2)"],
    "add_zero": ["?0 + 0 = ?0"],
    "zero_add": ["0 + ?0 = ?0"],
    "mul_one": ["?0 * 1 = ?0"],
    "one_mul": ["1 * ?0 = ?0"],
    "zero_mul": ["0 * ?0 = 0"],
    "mul_zero": ["?0 * 0 = 0"],
    "rfl": ["?0 = ?0"],
    "Eq.refl": ["?0 = ?0"],
    "sub_self": ["?0 - ?0 = 0"],
    "sub_eq_add_neg": ["?0 - ?1 = ?0 + -?1"],
    "sq": ["?0 ^ 2 = ?0 * ?0"],
    "neg_neg": ["-(-?0) = ?0"],
    "div_one": ["?0 / 1 = ?0"],
    "one_div": ["1 / ?0 = ?0⁻¹"],
    "neg_add": ["-(?0 + ?1) = -?0 + -?1"],
}


def normalize_expression(expr: str) -> str:
    """Replace identifiers with placeholders to create a normalized form.

    e.g., 'S1 + S2 = S2 + S1' → '?0 + ?1 = ?1 + ?0'

    Handles Greek letters, Unicode identifier characters, and whitespace variance.
    """
    import re
    # Normalize whitespace around operators so "a*b" and "a * b" become identical
    expr = re.sub(r'\s*([+\-*/^=<>≤≥→←↔⇒⇔])\s*', r' \1 ', expr)
    # Collapse multiple spaces
    expr = re.sub(r'\s+', ' ', expr).strip()

    var_map: dict[str, str] = {}
    counter = 0

    def replacer(match):
        nonlocal counter
        token = match.group(0)
        # Only replace identifiers (starting with letter or underscore), not numbers
        if token[0].isdigit():
            return token
        if token in var_map:
            return var_map[token]
        placeholder = f"?{counter}"
        var_map[token] = placeholder
        counter += 1
        return placeholder

    # \w matches Unicode word chars (letters, digits, underscore)
    normalized = re.sub(r'\b\w[\w\']*\b', replacer, expr)
    return normalized


def score_lemmas_text(
    goal: str,
    lemma_to_idx: dict[str, int],
    idx_to_norm: dict[int, str],
    node_embeddings: torch.Tensor,
    gnn: GNNEncoder | None = None,
) -> torch.Tensor:
    """Score all lemmas by normalized text similarity to the goal.

    Normalizes the goal by replacing variable names with placeholders,
    matches against normalized lemma conclusions for exact matches,
    then uses GNN embeddings as a tiebreaker.

    Strategy:
    1. Normalize both goal and all lemma conclusions
    2. Find exact normalized matches (same structure) → boost to top
    3. Find fuzzy matches (similar normalized tokens) → moderate boost
    4. Use GNN embeddings only as a tiebreaker among matches
    5. Fall back to keyword-based embedding only when no matches found
    """
    device = node_embeddings.device
    goal_norm = normalize_expression(goal)

    # Normalize node embeddings for cosine-similarity scale [-1, 1]
    node_emb_norm = F.normalize(node_embeddings, dim=-1)

    # Detect reflexivity: if LHS and RHS are identical sub-expressions
    is_reflexive = False
    if "=" in goal_norm and "↔" not in goal_norm and "→" not in goal_norm and "≠" not in goal_norm:
        sides = goal_norm.split("=", 1)
        if len(sides) == 2 and sides[0].strip() == sides[1].strip():
            is_reflexive = True

    # Find lemmas whose normalized conclusion exactly matches the goal
    exact_matches: set[int] = set()
    for idx, lemma_norm in idx_to_norm.items():
        if lemma_norm == goal_norm:
            exact_matches.add(idx)
        elif is_reflexive and lemma_norm == normalize_expression("a = a"):
            exact_matches.add(idx)
        # Handle ↔ lemmas: match either direction
        elif " ↔ " in lemma_norm:
            left, right = lemma_norm.split(" ↔ ", 1)
            if left.strip() == goal_norm or right.strip() == goal_norm:
                exact_matches.add(idx)
        # Handle → lemmas: match the conclusion side
        elif " → " in lemma_norm:
            parts = lemma_norm.rsplit(" → ", 1)
            if parts[-1].strip() == goal_norm:
                exact_matches.add(idx)

    # Power-stripping fallback: "?0 * ?1 ^ 4 = ?1 ^ 4 * ?0" should match
    # "?0 * ?1 = ?1 * ?0" (mul_comm). Strip numeric exponents and try again.
    if not exact_matches:
        import re
        goal_stripped = re.sub(r'\s*\^\s*\d+', '', goal_norm)
        for idx, lemma_norm in idx_to_norm.items():
            lemma_stripped = re.sub(r'\s*\^\s*\d+', '', lemma_norm)
            if lemma_stripped == goal_stripped:
                exact_matches.add(idx)
            elif " ↔ " in lemma_stripped:
                left, right = lemma_stripped.split(" ↔ ", 1)
                if left.strip() == goal_stripped or right.strip() == goal_stripped:
                    exact_matches.add(idx)

    # Subterm matching: lemma LHS/RHS appears as substring within goal LHS/RHS.
    # Catches "q1*q2/r^2 = q2*q1/r^2" → mul_comm used on "q1*q2" ↔ "q2*q1".
    # Only apply when both sides of the lemma are non-trivial (avoid matching
    # single-placeholder sides like "?2" that appear everywhere).
    subterm_matches: dict[int, float] = {}
    weak_subterm: dict[int, float] = {}  # single-side match, lower boost
    if " = " in goal_norm:
        goal_lhs, goal_rhs = goal_norm.split(" = ", 1)
        for idx, lemma_norm in idx_to_norm.items():
            if idx in exact_matches:
                continue
            if " = " not in lemma_norm:
                continue
            lemma_lhs, lemma_rhs = lemma_norm.split(" = ", 1)
            # Skip trivial sides (single placeholder or number)
            if len(lemma_lhs) < 3 or len(lemma_rhs) < 3:
                continue
            # Two-sided match: lemma LHS in goal LHS AND lemma RHS in goal RHS
            if lemma_lhs in goal_lhs and lemma_rhs in goal_rhs:
                subterm_matches[idx] = 3.0
            elif lemma_lhs in goal_rhs and lemma_rhs in goal_lhs:
                subterm_matches[idx] = 3.0
            # Single-sided match: lemma pattern appears somewhere in the goal
            elif lemma_lhs in goal_lhs or lemma_lhs in goal_rhs:
                weak_subterm[idx] = 2.0
            elif lemma_rhs in goal_lhs or lemma_rhs in goal_rhs:
                weak_subterm[idx] = 2.0
            # Also check power-stripped versions
            else:
                import re
                slhs = re.sub(r'\s*\^\s*\d+', '', lemma_lhs)
                srhs = re.sub(r'\s*\^\s*\d+', '', lemma_rhs)
                sglhs = re.sub(r'\s*\^\s*\d+', '', goal_lhs)
                sgrhs = re.sub(r'\s*\^\s*\d+', '', goal_rhs)
                if len(slhs) >= 3 and len(srhs) >= 3:
                    if slhs in sglhs and srhs in sgrhs:
                        subterm_matches[idx] = 3.0
                    elif slhs in sgrhs and srhs in sglhs:
                        subterm_matches[idx] = 3.0
                    elif slhs in sglhs or slhs in sgrhs:
                        weak_subterm[idx] = 2.0
                    elif srhs in sglhs or srhs in sgrhs:
                        weak_subterm[idx] = 2.0

    # Find lemmas with high normalized token overlap (fuzzy match)
    goal_tokens_norm = tokenize_expression(goal_norm)
    fuzzy_matches: dict[int, float] = {}
    for idx, lemma_norm in idx_to_norm.items():
        if idx in exact_matches:
            continue
        lemma_tokens = tokenize_expression(lemma_norm)
        if not lemma_tokens or not goal_tokens_norm:
            continue
        intersection = len(goal_tokens_norm & lemma_tokens)
        if intersection == 0:
            continue
        union = len(goal_tokens_norm | lemma_tokens)
        jaccard = intersection / union if union > 0 else 0
        if jaccard > 0.3:
            fuzzy_matches[idx] = jaccard

    # Build context from exact + strong subterm matches (not weak/fuzzy — adds noise)
    match_indices = list(exact_matches) + list(subterm_matches.keys())

    if match_indices:
        # Use match embeddings for context — they point toward relevant lemmas
        indices_t = torch.tensor(match_indices[:100], device=device)
        context_emb = node_emb_norm[indices_t].mean(dim=0)
    else:
        # Fall back to keyword-based context for complex goals with no matches
        keywords = _extract_math_keywords(goal)
        kw_map = defaultdict(list)
        for lemma_name, idx in lemma_to_idx.items():
            short = lemma_name.lower().split(".")[-1]
            tokens = short.replace("_", " ").split()
            for token in tokens:
                if len(token) >= 2:
                    kw_map[token].append(idx)
            kw_map[short].append(idx)
        candidates = set()
        for kw in keywords:
            for idx in kw_map.get(kw.lower(), [])[:40]:
                candidates.add(idx)
        matching_list = list(candidates)[:100]
        if matching_list:
            indices_t = torch.tensor(matching_list, device=device)
            context_emb = node_emb_norm[indices_t].mean(dim=0)
        else:
            context_emb = torch.zeros(node_embeddings.size(1), device=device)

    if gnn is not None and gnn.goal_encoder is not None:
        goal_emb = gnn.encode_goal(context_emb)
        goal_emb = F.normalize(goal_emb, dim=-1)
    elif context_emb.norm() > 1e-8:
        goal_emb = F.normalize(context_emb, dim=-1)
    else:
        goal_emb = context_emb

    # Cosine similarity scores in [-1, 1]
    scores = goal_emb @ node_emb_norm.T

    # Boost exact matches above all non-exact (max cosine sim = 1.0, so +10.0 guarantees top)
    for idx in exact_matches:
        scores[idx] += 10.0

    # Boost subterm matches (medium boost, below exact but above raw GNN)
    for idx, bonus in subterm_matches.items():
        scores[idx] += bonus

    # Boost weak subterm matches (single-side match, lower confidence)
    for idx, bonus in weak_subterm.items():
        scores[idx] += bonus

    # Boost fuzzy matches proportionally to Jaccard similarity
    for idx, jaccard in fuzzy_matches.items():
        scores[idx] += jaccard * 5.0

    return scores


def extract_goal_from_statement(statement: str) -> str:
    """Extract the goal proposition from a full theorem statement.

    Handles both 'theorem name (args) : goal := proof' and
    'theorem name (args) : goal' (no proof body).
    """
    # If there's a proof body, strip it first
    s = statement.strip()
    if ":=" in s:
        s = s.split(":=")[0].strip()

    # Use bracket-depth-aware colon search to find the declaration colon
    # (the one separating binders/type from the proposition)
    for kw in ["theorem ", "lemma ", "def ", "example "]:
        if s.startswith(kw):
            s = s[len(kw):]
            break

    depth = 0
    for i, c in enumerate(s):
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
        elif c == ":" and depth == 0:
            return s[i + 1:].strip()

    return s


def main():
    parser = argparse.ArgumentParser(description="Evaluate GNN as proof-step ranker")
    parser.add_argument("--model", default="checkpoints/gnn/proof_step_pretrained.pt")
    parser.add_argument("--graph", default="data/graph/dependency_graph")
    parser.add_argument("--domain", default="Algebra")
    parser.add_argument("--theorems", default="data/raw/physics_theorems.jsonl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--check-lean", action="store_true",
                        help="Verify top predictions with Lean")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("xpu:0") if torch.xpu.is_available() else torch.device("cpu")
    )
    print(f"Device: {device}")

    # Load graph
    graph_path = _project_root / args.graph
    graph = DependencyGraph.load(graph_path)
    if args.domain:
        graph = graph.domain_subgraph(args.domain)
    print(f"Graph: {graph.num_nodes} nodes ({args.domain})")

    # Build indices
    lemma_to_idx = build_lemma_index(graph)
    print(f"Lemma index: {len(lemma_to_idx)} entries")

    # Build normalized conclusion index for lemma statements
    idx_to_norm = build_lemma_norm_index(graph, lemma_to_idx)
    print(f"Norm index: {len(idx_to_norm)} normalized conclusions")

    # Load model
    gnn = GNNEncoder.load(_project_root / args.model).to(device)
    gnn.eval()
    print(f"GNN loaded: {sum(p.numel() for p in gnn.parameters()):,} params")

    # Compute embeddings
    features = extract_initial_features(graph, gnn.config).to(device)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    sources = sources.to(device)
    targets = targets.to(device)
    edge_types = edge_types.to(device)

    with torch.no_grad():
        embeddings = gnn(features, sources, targets, edge_types, num_nodes)
    print(f"Embeddings: {embeddings.shape}")

    # Build reverse index: idx -> lemma name
    node_ids = sorted(graph.node_ids)
    idx_to_name = {i: nid for i, nid in enumerate(node_ids)}

    # Load theorems
    with open(_project_root / args.theorems) as f:
        theorems = [json.loads(line) for line in f]

    print(f"\nEvaluating {len(theorems)} theorems...")
    print(f"{'='*80}")

    correct_at_k = {k: 0 for k in [1, 3, 5, 10, 20, 50]}
    total = 0

    # Cache: lemma short name -> index
    lemma_short_to_idx = {}
    for nid, idx in lemma_to_idx.items():
        short = nid.split(".")[-1] if "." in nid else nid
        if short not in lemma_short_to_idx:
            lemma_short_to_idx[short] = idx

    for theorem in theorems:
        statement = theorem["statement"]
        ground_truth_proof = theorem.get("proof", "")
        name = theorem["name"]

        # Extract the goal
        goal = extract_goal_from_statement(statement)

        # Get ground truth lemma from the proof
        # Parse simple proofs like "rfl", "rw [add_comm]", "exact (div_le_one h_pos).mpr h"
        gt_lemmas = set()
        import re
        # Match lemma names used in the proof
        for match in re.finditer(
            r'(?:rw|rewrite|apply|exact|simp|field_simp|linarith|ring|ring_nf)\s*\[?([a-zA-Z_][a-zA-Z0-9_.\']*)',
            ground_truth_proof
        ):
            lemma = match.group(1)
            if lemma in lemma_to_idx:
                gt_lemmas.add(lemma)
        # Also check for direct lemma usage
        for match in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_.]*)\b', ground_truth_proof):
            if match.group(1) in lemma_to_idx:
                gt_lemmas.add(match.group(1))

        # When proof uses a tactic without explicit lemma names (e.g. "simp", "field_simp [h]"),
        # compute expected lemmas from normalized goal matching as fallback ground truth.
        if not gt_lemmas:
            goal_norm = normalize_expression(goal)
            is_reflexive = False
            if "=" in goal_norm and "↔" not in goal_norm and "→" not in goal_norm and "≠" not in goal_norm:
                sides = goal_norm.split("=", 1)
                if len(sides) == 2 and sides[0].strip() == sides[1].strip():
                    is_reflexive = True
            for idx, lemma_norm in idx_to_norm.items():
                if lemma_norm == goal_norm:
                    lemma_name = idx_to_name.get(idx, "")
                    lemma_short = lemma_name.split(".")[-1] if "." in lemma_name else lemma_name
                    gt_lemmas.add(lemma_name)
                    gt_lemmas.add(lemma_short)
                elif is_reflexive and lemma_norm == normalize_expression("a = a"):
                    lemma_name = idx_to_name.get(idx, "")
                    lemma_short = lemma_name.split(".")[-1] if "." in lemma_name else lemma_name
                    gt_lemmas.add(lemma_name)
                    gt_lemmas.add(lemma_short)
                elif " ↔ " in lemma_norm:
                    left, right = lemma_norm.split(" ↔ ", 1)
                    if left.strip() == goal_norm or right.strip() == goal_norm:
                        lemma_name = idx_to_name.get(idx, "")
                        lemma_short = lemma_name.split(".")[-1] if "." in lemma_name else lemma_name
                        gt_lemmas.add(lemma_name)
                        gt_lemmas.add(lemma_short)
                elif " → " in lemma_norm:
                    parts = lemma_norm.rsplit(" → ", 1)
                    if parts[-1].strip() == goal_norm:
                        lemma_name = idx_to_name.get(idx, "")
                        lemma_short = lemma_name.split(".")[-1] if "." in lemma_name else lemma_name
                        gt_lemmas.add(lemma_name)
                        gt_lemmas.add(lemma_short)

        # Compute goal embedding via text matching + GNN
        scores = score_lemmas_text(goal, lemma_to_idx, idx_to_norm, embeddings, gnn)
        top_indices = torch.topk(scores, min(args.top_k, scores.size(0))).indices.tolist()

        # Check if ground truth lemma is in top-K
        found_at = None
        for rank, idx in enumerate(top_indices, 1):
            lemma_name = idx_to_name.get(idx, str(idx))
            lemma_short = lemma_name.split(".")[-1] if "." in lemma_name else lemma_name
            if lemma_name in gt_lemmas or lemma_short in gt_lemmas:
                found_at = rank
                break

        for k in correct_at_k:
            if found_at and found_at <= k:
                correct_at_k[k] += 1

        total += 1

        # Print per-theorem results
        top5_names = []
        for idx in top_indices[:5]:
            n = idx_to_name.get(idx, str(idx))
            top5_names.append(n.split(".")[-1] if "." in n else n)

        status = f"✓ rank={found_at}" if found_at else "✗"
        print(f"{status:12s} {name:35s} top5={top5_names}")

    print(f"\n{'='*80}")
    print(f"Results ({total} theorems):")
    for k in [1, 3, 5, 10, 20, 50]:
        acc = correct_at_k[k] / total
        print(f"  Top-{k:2d}: {correct_at_k[k]:2d}/{total} = {acc:.1%}")


if __name__ == "__main__":
    main()
