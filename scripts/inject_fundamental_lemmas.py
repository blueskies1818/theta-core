#!/usr/bin/env python3
"""Inject synthetic nodes for fundamental typeclass-based lemmas into the graph.

Mathlib4 provides add_comm, mul_comm, add_assoc, etc. via typeclass instances,
not standalone theorem declarations. They're the most commonly used lemmas in
proofs but are absent from the dependency graph.

This script adds synthetic nodes for the top missing lemmas, seeding their
embeddings from related graph nodes so the GNN can learn to score them.
"""

import argparse, json, sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import (
    DependencyGraph, DependencyNode, NodeType, EdgeType,
)

# Fundamental lemmas missing from the graph, with their Mathlib4 statements
FUNDAMENTAL_LEMMAS = {
    "add_comm": {
        "statement": "lemma add_comm (a b : α) [AddCommSemigroup α] : a + b = b + a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "add_assoc": {
        "statement": "lemma add_assoc (a b c : α) [AddSemigroup α] : a + b + c = a + (b + c)",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "add_zero": {
        "statement": "lemma add_zero (a : α) [AddZeroClass α] : a + 0 = a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "zero_add": {
        "statement": "lemma zero_add (a : α) [AddZeroClass α] : 0 + a = a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "add_left_neg": {
        "statement": "lemma add_left_neg (a : α) [AddGroup α] : -a + a = 0",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "add_right_neg": {
        "statement": "lemma add_right_neg (a : α) [AddGroup α] : a + -a = 0",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "sub_eq_add_neg": {
        "statement": "lemma sub_eq_add_neg (a b : α) [AddGroup α] : a - b = a + -b",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "sub_self": {
        "statement": "lemma sub_self (a : α) [AddGroup α] : a - a = 0",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "sub_add_cancel": {
        "statement": "lemma sub_add_cancel (a b : α) [AddGroup α] : a - b + b = a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "add_sub_cancel": {
        "statement": "lemma add_sub_cancel (a b : α) [AddGroup α] : a + b - b = a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "add_left_cancel": {
        "statement": "lemma add_left_cancel (a b c : α) [AddCancelCommMonoid α] (h : a + b = a + c) : b = c",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "add_right_cancel": {
        "statement": "lemma add_right_cancel (a b c : α) [AddCancelCommMonoid α] (h : b + a = c + a) : b = c",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "mul_comm": {
        "statement": "lemma mul_comm (a b : α) [CommSemigroup α] : a * b = b * a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "mul_assoc": {
        "statement": "lemma mul_assoc (a b c : α) [Semigroup α] : a * b * c = a * (b * c)",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "mul_one": {
        "statement": "lemma mul_one (a : α) [MulOneClass α] : a * 1 = a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "one_mul": {
        "statement": "lemma one_mul (a : α) [MulOneClass α] : 1 * a = a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "zero_mul": {
        "statement": "lemma zero_mul (a : α) [MulZeroClass α] : 0 * a = 0",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "mul_zero": {
        "statement": "lemma mul_zero (a : α) [MulZeroClass α] : a * 0 = 0",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "mul_inv_cancel": {
        "statement": "lemma mul_inv_cancel (a : α) [Group α] (h : a ≠ 0) : a * a⁻¹ = 1",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "inv_mul_cancel": {
        "statement": "lemma inv_mul_cancel (a : α) [Group α] (h : a ≠ 0) : a⁻¹ * a = 1",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "div_eq_mul_inv": {
        "statement": "lemma div_eq_mul_inv (a b : α) [DivisionRing α] : a / b = a * b⁻¹",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "div_one": {
        "statement": "lemma div_one (a : α) [DivisionRing α] : a / 1 = a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "one_div": {
        "statement": "lemma one_div (a : α) [DivisionRing α] : 1 / a = a⁻¹",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "eq_comm": {
        "statement": "lemma eq_comm {a b : α} : a = b ↔ b = a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "eq_self_iff_true": {
        "statement": "lemma eq_self_iff_true (a : α) : (a = a) ↔ True",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "ne_comm": {
        "statement": "lemma ne_comm {a b : α} : a ≠ b ↔ b ≠ a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "le_trans": {
        "statement": "lemma le_trans [Preorder α] {a b c : α} (h₁ : a ≤ b) (h₂ : b ≤ c) : a ≤ c",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "lt_of_lt_of_le": {
        "statement": "lemma lt_of_lt_of_le [Preorder α] {a b c : α} (h₁ : a < b) (h₂ : b ≤ c) : a < c",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "sq": {
        "statement": "lemma sq (a : α) [Pow α ℕ] : a ^ 2 = a * a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "pow_succ": {
        "statement": "lemma pow_succ (a : α) [Monoid α] (n : ℕ) : a ^ (n + 1) = a ^ n * a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "neg_neg": {
        "statement": "lemma neg_neg (a : α) [AddGroup α] : -(-a) = a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "neg_add": {
        "statement": "lemma neg_add (a b : α) [AddGroup α] : -(a + b) = -a + -b",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "neg_mul": {
        "statement": "lemma neg_mul (a b : α) [Ring α] : -a * b = -(a * b)",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "mul_neg": {
        "statement": "lemma mul_neg (a b : α) [Ring α] : a * -b = -(a * b)",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "distrib": {
        "statement": "lemma distrib (a b c : α) [Distrib α] : a * (b + c) = a * b + a * c",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "left_distrib": {
        "statement": "lemma left_distrib (a b c : α) [Distrib α] : a * (b + c) = a * b + a * c",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "right_distrib": {
        "statement": "lemma right_distrib (a b c : α) [Distrib α] : (a + b) * c = a * c + b * c",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "rfl": {
        "statement": "lemma rfl {a : α} : a = a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "Eq.refl": {
        "statement": "lemma Eq.refl {a : α} : a = a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "Eq.symm": {
        "statement": "lemma Eq.symm {a b : α} (h : a = b) : b = a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "Eq.trans": {
        "statement": "lemma Eq.trans {a b c : α} (h₁ : a = b) (h₂ : b = c) : a = c",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "congr_arg": {
        "statement": "lemma congr_arg {α β} (f : α → β) {a b : α} (h : a = b) : f a = f b",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "Nat.add_comm": {
        "statement": "lemma Nat.add_comm (a b : ℕ) : a + b = b + a",
        "domain": "Data/Nat",
        "node_type": "lemma",
    },
    "Nat.add_assoc": {
        "statement": "lemma Nat.add_assoc (a b c : ℕ) : a + b + c = a + (b + c)",
        "domain": "Data/Nat",
        "node_type": "lemma",
    },
    "Nat.mul_comm": {
        "statement": "lemma Nat.mul_comm (a b : ℕ) : a * b = b * a",
        "domain": "Data/Nat",
        "node_type": "lemma",
    },
    "Nat.mul_assoc": {
        "statement": "lemma Nat.mul_assoc (a b c : ℕ) : a * b * c = a * (b * c)",
        "domain": "Data/Nat",
        "node_type": "lemma",
    },
    "Nat.succ_eq_add_one": {
        "statement": "lemma Nat.succ_eq_add_one (n : ℕ) : Nat.succ n = n + 1",
        "domain": "Data/Nat",
        "node_type": "lemma",
    },
    "Int.add_comm": {
        "statement": "lemma Int.add_comm (a b : ℤ) : a + b = b + a",
        "domain": "Data/Int",
        "node_type": "lemma",
    },
    "abs_of_nonneg": {
        "statement": "lemma abs_of_nonneg (h : 0 ≤ a) : |a| = a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "abs_mul": {
        "statement": "lemma abs_mul (a b : α) [LinearOrderedRing α] : |a * b| = |a| * |b|",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "div_le_one": {
        "statement": "lemma div_le_one (h : 0 < b) : a / b ≤ 1 ↔ a ≤ b",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "mul_self_eq": {
        "statement": "lemma mul_self_eq (a : α) [CommRing α] : a * a = a ^ 2",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "mod_cast": {
        "statement": "lemma mod_cast {α β} [HasModCast α β] (h : α) : β",
        "domain": "Data/Int",
        "node_type": "lemma",
    },
    "Fintype.ofFinite": {
        "statement": "lemma Fintype.ofFinite (α : Type) [Finite α] : Fintype α",
        "domain": "Data/Fintype",
        "node_type": "lemma",
    },
    # -- Second batch: frequently referenced in training pairs --
    "eq_or_ne": {
        "statement": "lemma eq_or_ne (a b : α) : a = b ∨ a ≠ b",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "Ne": {
        "statement": "lemma Ne {a b : α} : a ≠ b := ...",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "ne_eq": {
        "statement": "lemma ne_eq (a b : α) : (a ≠ b) = ¬(a = b)",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "Subsingleton.elim": {
        "statement": "lemma Subsingleton.elim [Subsingleton α] (a b : α) : a = b",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "exists_ne": {
        "statement": "lemma exists_ne [Nontrivial α] (a : α) : ∃ b, b ≠ a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "mul_add": {
        "statement": "lemma mul_add (a b c : α) [Distrib α] : a * (b + c) = a * b + a * c",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "add_mul": {
        "statement": "lemma add_mul (a b c : α) [Distrib α] : (a + b) * c = a * c + b * c",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "mul_sub": {
        "statement": "lemma mul_sub (a b c : α) [Ring α] : a * (b - c) = a * b - a * c",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "sub_mul": {
        "statement": "lemma sub_mul (a b c : α) [Ring α] : (a - b) * c = a * c - b * c",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "dif_pos": {
        "statement": "lemma dif_pos (h : P) : (if P then t else e) = t",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "if_pos": {
        "statement": "lemma if_pos (h : P) : (if P then t else e) = t",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "if_neg": {
        "statement": "lemma if_neg (h : ¬P) : (if P then t else e) = e",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "Function.comp_apply": {
        "statement": "lemma Function.comp_apply (f : β → γ) (g : α → β) (x : α) : (f ∘ g) x = f (g x)",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "SetLike.mem_coe": {
        "statement": "lemma SetLike.mem_coe {S : Type} [SetLike S α] (s : S) (x : α) : x ∈ (s : Set α) ↔ x ∈ s",
        "domain": "Algebra",
        "node_type": "lemma",
    },
    "logb": {
        "statement": "lemma logb (b x : ℝ) : logb b x = log x / log b",
        "domain": "Analysis",
        "node_type": "lemma",
    },
    "two_zsmul": {
        "statement": "lemma two_zsmul (a : α) [AddCommGroup α] : (2 : ℤ) • a = a + a",
        "domain": "Algebra",
        "node_type": "lemma",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="data/graph/dependency_graph")
    parser.add_argument("--output", default="data/graph/dependency_graph")
    args = parser.parse_args()

    graph_path = _project_root / args.graph
    graph = DependencyGraph.load(graph_path)
    print(f"Loaded: {graph.summary()}")

    # Check which lemmas need to be added or overwritten
    existing_names = set(graph.node_ids)
    to_add = {}
    to_overwrite = {}
    for name, info in FUNDAMENTAL_LEMMAS.items():
        if name not in existing_names:
            to_add[name] = info
        else:
            # Check if existing node has a wrong statement
            existing_node = graph.get_node(name)
            existing_stmt = existing_node.get("statement", "") if existing_node else ""
            expected_prefix = info["statement"][:40]
            # Overwrite if the existing statement doesn't start with our expected prefix
            if expected_prefix not in existing_stmt:
                to_overwrite[name] = info

    print(f"Fundamental lemmas: {len(FUNDAMENTAL_LEMMAS)} total, "
          f"{len(to_add)} to add, "
          f"{len(to_overwrite)} to overwrite, "
          f"{len(FUNDAMENTAL_LEMMAS) - len(to_add) - len(to_overwrite)} already correct")

    all_changed = dict(to_add)
    all_changed.update(to_overwrite)

    if not all_changed:
        print("All fundamental lemmas already correct in graph. Nothing to do.")
        return

    # Add or overwrite synthetic nodes
    added = 0
    overwritten = 0
    for name, info in all_changed.items():
        node = DependencyNode(
            id=name,
            name=name,
            node_type=NodeType.LEMMA,
            statement=info["statement"],
            proof="(typeclass instance)",
            source_file="synthetic",
            domain=info["domain"],
        )
        if name in to_overwrite:
            # Overwrite: remove old node first, then add new one
            if name in graph._graph:
                graph._graph.remove_node(name)
            overwritten += 1
        graph.add_node(node)
        if name in to_add:
            added += 1

    # Rebuild indices after node changes
    graph._rebuild_indices()

    print(f"Added {added} new nodes, overwrote {overwritten} existing nodes")

    # Connect synthetic nodes to related existing nodes
    edges_added = 0
    for name, info in all_changed.items():
        domain = info["domain"]
        # Find existing nodes in the same domain
        domain_nodes = graph.get_node_ids_by_domain(domain)
        # Connect to topologically early nodes (foundational deps)
        gens = graph.topological_generations()
        early_nodes = set()
        for gen in gens[:3]:  # first 3 generations
            for nid in gen:
                if nid in domain_nodes and nid not in all_changed:
                    early_nodes.add(nid)

        # Add a few edges: new node depends on foundational nodes in its domain
        for target in list(early_nodes)[:5]:
            if target in graph._graph and graph.add_edge(name, target, EdgeType.USES_IN_PROOF):
                edges_added += 1

    print(f"Added {edges_added} edges connecting new nodes")

    # Save
    output_path = _project_root / args.output
    graph.save(output_path)
    print(f"Saved to {output_path}")

    # Show new stats
    stats = graph.get_statistics()
    print(f"\nUpdated graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
    print(f"Domains: {list(stats.get('nodes_by_domain', {}).keys())[:12]}")


if __name__ == "__main__":
    main()
