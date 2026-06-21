"""Phase 2 explorer: GNN + MCTS proof search over math dependency graphs.

The explorer replaces the LLM proof generator from Phase 1. Instead of
token-by-token generation, it searches the space of possible proof steps
(tactics) guided by graph neural network evaluations of proof states.

Sub-modules:
- dependency_graph: Graph data structures (nodes, edges, serialization)
- graph_builder: Parses Mathlib4 proofs into dependency edges
- gnn_encoder: GNN that learns embeddings over the dependency graph
- mcts: Monte Carlo Tree Search using GNN state evaluations
"""
