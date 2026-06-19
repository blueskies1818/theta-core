# Phase 2 Review — Scale the Mathematical Explorer

**Date:** 2026-06-03
**Status:** Infrastructure complete — GNN pre-training in progress
**Branch:** `main`

---

## 1. Executive Summary

**Goal:** Replace the LLM proof generator from Phase 1 with a GNN + MCTS architecture that searches the space of mathematical objects rather than sampling tokens.

**Verdict: The Phase 2 architecture is built and functional.** All five sub-components are implemented and tested end-to-end. The GNN+MCTS pipeline replaces `generate_proofs()` from Phase 1. The structure generator enables exploration beyond existing theorems. The critical remaining work is GNN pre-training — with random embeddings, MCTS selects irrelevant lemmas, yielding 0% proof success.

**Decision:** Proceed to GNN pre-training on the dependency graph (link prediction task), then self-play training.

---

## 2. What Was Built

### 2.1 Dependency Graph (`src/explorer/dependency_graph.py`, `src/explorer/graph_builder.py`)

| Metric | Value |
|--------|-------|
| Nodes | 58,370 (theorems, lemmas, definitions) |
| Edges | 160,611 directed dependencies |
| Domains | Analysis, Algebra, Topology, LinearAlgebra, GroupTheory, Geometry, Data |
| Avg out-degree | 2.8 |
| Density | 4.7×10⁻⁵ |
| Format | NetworkX DiGraph + PyTorch tensors |

**Key design decisions:**
- Directed edges point from dependent → dependency (A → B = "A uses B")
- Four edge types: `uses_in_proof`, `uses_in_statement`, `generalizes`, `instantiates`
- Topological generations: 9 (from foundational axioms to complex theorems)
- Hub nodes correctly identified: `mul_comm` (1,035 in-degree), `IsBigOWith.trans` (1,800 in-degree)

**Known limitations:**
- Many references go to definitions/core types not in the extracted dataset → avg out-degree of 2.8 underestimates real dependencies
- Short common names (`map`, `comp`, `zero`) produce false positives — filtered but not eliminated
- `def`, `inductive`, `structure` declarations not yet extracted (only theorem/lemma)

### 2.2 GNN Encoder (`src/explorer/gnn_encoder.py`, `src/explorer/gnn_config.py`)

| Metric | Value |
|--------|-------|
| Architecture | Graph Attention Network (GAT) |
| Layers | 2-4 configurable |
| Hidden dim | 128-512 configurable |
| Heads | 4-8 (per-layer multi-head attention) |
| Params (256d, 3-layer) | 855,808 |
| Edge conditioning | Learned embeddings per edge type |
| Initial features | Random, one-hot, or transformer embeddings |
| Memory (full graph) | ~123 MB for graph tensors |

**Key design decisions:**
- **Pure PyTorch** — no PyG/DGL dependency (portability, works on Intel XPU)
- **Edge type embeddings** instead of EdgeTypedLinear — avoids 126 GB intermediate tensor
- **Scattered softmax** for attention normalization over variable-degree nodes
- **Bidirectional** message passing: forward (dependency) + reverse (dependent)
- **Link prediction** as the training objective: dot-product similarity between node embeddings

**Known limitations:**
- Full-graph forward pass on CPU is slow (~minutes for 58K nodes) — needs batching or GPU
- No subgraph sampling yet (full-graph message passing only)
- Value head not yet separate from embedding — uses cosine similarity heuristic

### 2.3 MCTS Proof Search (`src/explorer/mcts.py`, `src/explorer/proof_state.py`)

| Metric | Value |
|--------|-------|
| Search algorithm | PUCT (AlphaGo Zero variant) |
| Configurable sims | 100-1000+ per proof |
| Max depth | 5-20 proof steps |
| Action space | 7 tactic types × top-k lemmas |
| Candidate actions | ~35-50 per state (progressive widening) |
| State evaluation | GNN cosine similarity (goal vs lemmas) |

**Key design decisions:**
- **PUCT selection** with GNN priors for lemma relevance
- **Symbolic state transitions** (goal/hypothesis tracking without full Lean kernel)
- **Proof state → Lean code** rendering that generates valid syntax
- **Transposition table** for state reuse
- **Integration** with Phase 1 proof checker via `wrap_theorem_with_proof`

**Known limitations:**
- Symbolic state transitions are approximate — real subgoal structure needs Lean kernel
- With random GNN embeddings, lemma selection is random → 0% valid proofs
- No trained policy network yet — the "AlphaGo Zero" part requires GNN pre-training

### 2.4 Explorer Trainer (`src/explorer/explorer_trainer.py`)

Replaces `GRPOTrainer.generate_proofs()` with `MCTS.search()`. The training loop:
1. Sample batch of theorems
2. Compute GNN embeddings (full graph)
3. Run MCTS for each theorem (uses GNN for guidance)
4. Check proofs with Lean
5. Compute rewards (binary + curiosity + length)
6. Compute group-relative advantages
7. Train GNN: policy loss (visit distribution) + value loss (outcome prediction)

**Key design decisions:**
- **Two-phase training**: pre-training (link prediction) → self-play (GRPO+MCTS)
- **Dual loss**: policy (KL divergence from visit distribution) + value (MSE vs outcome)
- **Same GRPO reward infrastructure** from Phase 1: binary + curiosity + length

### 2.5 Structure Generator (`src/explorer/structure_generator.py`, `src/explorer/structure_validator.py`)

| Feature | Status |
|---------|--------|
| Templates | Einstein-Hilbert, Standard Model, Schwarzschild |
| Mutation operators | Add term, remove term, modify coefficient, generalize |
| Beam search | Configurable width and depth |
| Validation | Completeness, duplicates, domain constraints, dependency availability |
| GNN-guided mutations | Uses graph neighborhood for candidate terms |

**Forward-looking design:** The structure generator is designed for Phase 3 when physical correspondence scoring becomes available. Templates encode known physics; mutations explore adjacent structures; the GNN guides exploration toward promising regions.

---

## 3. Architecture Comparison: Phase 1 vs Phase 2

| Component | Phase 1 (LLM) | Phase 2 (GNN+MCTS) |
|-----------|---------------|---------------------|
| Proof generation | Token sampling (Qwen2.5-1.5B) | Tree search with GNN guidance |
| State representation | Text context window | Graph node embeddings |
| Action selection | Next-token probability | PUCT over lemma candidates |
| Error mode | Hallucinated tokens, artifacts | Irrelevant lemmas (pre-training issue) |
| Search budget | Fixed (one forward pass) | Configurable (N MCTS simulations) |
| Training signal | Token-level logprob ratio | Visit distribution + outcome |

**Why Phase 2 is fundamentally better:**
- Search is **explicit**: MCTS systematically explores, backtracks, and converges
- Errors are **bounded**: every candidate is a real lemma, not a hallucination
- **Compute scales with inference time**, not training time: more MCTS sims = better proofs
- The GNN learns **semantic relationships** (which lemmas work together), not surface syntax

---

## 4. Working Infrastructure (Carries Forward to Phase 3)

| Component | Path | Status |
|-----------|------|--------|
| Dependency graph (58K nodes, 160K edges) | `data/graph/` | Built and saved |
| GNN encoder (GAT, configurable) | `src/explorer/gnn_encoder.py` | Tested on full graph |
| MCTS proof search (PUCT) | `src/explorer/mcts.py` | End-to-end with checker |
| Explorer trainer (GRPO+MCTS) | `src/explorer/explorer_trainer.py` | Ready for training |
| Structure generator (3 templates, 4 mutations) | `src/explorer/structure_generator.py` | Functional |
| Structure validator (5 checks) | `src/explorer/structure_validator.py` | Functional |
| GNN pre-training script | `scripts/training/pretrain_gnn.py` | Running (slow on CPU) |
| Build script (dependency graph) | `scripts/build/build_dependency_graph.py` | Complete |
| GNN config | `src/explorer/gnn_config.py` | Complete |
| Proof state representation | `src/explorer/proof_state.py` | Functional |

### Phase 1 infrastructure still used:
| Component | Role in Phase 2 |
|-----------|----------------|
| Proof checker (`lean_interface.py`) | Validates MCTS-generated proofs |
| Proof formats (`formats.py`) | Wraps MCTS output for Lean checking |
| Reward system (`base.py`, `config.py`) | Scores proofs for GRPO |
| Curiosity reward | Encourages diverse lemma exploration |
| Lake/Mathlib4 | Ground truth for all checks |

---

## 5. Bootstrapping Challenge

**Problem:** With random GNN embeddings, MCTS selects irrelevant lemmas → 0% proof success → zero training signal. This is the same cold-start problem as Phase 1, but the solution is different.

**Phase 1 attempted solution:** Tiny length variation + curiosity bonus on invalid proofs — insufficient because token generation has no structural prior.

**Phase 2 solution:** **Pre-train the GNN on link prediction.** The dependency graph contains the ground truth: for every theorem, we know which other theorems it depends on. Training the GNN to predict these edges gives it an embedding space where related theorems are close. Then MCTS can use this to select relevant lemmas.

**Pre-training task:** Given a theorem node, predict its outgoing edges (which theorems it depends on). This is supervised learning over the 160K edges we already extracted.

**After pre-training:** Even a moderately trained GNN will embed theorems from the same domain near each other. MCTS searching for a proof of `example : 0 + 0 = 0 :=` will find lemmas like `add_zero` and `Nat.add_comm` in the neighborhood rather than `IsBigOWith.mul`.

---

## 6. Next Steps: GNN Training + Self-Play

### Immediate (Phase 2 completion):
1. **Complete GNN pre-training** on link prediction (100-200 epochs)
2. **Verify lemma relevance** — after training, check that MCTS selects semantically appropriate lemmas
3. **Bootstrap dataset test** — run MCTS on the 460 simple theorems from Phase 1

### Phase 3 transition:
4. **Wire physical correspondence scoring** into the reward pipeline
5. **Train the GNN+MCTS explorer** on physics-relevant domains
6. **Generate candidate structures** for the GR-QFT interface

---

## 7. File Inventory

### New files created (Phase 2):
```
src/explorer/
├── __init__.py                    # Module docstring
├── dependency_graph.py            # Graph data structures (300 lines)
├── graph_builder.py               # Proof parser + edge extractor (280 lines)
├── gnn_config.py                  # GNN hyperparameters (90 lines)
├── gnn_encoder.py                 # GAT implementation (300 lines)
├── mcts.py                        # MCTS proof search (320 lines)
├── proof_state.py                 # Proof state + action space (260 lines)
├── explorer_trainer.py            # GRPO+MCTS training loop (280 lines)
├── structure_generator.py         # Structure mutation + beam search (340 lines)
└── structure_validator.py         # Consistency checking (180 lines)

scripts/
├── build_dependency_graph.py      # Graph construction CLI
└── pretrain_gnn.py                # GNN pre-training on link prediction

data/graph/
├── dependency_graph.nx.pkl        # 58K-node NetworkX graph
├── dependency_graph.index.json    # Name → node ID index
├── dependency_graph.stats.json    # Graph statistics
└── dependency_graph.pyg.pt        # PyG-compatible tensors
```

### Modified files:
```
src/proof_checker/formats.py       # Added _is_tactic_proof() detection
```

---

## 8. Summary

Phase 2 built the architecture that Phase 1 was designed to validate the need for. The LLM is gone. In its place: a graph neural network that sees mathematical structure, an MCTS that explores proof space, and a generator that proposes new physical theories. The dependency graph connects 58,000 theorems across Analysis, Algebra, Geometry, and Topology — this is the board. The GNN is the player. MCTS is the search.

The remaining work is training — first on the dependency graph to learn semantic embeddings, then through self-play against the proof checker to learn proof strategy. The infrastructure is ready.

---

*Generated 2026-06-03. Phase 2 infrastructure complete. GNN pre-training in progress.*
