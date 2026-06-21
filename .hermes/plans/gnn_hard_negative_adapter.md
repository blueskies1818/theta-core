# GNN + Hard-Negative Adapter — Implementation Plan

> **For Hermes:** Use kanban-worker skill. Single task, auto-complete.

**Goal:** Fine-tune GNN embeddings with proof-checker rejection feedback to
discriminate proof-closing lemmas from merely graph-adjacent ones.

**Architecture:** Full GNN fine-tuning (1.1M params, ALL trainable) with
TRIPLET-ONLY margin loss on proof-step hard negatives. NO InfoNCE.
Checkpoints at every epoch for safe revert.

**Why triplet-only, no InfoNCE:** InfoNCE pushes ALL lemma embeddings apart
simultaneously (in-batch soft negatives). 70K points in 256 dims can't all be
far from each other — the space scrambles and MRR→0. Triplet margin loss only
pushes SPECIFIC Lean-rejected lemmas away from goals they don't solve. The
rest of the embedding space stays intact. Temperature=1.0, lr=1e-4.

**Prior failures:**
- Frozen adapter: collapsed (rank 197/256, MRR 0.0001)
- Full GNN + InfoNCE: MRR 0.0001 through 8 epochs, no improvement

**Tech Stack:** PyTorch, existing GNN checkpoint, existing contrastive loss code,
existing hard-negative miner, existing 226K proof-step pairs.

**Safety Gates (in-training, abort if ANY trip):**
- Gate A: Link-prediction preservation loss ≤30% above baseline
- Gate B: Validation MRR ≥ 0.60 (was 0.0001 with InfoNCE; triplet should preserve retrieval)
- Gate C: Lemma embedding diversity (cosine std > 0.05, rank > 128)
- Gate D: Checkpoint saved every epoch — revert to any epoch if later epoch degrades
- Gate E: Post-training Gate 3 must beat 15.6% baseline

---

## Task 1: Create the GNN Adapter module

**Files:**
- Create: `src/explorer/gnn_adapter.py`

```python
"""Projection adapter that sits on top of a frozen GNN.

Freeze the full GNN. Train only this small head to re-weight
embedding dimensions for proof utility. Preserves graph topology
knowledge while learning which dimensions matter for proof-closing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNAdapterHead(nn.Module):
    """Two-layer projection head over frozen GNN embeddings.

    Input: 256-dim GNN node embedding
    Output: 256-dim proof-utility embedding (L2-normalized)

    ~130K params, designed to be the ONLY trainable component
    when paired with a frozen GNN backbone.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [*, D] GNN embeddings → [*, D] adapted embeddings."""
        adapted = self.proj(x)
        return F.normalize(adapted, dim=-1)
```

**Step 1: Create file** at `src/explorer/gnn_adapter.py`

**Step 2: Verify imports work**
```bash
cd /home/blueman1818/Projects/theta-core && python -c "from src.explorer.gnn_adapter import GNNAdapterHead; m = GNNAdapterHead(); print(f'params: {sum(p.numel() for p in m.parameters()):,}')"
```
Expected: `params: 132,352` (roughly 130K)

**Step 3: Commit**
```bash
git add src/explorer/gnn_adapter.py
git commit -m "feat: GNNAdapterHead for frozen-GNN proof-utility fine-tuning"
```

---

## Task 2: Build the training script

**Files:**
- Create: `scripts/training/train_gnn_adapter.py`

Builds on existing infra:
- `src/explorer/gnn_encoder.py` — GoalEncoder.load() returns frozen GNN
- `src/contrastive/hard_negative_loss.py` — compute_combined_loss()
- `src/contrastive/hard_negative_miner.py` — existing miner
- `src/explorer/gnn_adapter.py` — new adapter head

```python
"""Train a proof-utility adapter on top of frozen GNN embeddings.

Loads pre-trained GNN, freezes it, adds GNNAdapterHead, trains
with contrastive loss on proof-step pairs + hard negatives from
proof checker. Includes link-prediction anchor loss to prevent
catastrophic forgetting.

Safety gates:
  - GNN frozen (≤150K trainable params)
  - Link-prediction preservation loss monitored
  - Validation retrieval MRR tracked vs GNN baseline (0.786)
  - Embedding health (diversity, rank) checked per epoch
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.explorer.gnn_encoder import GoalEncoder
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_adapter import GNNAdapterHead
from src.contrastive.hard_negative_loss import (
    compute_combined_loss,
    compute_retrieval_accuracy,
)
from src.contrastive.hard_negative_miner import HardNegativeMiner


def load_gnn(checkpoint_path: str) -> GoalEncoder:
    """Load frozen GNN."""
    gnn = GoalEncoder.load(checkpoint_path)
    for p in gnn.parameters():
        p.requires_grad = False
    gnn.eval()
    return gnn


def train_adapter(
    gnn_checkpoint: str,
    proof_step_pairs_path: str,
    output_dir: str,
    num_epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hard_neg_weight: float = 0.5,
    preservation_weight: float = 0.1,
    margin: float = 0.3,
    num_threads: int = 4,
    val_split: float = 0.2,
    max_hard_negatives: int = 5,
):
    # --- Setup ---
    torch.set_num_threads(num_threads)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen GNN
    gnn = load_gnn(gnn_checkpoint)
    adapter = GNNAdapterHead()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)

    # Load pairs
    pairs = load_pairs(proof_step_pairs_path)
    split_idx = int(len(pairs) * (1 - val_split))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # Build graph tensors once (frozen GNN, so pre-computable)
    # ... (load graph, compute initial embeddings)

    # Baseline MRR
    baseline_mrr = compute_gate3_mrr(gnn)  # ~0.786

    # Training loop with safety gates
    stats_history = []
    for epoch in range(num_epochs):
        # ... training steps ...

        # Safety Gate B: link-prediction preservation
        # Safety Gate C: val MRR vs baseline
        # Safety Gate D: embedding health
        # (trip = abort and report)

    # Save adapter
    torch.save(adapter.state_dict(), output_dir / "adapter.pt")

    # Export stats
    with open(output_dir / "training_stats.json", "w") as f:
        json.dump(stats_history, f, indent=2)

    return stats_history
```

**Step 1: Write full training script** with data loading, training loop, all five safety gates inline. Abort on any gate trip with clear error message and partial stats saved.

**Step 2: Smoke test with 100 pairs**
```bash
cd /home/blueman1818/Projects/theta-core && python scripts/training/train_gnn_adapter.py \
  --gnn-checkpoint checkpoints/gnn/full_graph_pretrained.pt \
  --pairs data/raw/proof_step_pairs.jsonl \
  --output-dir data/adapter_smoke \
  --epochs 2 --max-pairs 100 --num-threads 4
```
Expected: Runs 2 epochs, saves adapter.pt and training_stats.json. Embedding rank stays 256, loss decreases.

**Step 3: Commit**
```bash
git add scripts/training/train_gnn_adapter.py
git commit -m "feat: GNN adapter training with safety-gated contrastive loss"
```

---

## Task 3: Build the evaluation script

**Files:**
- Create: `scripts/eval/eval_gnn_adapter.py`

```python
"""Evaluate GNN+Adapter on gate3_v2 benchmark.

Loads frozen GNN + trained adapter, runs best-first search
on all 64 gate3_v2 theorems. Compares to GNN-only baseline.
"""

import json
import time
from pathlib import Path

import torch

from src.explorer.gnn_encoder import GoalEncoder
from src.explorer.gnn_adapter import GNNAdapterHead
from src.explorer.gnn_best_first_search import GNNBestFirstSearch


def eval_adapter(
    gnn_checkpoint: str,
    adapter_checkpoint: str,
    graph_path: str = "data/graph/dependency_graph_full",
    theorems_path: str = "data/raw/gate3_v2.jsonl",
    output_path: str = "data/adapter_gate3_result.json",
    num_threads: int = 4,
):
    # Load frozen GNN
    gnn = GoalEncoder.load(gnn_checkpoint)
    for p in gnn.parameters():
        p.requires_grad = False
    gnn.eval()

    # Load adapter
    adapter = GNNAdapterHead()
    adapter.load_state_dict(torch.load(adapter_checkpoint))
    adapter.eval()

    # Build combined encoder
    def encode_goal(goal_text, graph):
        emb = gnn.encode_goal(goal_text, graph)
        return adapter(emb)

    # Run best-first search
    search = GNNBestFirstSearch(...)
    results = search.evaluate_all(theorems_path, encode_goal, ...)

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
```

**Step 1: Write eval script**

**Step 2: Smoke test on 5 theorems**
```bash
cd /home/blueman1818/Projects/theta-core && python scripts/eval/eval_gnn_adapter.py \
  --gnn-checkpoint checkpoints/gnn/full_graph_pretrained.pt \
  --adapter data/adapter_smoke/adapter.pt \
  --output data/adapter_smoke_eval.json \
  --max-theorems 5
```
Expected: Runs, produces JSON. Score should match GNN baseline since adapter is barely trained.

**Step 3: Commit**
```bash
git add scripts/eval/eval_gnn_adapter.py
git commit -m "feat: GNN+Adapter evaluation script for gate3_v2"
```

---

## Task 4: Full training run + evaluation

**Step 1: Run full training** (20 epochs, all 226K pairs, 4 threads)
```bash
cd /home/blueman1818/Projects/theta-core && python scripts/training/train_gnn_adapter.py \
  --gnn-checkpoint checkpoints/gnn/full_graph_pretrained.pt \
  --pairs data/raw/proof_step_pairs.jsonl \
  --output-dir data/adapter_full \
  --epochs 20 --batch-size 256 --num-threads 4
```
Expected: Training completes with safety gates passing. If any gate trips, abort and report.

**Step 2: Run full evaluation** (64 theorems, gate3_v2)
```bash
cd /home/blueman1818/Projects/theta-core && python scripts/eval/eval_gnn_adapter.py \
  --gnn-checkpoint checkpoints/gnn/full_graph_pretrained.pt \
  --adapter data/adapter_full/adapter.pt \
  --output data/adapter_gate3_result.json
```
Target: gate3 rate > 15.6% (baseline). Any improvement is a win.

**Step 3: Compare to baseline**
Print side-by-side comparison: GNN-only vs GNN+Adapter per domain, per proof type.

**Step 4: Archive artifacts**
```bash
cp data/adapter_full/adapter.pt checkpoints/gnn/adapter_v1.pt
cp data/adapter_gate3_result.json data/adapter_v1_result.json
```

**Step 5: Commit and push**
```bash
git add checkpoints/gnn/adapter_v1.pt data/adapter_v1_result.json
git commit -m "feat: GNN+Adapter v1 — proof-utility fine-tuned with hard negatives
$(cat data/adapter_v1_result.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f\"gate3: {d[\"rate\"]} ({d[\"proved\"]}/{d[\"total\"]})\")' 2>/dev/null)"
git push origin main
```

---

## Task 5: Honesty Gate re-run

Re-run all 5 gates with the adapted model:

```bash
# Gate 1: Purity
python scripts/gates/audit_purity.py --checkpoint checkpoints/gnn/adapter_v1.pt
# Must PASS

# Gate 2: Structural Independence
python scripts/gates/audit_structural.py --checkpoint checkpoints/gnn/adapter_v1.pt
# Must PASS

# Gate 3: Lemma Novelty (already run)
python scripts/eval/eval_gnn_adapter.py --output data/adapter_gate3_result.json
# Must beat 15.6%

# Gate 5: Statistical Validation (3 replicates)
python scripts/eval/eval_gnn_adapter.py --repeat 3 --output data/adapter_gate5_result.json
# Std < 3pp
```

Gate 4 (era discrimination) expected FAIL (pre-existing).

---

## Task 6: Final report

Write results to `docs/reviews/gnn_adapter_results.md`:
- Training stats (loss curves, safety gate status)
- Gate 3 result vs baseline
- Honesty gate status
- Next steps recommendation
