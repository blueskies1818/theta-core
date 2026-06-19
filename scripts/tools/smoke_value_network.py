#!/usr/bin/env python3
"""Quick smoke test for value network module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explorer.value_network import ValueNetwork, ValueHead
print("ValueNetwork module: OK")

from src.explorer.gnn_best_first_search import GNNBestFirstConfig
cfg = GNNBestFirstConfig()
print(f"Config: value_weight={cfg.value_weight}, prune={cfg.value_prune_threshold}")

from src.explorer.gnn_encoder import GNNEncoder
import os

# Find available checkpoint
ckpt = None
for alt in [
    "checkpoints/gnn/gate2_fullgraph_finetuned.pt",
    "checkpoints/gnn/full_graph_pretrained.pt",
    "checkpoints/gnn/gnn_best.pt",
]:
    if os.path.exists(alt):
        ckpt = alt
        break

if ckpt:
    gnn = GNNEncoder.load(ckpt)
    print(f"GNN: {sum(p.numel() for p in gnn.parameters()):,} params from {ckpt}")
    vn = ValueNetwork(gnn, freeze_encoder=True)
    vn_params = sum(p.numel() for p in vn.value_head.parameters())
    gnn_trainable = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
    print(f"Value head: {vn_params:,} trainable params")
    print(f"GNN trainable (should be 0): {gnn_trainable}")
    print("ValueNetwork instantiation: OK")
else:
    print("NO GNN CHECKPOINT FOUND")
    sys.exit(1)
