#!/usr/bin/env python3
"""Quick import/syntax check for H2 scoring module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.explorer.h2_scoring import (
    TwoTowerBilinear,
    CrossAttentionScorer,
    graph_filtered_retrieval,
    cosine_similarity_scoring,
)
print("All imports OK")

tt = TwoTowerBilinear()
ca = CrossAttentionScorer()
print(f"TwoTower params: {sum(p.numel() for p in tt.parameters()):,}")
print(f"CrossAttn params: {sum(p.numel() for p in ca.parameters()):,}")

# Quick smoke test: score random embeddings
import torch
goal = torch.randn(256)
cands = torch.randn(50, 256)

scores_cos = cosine_similarity_scoring(goal, cands)
print(f"Cosine scores: {scores_cos.shape}, range [{scores_cos.min():.3f}, {scores_cos.max():.3f}]")

scores_tt = tt(goal, cands)
print(f"TwoTower scores: {scores_tt.shape}, range [{scores_tt.min():.3f}, {scores_tt.max():.3f}]")

scores_ca = ca(goal, cands)
print(f"CrossAttn scores: {scores_ca.shape}, range [{scores_ca.min():.3f}, {scores_ca.max():.3f}]")

print("\nAll smoke tests passed!")
