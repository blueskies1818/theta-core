#!/usr/bin/env python3
"""Test import of train_gnn_triplet module."""
import sys
from pathlib import Path
_project_root = Path("/home/blueman1818/Projects/theta-core")
sys.path.insert(0, str(_project_root))
import scripts.training.train_gnn_triplet
print("Import OK")
# Check key functions
assert hasattr(scripts.training.train_gnn_triplet, 'compute_triplet_loss')
assert hasattr(scripts.training.train_gnn_triplet, 'compute_link_prediction_loss')
print("All functions accessible")
