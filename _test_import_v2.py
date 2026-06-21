#!/usr/bin/env python3
import sys, time
sys.path.insert(0, '/home/blueman1818/Projects/theta-core')
t0 = time.time()
print("Starting imports...", flush=True)
from scripts.training.train_multitask_v2_gnn import load_pairs, compute_import_loss, compute_proof_infonce_loss, build_goal_context_embedding
print(f"Imports OK ({time.time()-t0:.1f}s)", flush=True)
