#!/usr/bin/env python3
"""Quick test: load v3 graph and verify training script entry."""
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from src.explorer.dependency_graph import DependencyGraph
print("Loading graph...", flush=True)
graph = DependencyGraph.load("data/graph/dependency_graph_full_v3")
print(f"Loaded: {graph.num_nodes} nodes, {graph.num_edges} edges", flush=True)
print("OK", flush=True)
