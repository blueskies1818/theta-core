#!/usr/bin/env python3
"""Ultra-fast value network smoke test on 5 theorems."""
import json, sys, time, threading
from pathlib import Path
import torch

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import GNNEncoder, extract_initial_features, prepare_graph_tensors
from src.explorer.gnn_best_first_search import GNNBestFirstSearch, GNNBestFirstConfig
from src.explorer.value_network import ValueNetwork
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from scripts.eval.eval_gnn_prover import build_lemma_index, build_lemma_norm_index

torch.set_num_threads(4)

print("Loading GNN...")
gnn = GNNEncoder.load(str(root / "checkpoints/gnn/proof_step_pretrained.pt"))
gnn.eval()
print(f"  {sum(p.numel() for p in gnn.parameters()):,} params")

print("Loading value network...")
vn = ValueNetwork.load(str(root / "checkpoints/value_network.pt"), gnn)
vn.eval()

print("Loading graph...")
graph = DependencyGraph.load(root / "data/graph/dependency_graph")
print(f"  {graph.summary()}")

lemma_to_idx = build_lemma_index(graph)
idx_to_norm = build_lemma_norm_index(graph, lemma_to_idx)

print("Computing embeddings...")
features = extract_initial_features(graph, gnn.config)
sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
with torch.no_grad():
    node_embeddings = gnn(features, sources, targets, edge_types, num_nodes)
print(f"  {node_embeddings.shape}")

theorems = [json.loads(l) for l in open(root / "data/raw/gate3_v2.jsonl")][:5]
checker = BatchChecker(max_workers=2, timeout=60)

def run(name, cfg, vn_net):
    bf = GNNBestFirstSearch(gnn=gnn, graph=graph, node_embeddings=node_embeddings,
                            lemma_index=lemma_to_idx, idx_to_norm=idx_to_norm,
                            config=cfg, proof_checker=checker, value_network=vn_net)
    ok = 0
    for thm in theorems:
        steps, _ = bf.search(thm["statement"], domain=thm.get("domain"), verbose=False)
        if steps:
            code = wrap_theorem_with_proof(thm["statement"], ProofState._render_proof(steps))
            cr = checker.check_batch([code])
            if cr[0].success:
                ok += 1
                print(f"  {name}: {thm['name']} OK - {[s.to_lean() for s in steps[:3]]}")
            else:
                print(f"  {name}: {thm['name']} FAIL (Lean rejects: {cr[0].errors[0][:60]})")
        else:
            print(f"  {name}: {thm['name']} FAIL (no proof)")
    return ok

print("\n=== BLIND ===")
blind_ok = run("BLIND", GNNBestFirstConfig(max_expansions=100, value_weight=0.0, num_threads=4), None)

print("\n=== VALUE-GUIDED ===")
value_ok = run("VALUE", GNNBestFirstConfig(max_expansions=100, value_weight=0.3, value_prune_threshold=0.1, num_threads=4), vn)

print(f"\nRESULTS: blind={blind_ok}/5, value={value_ok}/5")

# Save smoke result
out = {
    "task": "value_net_smoke_test",
    "blind_ok": blind_ok, "value_ok": value_ok,
    "gnn": str(root / "checkpoints/gnn/proof_step_pretrained.pt"),
    "value_checkpoint": str(root / "checkpoints/value_network.pt"),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
json.dump(out, open(root / "data/value_net_smoke.json", "w"), indent=2)
print("Smoke result saved to data/value_net_smoke.json")
