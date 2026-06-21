"""
Quick eval: use link-prediction GNN embeddings for lemma retrieval on gate3_v2.
"""
import json, sys, time, torch, torch.nn.functional as F
from pathlib import Path

_project_root = Path("/home/blueman1818/Projects/theta-core")
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features

# Load enriched graph
print("Loading graph...", flush=True)
dg = DependencyGraph.load(_project_root / "data/graph/dependency_graph_full_v2")
print(f"  {dg.summary()}", flush=True)

# Load link-prediction pretrained model
print("Loading model...", flush=True)
ckpt = torch.load(_project_root / "checkpoints/gnn/gnn_best.pt", map_location="cpu", weights_only=False)
config = GNNConfig(**{k: v for k, v in ckpt.get("config_dict", {}).items() 
                       if k in ["hidden_dim", "num_layers", "num_heads", "input_dim", 
                                "dropout", "activation", "use_edge_types", "num_edge_types",
                                "bidirectional", "use_goal_encoder", "goal_encoder_expansion",
                                "goal_encoder_dropout"]})
gnn = GNNEncoder(config)
gnn.load_state_dict(ckpt["model_state_dict"])
gnn.eval()
total_p = sum(v.numel() for v in ckpt["model_state_dict"].values())
print(f"  {total_p:,} params", flush=True)

# Compute graph tensors
print("Computing graph tensors...", flush=True)
features = extract_initial_features(dg, config)
sources, targets, edge_types, num_nodes = prepare_graph_tensors(dg)
print(f"  {num_nodes} nodes, {sources.size(0)} edges", flush=True)

# Compute all lemma embeddings
print("Computing embeddings...", end=" ", flush=True)
t0 = time.time()
with torch.no_grad():
    lemma_embs = gnn(features, sources, targets, edge_types, num_nodes)
lemma_embs = F.normalize(lemma_embs, dim=-1)
print(f"done ({time.time()-t0:.1f}s). Shape: {list(lemma_embs.shape)}", flush=True)

# Load lemma index
with open(_project_root / "data/graph/dependency_graph_full_v2.lemma_index.json") as f:
    lemma_to_idx = json.load(f)

# Load gate3_v2
with open(_project_root / "data/raw/gate3_v2.jsonl") as f:
    theorems = [json.loads(line) for line in f if line.strip()]

print(f"\nGate3 v2: {len(theorems)} theorems", flush=True)

# For retrieval: since we don't have a goal encoder, we use keyword-matched lemmas
# as a proxy. For each theorem, find which lemma nodes share keywords with the goal.
from collections import defaultdict
from src.explorer.mcts import _extract_math_keywords

# Build keyword index over lemma names (fast)
print("Building keyword index...", end=" ", flush=True)
kw_to_lemmas = defaultdict(set)
for lemma_name, idx in lemma_to_idx.items():
    for kw in _extract_math_keywords(lemma_name):
        kw_to_lemmas[kw].add(idx)
print(f"{len(kw_to_lemmas)} keywords", flush=True)

# For each theorem, get candidate lemmas by keyword matching
# Then rank them by cosine similarity of their embeddings to the centroid
print("\nEvaluating...", flush=True)
total = 0
top1 = 0
top5 = 0
top10 = 0
mrr_sum = 0.0

for i, theorem in enumerate(theorems):
    goal = theorem.get("goal", theorem.get("statement", ""))
    ground_truth = theorem.get("lemma", theorem.get("ground_truth_lemma", ""))
    
    # Get ground truth index
    gt_idx = lemma_to_idx.get(ground_truth)
    if gt_idx is None:
        continue
    total += 1
    
    # Get candidate lemmas via keyword matching
    keywords = _extract_math_keywords(goal)
    candidates = set()
    for kw in keywords:
        candidates.update(kw_to_lemmas.get(kw, set()))
    
    if not candidates:
        continue
    
    cand_list = sorted(candidates)
    cand_tensor = torch.tensor(cand_list)
    
    # Compute centroid of candidate embeddings
    cand_embs = lemma_embs[cand_tensor]
    centroid = cand_embs.mean(dim=0, keepdim=True)
    
    # Rank by cosine similarity to centroid
    sims = F.cosine_similarity(centroid, cand_embs, dim=-1)
    _, sorted_idx = torch.sort(sims, descending=True)
    
    # Find rank of ground truth
    gt_pos = (cand_tensor == gt_idx).nonzero(as_tuple=True)
    if len(gt_pos[0]) == 0:
        continue
    
    gt_cand_idx = gt_pos[0][0].item()
    rank_positions = (sorted_idx == gt_cand_idx).nonzero(as_tuple=True)
    if len(rank_positions[0]) == 0:
        continue
    rank = rank_positions[0][0].item() + 1
    
    mrr_sum += 1.0 / rank
    if rank == 1:
        top1 += 1
    if rank <= 5:
        top5 += 1
    if rank <= 10:
        top10 += 1

n = max(1, total)
results = {
    "mrr": mrr_sum / n,
    "top1": top1 / n,
    "top5": top5 / n,
    "top10": top10 / n,
    "total": total,
    "num_with_candidates": sum(1 for t in theorems if 
        any(kw in kw_to_lemmas for kw in _extract_math_keywords(
            t.get("goal", t.get("statement", ""))))),
}

print(f"\nResults (keyword-centroid retrieval on gate3_v2):")
for k, v in results.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# Save
out_path = _project_root / "data/gnn_enriched_baseline_gate3.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
