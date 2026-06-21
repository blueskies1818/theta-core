#!/usr/bin/env python3
"""Inference: score gate3_v2 theorems against all lemma candidates using binary scorer."""
import sys, json, time, torch
import torch.nn.functional as F
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.scoring.binary_scorer import FrozenGNNEncoder, BinaryScorer

print("=" * 60, flush=True)
print("BINARY SCORER GATE3 INFERENCE", flush=True)
print("=" * 60, flush=True)

# Load encoder
print("Loading encoder...", flush=True)
enc = FrozenGNNEncoder(
    'checkpoints/gnn/gate2_fullgraph_finetuned.pt',
    'data/graph/dependency_graph_full'
)

# Load scorer
scorer_path = _PROJECT_ROOT / 'checkpoints/scorer/binary_scorer.pt'
print(f"Loading scorer from {scorer_path}...", flush=True)
state = torch.load(str(scorer_path), map_location='cpu', weights_only=False)
scorer = BinaryScorer(hidden_dim=enc.hidden_dim)
scorer.load_state_dict(state['model_state_dict'])
scorer.eval()
print(f"Scorer loaded (epoch {state.get('epoch', '?')})", flush=True)

# Load gate3_v2
gate3_path = _PROJECT_ROOT / 'data/raw/gate3_v2.jsonl'
with open(gate3_path) as f:
    theorems = [json.loads(line) for line in f]
print(f"Loaded {len(theorems)} gate3_v2 theorems", flush=True)

# Get lemma candidates from graph
lemma_candidates = sorted(enc.graph.node_ids)
print(f"Lemma candidates: {len(lemma_candidates)}", flush=True)

# Pre-compute lemma embeddings
print("Pre-computing lemma embeddings...", flush=True)
lemma_embs = []
valid_lemmas = []
for name in lemma_candidates:
    idx = enc.lemma_to_idx.get(name)
    if idx is not None and idx < enc.num_nodes:
        lemma_embs.append(enc.node_embeddings[idx])
        valid_lemmas.append(name)
lemma_embs_t = torch.stack(lemma_embs)  # [M, 256]
print(f"Valid lemmas: {len(valid_lemmas)}", flush=True)

# Score each theorem
top_k = 30
results = []
start = time.time()

for i, theorem in enumerate(theorems):
    statement = theorem.get('statement', '')
    name = theorem.get('name', f't{i}')
    
    # Encode goal
    goal_emb = enc.encode_goal(statement).unsqueeze(0)  # [1, 256]
    
    # Score against all candidates in batches
    batch_size = 16384
    all_scores = []
    with torch.no_grad():
        for bi in range(0, len(lemma_embs_t), batch_size):
            batch = lemma_embs_t[bi:bi + batch_size]
            goal_batch = goal_emb.expand(batch.size(0), -1)
            scores = torch.sigmoid(scorer(goal_batch, batch)).squeeze(-1)
            all_scores.append(scores)
    all_scores = torch.cat(all_scores)
    
    # Top-K
    topk = torch.topk(all_scores, min(top_k, all_scores.size(0)))
    top_indices = topk.indices.tolist()
    top_scores = topk.values.tolist()
    
    top_lemmas = [
        {'lemma': valid_lemmas[idx], 'score': round(score, 6)}
        for idx, score in zip(top_indices, top_scores)
    ]
    
    results.append({
        'name': name,
        'statement': statement,
        'ground_truth_proof': theorem.get('proof', ''),
        'domain': theorem.get('domain', ''),
        'top_lemmas': top_lemmas,
    })
    
    if (i + 1) % 10 == 0:
        elapsed = time.time() - start
        eta = elapsed / (i + 1) * (len(theorems) - i - 1)
        print(f"  [{i+1}/{len(theorems)}] {elapsed:.0f}s elapsed, ETA: {eta:.0f}s", flush=True)

elapsed = time.time() - start
print(f"\nInference complete: {len(theorems)} theorems in {elapsed:.1f}s", flush=True)

# Save
output_path = _PROJECT_ROOT / 'data/binary_scorer_gate3.json'
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump({
        'architecture': 'Binary Scorer (frozen GNN → bilinear → MLP)',
        'scorer_checkpoint': 'checkpoints/scorer/binary_scorer.pt',
        'gnn_checkpoint': 'checkpoints/gnn/gate2_fullgraph_finetuned.pt',
        'num_candidates': len(valid_lemmas),
        'top_k': top_k,
        'training_epochs': state.get('epoch', '?'),
        'training_loss': state.get('val_loss', '?'),
        'results': results,
    }, f, indent=2)

print(f"Results saved to {output_path}", flush=True)
