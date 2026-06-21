#!/usr/bin/env python3
"""Debug: search one theorem and trace the search."""
import sys, json, time
sys.path.insert(0, '/home/blueman1818/Projects/theta-core')

import torch
from src.retrieval.goal_only_encoder import (
    GoalOnlyEncoder, build_vocabulary, prepare_lemma_groups,
    retrieve_lemmas, _tokenize_batch,
)
from src.explorer.proof_state import ProofState, Tactic, TacticType
from src.proof_checker.lean_interface import LeanProofChecker

# Load encoder
encoder = GoalOnlyEncoder.load('checkpoints/gnn/goal_only_encoder.pt')
encoder.eval()
print(f"Encoder: {encoder.count_params():,} params")

# Load index
pairs_path = 'data/raw/proof_step_pairs.jsonl'
goals, lemmas, _ = prepare_lemma_groups(pairs_path, max_pairs=None)
vocab = build_vocabulary(goals, max_vocab=3000)
print(f"Index: {len(goals)} pairs, {len(vocab)} tokens")

# Encode index
device = torch.device("cpu")
encoder = encoder.to(device)
index_embs_list = []
with torch.no_grad():
    for i in range(0, len(goals), 256):
        batch = goals[i : i + 256]
        batch_ids = _tokenize_batch(batch, vocab, 128).to(device)
        embs = encoder(batch_ids)
        index_embs_list.append(embs.cpu())
index_embeddings = torch.cat(index_embs_list, dim=0)
print(f"Index embeddings: {index_embeddings.shape}")

# Test theorem
theorem = "theorem alg_subst_expand (x y : ℝ) (h : x = y + 1) : x^2 - 2*x + 1 = y^2"
scored_lemmas = retrieve_lemmas(
    encoder, vocab, theorem, goals, lemmas, index_embeddings,
    k=50, top_n=20,
)
print(f"\nTop-5 lemmas for '{theorem[:50]}...':")
for name, score in scored_lemmas[:5]:
    print(f"  {name:40s} {score:.4f}")

# Manual search simulation
from heapq import heappush, heappop
from dataclasses import dataclass, field

@dataclass(order=True)
class PrioritizedState:
    priority: float
    depth: int
    tiebreaker: int
    state: ProofState = field(compare=False)
    steps: list = field(compare=False)

# Build root
root_state = ProofState.initial(theorem)
print(f"\nRoot hypotheses: {root_state.hypotheses}")
print(f"Root goals: {root_state.goals[:1]}")

tiebreaker = 0
root = PrioritizedState(priority=-1.0, depth=0, tiebreaker=0, state=root_state, steps=[])
heap = [root]
expansions = 0
complete_found = 0

while heap and expansions < 200:
    current = heappop(heap)
    state = current.state
    depth = current.depth
    
    if state.is_complete:
        complete_found += 1
        steps_str = [s.to_lean() for s in current.steps]
        print(f"\nComplete state #{complete_found} at depth {depth}, expansions={expansions}:")
        print(f"  Steps: {steps_str}")
        
        # Render and check with Lean
        from src.proof_checker.formats import wrap_theorem_with_proof
        proof_body = ProofState._render_proof(current.steps)
        code = wrap_theorem_with_proof(state.theorem_statement, proof_body)
        print(f"  Code:\n{code}")
        
        checker = LeanProofChecker(project_dir=None, timeout=15)
        result = checker.check(code)
        print(f"  Lean: success={result.success}, errors={result.errors[:2]}")
        
        if result.success:
            print("  ✓ VERIFIED!")
            break
        else:
            state.is_complete = False
            state.is_dead = True
            continue
    
    if state.is_dead or depth >= 10:
        continue
    
    expansions += 1
    
    # Generate candidates (updated version with structural-first + boosted scores)
    candidates = []
    
    # 1. Structural tactics FIRST with boosted scores
    for hyp_name in list(state.hypotheses.keys())[:5]:
        candidates.append((Tactic(TacticType.EXACT, hypothesis=hyp_name), 0.85))
    
    for hyp_name, hyp_type in state.hypotheses.items():
        if "=" in hyp_type or "↔" in hyp_type:
            candidates.append((Tactic(TacticType.REWRITE, hypothesis=hyp_name), 0.92))
            break
    
    if state.goals and ("→" in state.goals[0] or "∀" in state.goals[0]):
        candidates.append((Tactic(TacticType.INTRO, hypothesis="h"), 0.90))
    
    if state.goals:
        goal = state.goals[0]
        has_impl = "→" in goal or "∀" in goal
        if not has_impl and any(op in goal for op in ("*", "^", "+", "-", "=")):
            candidates.append((Tactic(TacticType.RING), 0.88))
        if ("/" in goal or "⁻¹" in goal) and not has_impl:
            candidates.append((Tactic(TacticType.FIELD_SIMP), 0.85))
        if any(op in goal for op in ("≤", "≥", "<", ">", "=")) and not has_impl:
            candidates.append((Tactic(TacticType.LINARITH), 0.85))
        candidates.append((Tactic(TacticType.SIMP), 0.80))
    
    # 2. Lemma applications from retrieval
    for lemma, score in scored_lemmas:
        candidates.append((Tactic(TacticType.APPLY, lemma=lemma), score))
        candidates.append((Tactic(TacticType.EXACT, lemma=lemma), score * 0.9))
    
    # Limit candidates
    max_actions = 40 * 2 + 12
    if len(candidates) > max_actions:
        candidates = candidates[:max_actions]
    
    if depth == 0:
        print(f"\nRoot candidates ({len(candidates)}):")
        for t, s in candidates[:10]:
            print(f"  {t.to_lean():30s} score={s:.3f}")
    
    for tactic, score in candidates:
        child_state = state.apply_tactic(tactic)
        if child_state is None or child_state.is_dead:
            continue
        
        priority = -(score / (1 + (depth + 1) * 0.05))
        tiebreaker += 1
        heappush(heap, PrioritizedState(
            priority=priority, depth=depth+1,
            tiebreaker=tiebreaker,
            state=child_state,
            steps=current.steps + [tactic],
        ))

print(f"\nSearch finished: {expansions} expansions, {complete_found} complete states")
