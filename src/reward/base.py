"""Reward computation for GRPO training.

Phase 1 reward components:
- Binary: 1.0 for valid proofs, 0.0 for invalid
- Optional length bonus for shorter valid proofs
- Curiosity/exploration bonus (Phase 1.5): count-based bonus for novel proofs
  CRITICAL for preventing mode collapse during self-play.

Phase H3 (traversal bonus):
- Rewards proofs that use lemmas 3+ hops from any training lemma in the
  dependency graph. Encourages GNN to explore deeper graph regions.
"""

import hashlib
import re
import torch
from collections import Counter
from typing import TYPE_CHECKING

from src.proof_checker.formats import ProofResult
from src.reward.config import RewardConfig

if TYPE_CHECKING:
    from src.explorer.dependency_graph import DependencyGraph

# Module-level proof signature counter.
# Tracks how many times each proof pattern has been generated.
# Persists across compute_reward calls within a training run.
_proof_signature_counter: Counter = Counter()

# ── Lemma extraction regex ────────────────────────────────────────────────
# Matches lemma references in Lean proof text. Patterns:
#   rw [lemma1, lemma2]    → captures lemma1, lemma2
#   apply lemma_name       → captures lemma_name
#   exact lemma_name       → captures lemma_name
#   simpa [...] using X    → captures X
#   have h := lemma        → captures lemma
#   refine lemma_name      → captures lemma_name
#   calc ... := lemma      → captures lemma (in calc chains)
#   ring, simp, linarith   → captures tactic names as "lemmas"
_LEMMA_RE = re.compile(
    r'(?:rw|rewrite|simp|simpa|dsimp)\s*\[([^\]]+)\]'  # rw [lemma1, lemma2]
    r'|(?<!\w)(?:apply|exact|refine|have\s+\S+\s+:=)\s+(\w[\w.]*\w)'  # apply/exact/refine/have
    r'|(?<!\w)(?:simpa|simp|ring|linarith|nlinarith|field_simp|omega|positivity|norm_num)(?![\w.])'  # tactic names
    r'|(?:calc\b.*?:\s*=\s*(?:by\s+)?)(\w[\w.]*\w)'  # calc := lemma
    r'|(?<!\w)using\s+(\w[\w.]*\w)'  # using lemma
    r'|(?<!\w)(?:rw|rewrite)\s+\[?([^\],;\s]+)\]?'  # rw lemma (without brackets)
)

# Common tactic names that are NOT real lemmas (to exclude from extraction)
_TACTIC_BLACKLIST = {
    'ring', 'simp', 'linarith', 'nlinarith', 'field_simp', 'omega',
    'positivity', 'norm_num', 'rfl', 'sorry', 'intro', 'apply',
    'exact', 'refine', 'have', 'cases', 'calc', 'rw', 'rewrite',
    'dsimp', 'simpa', 'by', 'fun', 'h', 'h1', 'h2', 'h3', 'h_',
    'ih', 'hsum', 'hcalc', 'hnQ', 'hn', 'hP', 'hQ', 'hR', 'hPQ',
    'hQR', 'hOr', 'hx', 'hy', 'hz', 'ha', 'hb', 'hc', 'hd',
    'hab', 'hbc', 'hac', 'hbd', 'hxy', 'hp', 'h0', 'h1', 'h2',
}


def _extract_lemmas_from_proof(proof_text: str) -> list[str]:
    """Extract lemma names referenced in a Lean proof text.

    Uses regex patterns to capture lemma references from common
    tactic applications. Filters out tactic names and local hypotheses.

    Returns:
        List of cleaned lemma name strings (may contain duplicates).
    """
    if not proof_text:
        return []

    lemmas = []

    # Pattern 1: rw/rewrite/simp/simpa/dsimp [lemmas]
    for m in re.finditer(r'(?:rw|rewrite|simp|simpa|dsimp)\s*\[([^\]]+)\]', proof_text):
        content = m.group(1)
        for token in re.split(r'[,;\s]+', content):
            token = token.strip().lstrip('←').lstrip('←')
            if token and token not in _TACTIC_BLACKLIST and not token.startswith('h'):
                lemmas.append(token)

    # Pattern 2: apply/exact/refine lemma_name
    for m in re.finditer(r'(?<!\w)(apply|exact|refine)\s+(\w[\w.]*\w)', proof_text):
        name = m.group(2)
        if name not in _TACTIC_BLACKLIST and not name.startswith('h'):
            lemmas.append(name)

    # Pattern 3: have h := lemma_name
    for m in re.finditer(r'have\s+\S+\s+:=\s+(\w[\w.]*\w)', proof_text):
        name = m.group(1)
        if name not in _TACTIC_BLACKLIST:
            lemmas.append(name)

    # Pattern 4: simpa [...] using lemma_name
    for m in re.finditer(r'using\s+(\w[\w.]*\w)', proof_text):
        name = m.group(1)
        if name not in _TACTIC_BLACKLIST and not name.startswith('h'):
            lemmas.append(name)

    # Deduplicate
    seen = set()
    result = []
    for l in lemmas:
        if l not in seen:
            seen.add(l)
            result.append(l)

    return result


def compute_traversal_bonus(
    proof_text: str,
    graph: "DependencyGraph | None" = None,
    training_lemma_set: set[str] | None = None,
    config: RewardConfig | None = None,
) -> float:
    """Compute graph-traversal bonus for a proof (H3 study).

    A proof gets a bonus if it uses lemmas that are 3+ hops away
    from any training lemma in the dependency graph. This encourages
    the GNN to explore lemmas far from the training distribution.

    For lemmas in different weakly connected components (no path),
    the distance is treated as infinite, which triggers the bonus.

    Args:
        proof_text: The generated proof text.
        graph: The dependency graph (required if traversal enabled).
        training_lemma_set: Set of lemma node IDs used in training proofs.
        config: Reward configuration.

    Returns:
        Scalar bonus value (0.0 if traversal disabled or no far lemmas found).
    """
    if config is None:
        config = RewardConfig()

    if not config.traversal_bonus_enabled:
        return 0.0

    if graph is None or training_lemma_set is None:
        return 0.0

    if not proof_text:
        return 0.0

    # Extract lemma names from the generated proof
    lemma_names = _extract_lemmas_from_proof(proof_text)
    if not lemma_names:
        return 0.0

    # Filter to lemmas that exist in the graph
    graph_lemmas = [l for l in lemma_names if l in graph.graph]
    if not graph_lemmas:
        return 0.0

    # Count how many are "far" (3+ hops from any training lemma)
    far_count = 0
    import networkx as nx

    for lemma in graph_lemmas:
        is_far = True
        for train_lemma in training_lemma_set:
            if train_lemma not in graph.graph:
                continue
            try:
                dist = nx.shortest_path_length(
                    graph.graph, lemma, train_lemma
                )
                if dist < config.traversal_hop_threshold:
                    is_far = False
                    break
            except nx.NetworkXNoPath:
                # No path → different component → effectively far
                pass
        if is_far:
            far_count += 1

    if far_count == 0:
        return 0.0

    # Bonus: weight * (fraction of lemmas that are far)
    fraction_far = far_count / len(graph_lemmas)
    return config.traversal_bonus_weight * fraction_far


def build_training_lemma_set(
    training_theorems: list[dict],
    graph: "DependencyGraph",
) -> set[str]:
    """Build the set of lemma node IDs referenced in training theorem proofs.

    Parses the ground-truth proof text of each training theorem to extract
    lemma names, then resolves them to graph node IDs.

    Args:
        training_theorems: List of theorem dicts with 'proof' key.
        graph: The dependency graph for name resolution.

    Returns:
        Set of graph node IDs that appear in training proofs.
    """
    lemma_set: set[str] = set()

    for theorem in training_theorems:
        proof = theorem.get("proof", "")
        if not proof:
            continue

        names = _extract_lemmas_from_proof(proof)
        for name in names:
            # Try direct lookup in graph
            if name in graph.graph:
                lemma_set.add(name)
            else:
                # Try resolving through the name index
                resolved = graph.resolve_name(name)
                if resolved and resolved in graph.graph:
                    lemma_set.add(resolved)

    return lemma_set


def _normalize_proof_for_signature(proof_text: str, max_chars: int = 200) -> str:
    """Normalize a proof for signature computation.

    Strips whitespace variations, comments, and trivial formatting
    differences so that semantically identical proofs share a signature.
    """
    # Collapse whitespace
    normalized = re.sub(r'\s+', ' ', proof_text.strip())
    # Remove Lean comments
    normalized = re.sub(r'--[^\n]*', '', normalized)
    normalized = re.sub(r'/-.*?-/', '', normalized, flags=re.DOTALL)
    # Truncate to signature length (catch the proof "shape")
    return normalized[:max_chars]


def _compute_proof_signature(proof_text: str, max_chars: int = 200) -> str:
    """Compute a hash-based signature for a proof.

    Uses SHA-256 truncated to 16 hex chars — fast, collision-resistant
    enough for proof deduplication.
    """
    normalized = _normalize_proof_for_signature(proof_text, max_chars)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def compute_curiosity_bonus(
    proof_text: str,
    config: RewardConfig | None = None,
) -> float:
    """Compute count-based exploration bonus for a proof.

    bonus = curiosity_weight / sqrt(count(signature) + 1)

    Novel proofs get the full bonus. Frequently generated proofs
    get diminishing returns, discouraging mode collapse.
    """
    if config is None:
        config = RewardConfig()

    if not config.curiosity_enabled:
        return 0.0

    sig = _compute_proof_signature(
        proof_text, config.curiosity_signature_length
    )
    count = _proof_signature_counter[sig]

    bonus = config.curiosity_weight / (count + 1) ** 0.5
    return bonus


def record_proof_signature(
    proof_text: str,
    config: RewardConfig | None = None,
) -> None:
    """Record a generated proof's signature in the global counter.

    Call this AFTER computing the reward so the curiosity bonus
    is based on the count BEFORE this proof was added.
    """
    if config is None:
        config = RewardConfig()

    if not config.curiosity_enabled:
        return

    sig = _compute_proof_signature(
        proof_text, config.curiosity_signature_length
    )
    _proof_signature_counter[sig] += 1

    # Prune old signatures if over the max
    if len(_proof_signature_counter) > config.curiosity_max_tracked:
        # Keep the most common half
        most_common = _proof_signature_counter.most_common(
            config.curiosity_max_tracked // 2
        )
        _proof_signature_counter.clear()
        _proof_signature_counter.update(dict(most_common))


def get_curiosity_stats() -> dict:
    """Return statistics about the curiosity tracker for logging."""
    if not _proof_signature_counter:
        return {"unique_signatures": 0, "total_counts": 0, "max_count": 0}

    return {
        "unique_signatures": len(_proof_signature_counter),
        "total_counts": sum(_proof_signature_counter.values()),
        "max_count": max(_proof_signature_counter.values()),
    }


def compute_reward(
    proof_result: ProofResult,
    config: RewardConfig | None = None,
    proof_text: str | None = None,
    graph: "DependencyGraph | None" = None,
    training_lemma_set: set[str] | None = None,
) -> float:
    """Compute reward for a single proof.

    Args:
        proof_result: Output from LeanProofChecker.check().
        config: Reward hyperparameters.
        proof_text: The generated proof text (required for curiosity bonus).
        graph: Dependency graph (required for traversal bonus, H3).
        training_lemma_set: Training lemma node IDs (required for traversal bonus, H3).

    Returns:
        Scalar reward value.
    """
    if config is None:
        config = RewardConfig()

    # Anti-reward-hacking: reject trivially short proofs
    if proof_result.success and proof_result.num_tokens < config.min_proof_tokens:
        return config.invalid_proof

    if not proof_result.success:
        # Invalid proofs get curiosity bonus + tiny length variation to
        # break reward symmetry during cold start. Without this, all
        # invalid proofs have identical reward → zero advantage → no gradient.
        reward = config.invalid_proof
        if config.curiosity_enabled and proof_text:
            reward += compute_curiosity_bonus(proof_text, config)
        # Tiny length-based variation (shorter proofs marginally better,
        # encouraging concise Lean-like output over rambling English)
        if proof_text:
            norm_len = min(len(proof_text), 500) / 500.0
            reward += 0.001 * (1.0 - norm_len)
        # Traversal bonus applies even to invalid proofs (encourages
        # exploration even when valid proof not found)
        if config.traversal_bonus_enabled and proof_text and graph and training_lemma_set:
            reward += compute_traversal_bonus(
                proof_text, graph, training_lemma_set, config
            )
        return reward

    base = config.valid_proof

    # Length bonus: shorter valid proofs score higher
    if config.length_bonus_enabled:
        n_tokens = proof_result.num_tokens
        excess = max(0, n_tokens - config.length_reference_tokens)
        bonus = max(0.0, 1.0 - excess * config.length_decay_rate)
        base += config.length_bonus_weight * bonus

    # Curiosity bonus: novel proofs get higher reward (Phase 1.5)
    if config.curiosity_enabled and proof_text:
        base += compute_curiosity_bonus(proof_text, config)

    # H3 Traversal bonus: proofs using far lemmas get higher reward
    if config.traversal_bonus_enabled and proof_text and graph and training_lemma_set:
        base += compute_traversal_bonus(
            proof_text, graph, training_lemma_set, config
        )

    return base


def compute_rewards_batch(
    results: list[ProofResult],
    config: RewardConfig | None = None,
    proof_texts: list[str] | None = None,
    graph: "DependencyGraph | None" = None,
    training_lemma_set: set[str] | None = None,
) -> torch.Tensor:
    """Compute rewards for a batch of proof results.

    Args:
        results: List of ProofResult from batch checking.
        config: Reward hyperparameters.
        proof_texts: Optional list of generated proof strings for curiosity bonus.
        graph: Dependency graph (for traversal bonus, H3).
        training_lemma_set: Training lemma node IDs (for traversal bonus, H3).

    Returns:
        Tensor of reward values, shape (len(results),).
    """
    texts = proof_texts or [None] * len(results)
    rewards = [
        compute_reward(r, config, proof_text=t,
                       graph=graph, training_lemma_set=training_lemma_set)
        for r, t in zip(results, texts)
    ]
    return torch.tensor(rewards, dtype=torch.float32)


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute group-relative advantages for GRPO.

    For each group of K proofs for one theorem:
      advantage_i = (r_i - mean(r_group)) / (std(r_group) + eps)

    Args:
        rewards: Flat tensor of rewards, shape (num_prompts * group_size,).
        group_size: K, number of proofs per theorem.
        eps: Small constant for numerical stability.

    Returns:
        Advantage tensor of same shape as rewards.
    """
    num_groups = rewards.numel() // group_size
    reshaped = rewards.view(num_groups, group_size)

    group_mean = reshaped.mean(dim=1, keepdim=True)
    group_std = reshaped.std(dim=1, keepdim=True)

    advantages = (reshaped - group_mean) / (group_std + eps)

    return advantages.view(-1)
