"""Hard-negative miner for contrastive lemma embeddings.

For each (goal, positive_lemma) proof-step pair, generates *confirmed* hard
negatives by running the Lean proof checker on candidate wrong lemmas and
collecting those that Lean rejects.

Hard negatives are confirmed by proof-checker pass/fail as ground truth,
not by any era label or external annotation. Zero era labels.

Architecture:
  1. Load all proof-step pairs + lemma candidates
  2. For each goal, sample N candidate wrong lemmas (same domain, high
     embedding similarity to the goal)
  3. Construct `example : <goal> := <candidate_lemma>` and check with Lean
  4. If Lean rejects → confirmed hard negative
  5. Output: (goal, positive_lemma, list[hard_negative_lemma]) triples

Usage as module:
    from src.contrastive.hard_negative_miner import HardNegativeMiner

    miner = HardNegativeMiner(project_dir="proof_checker_env")
    triples = miner.mine(pairs, num_hard_per_positive=5, max_pairs=50000)
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from collections import defaultdict
from typing import Optional

from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_lean_code, ProofResult


# ---------------------------------------------------------------------------
# Cache for proof checker results (goal, lemma) → pass/fail
# ---------------------------------------------------------------------------


class HardNegativeCache:
    """Persistent cache for proof checker results on lemma-goal pairs.

    Keyed by SHA-256 of (goal, lemma) to avoid re-running the same check.
    """

    def __init__(self, cache_path: Path | str | None = None):
        self._cache: dict[str, bool] = {}
        self._cache_path = Path(cache_path) if cache_path else None

        if self._cache_path and self._cache_path.exists():
            with open(self._cache_path) as f:
                for line in f:
                    entry = json.loads(line)
                    self._cache[entry["key"]] = entry["pass"]

    def _key(self, goal: str, lemma: str) -> str:
        return hashlib.sha256(
            f"{goal}|||{lemma}".encode("utf-8")
        ).hexdigest()[:16]

    def get(self, goal: str, lemma: str) -> Optional[bool]:
        """Return True (pass), False (fail), or None (not checked)."""
        return self._cache.get(self._key(goal, lemma))

    def set(self, goal: str, lemma: str, passed: bool) -> None:
        """Store a check result."""
        key = self._key(goal, lemma)
        self._cache[key] = passed

    def save(self) -> None:
        """Persist cache to disk."""
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w") as f:
            for key, passed in self._cache.items():
                json.dump({"key": key, "pass": passed}, f)
                f.write("\n")

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Lemma-goal proof script construction
# ---------------------------------------------------------------------------


def build_lemma_goal_proof_script(
    goal_statement: str,
    lemma_name: str,
) -> str:
    """Construct a Lean proof script: `example : <goal> := <lemma>`.

    The goal is wrapped as an `example` (to avoid name conflicts), and the
    lemma is used directly as a term proof. If the lemma can close the goal,
    Lean accepts; otherwise it produces a type error → confirmed hard negative.

    This avoids building full tactic proofs and works for any lemma that
    directly proves the goal (or fails).
    """
    # Clean up the goal: strip existing proof (:= ...), remove lemma/theorem prefix
    import re

    statement = goal_statement.strip()

    # Strip existing proof body
    statement = _strip_existing_proof(statement)

    # Convert "lemma name ..." / "theorem name ..." → "example ..."
    statement = re.sub(
        r'^(lemma|theorem)\s+\S+\s+', 'example ',
        statement, count=1,
    )

    # Ensure we end with := for the proof
    if not statement.rstrip().endswith(":="):
        if ":" in statement:
            statement = statement.rstrip().rstrip(":").rstrip() + " :="
        else:
            statement = statement.rstrip() + " :="

    return f"{statement} {lemma_name}"


def _strip_existing_proof(statement: str) -> str:
    """Remove any existing proof (:= term or := by ...) from a statement."""
    depth = 0
    i = 0
    while i < len(statement) - 1:
        ch = statement[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ":" and statement[i + 1] == "=" and depth == 0:
            return statement[:i].rstrip()
        i += 1
    return statement


# ---------------------------------------------------------------------------
# Hard-negative miner
# ---------------------------------------------------------------------------


class HardNegativeMiner:
    """Mine confirmed hard negatives by checking lemma-goal pairs.

    Uses the Lean proof checker to verify: does lemma X prove goal Y?
    If Lean rejects → X is a confirmed hard negative for Y.
    """

    def __init__(
        self,
        project_dir: str | Path | None = None,
        max_workers: int = 6,
        timeout: float = 10.0,
        cache_path: str | Path = "data/hard_negative_cache.jsonl",
    ):
        self.project_dir = project_dir
        self.max_workers = max_workers
        self.timeout = timeout
        self.cache = HardNegativeCache(cache_path)
        self._batch_checker: Optional[BatchChecker] = None

    def _get_checker(self) -> BatchChecker:
        if self._batch_checker is None:
            self._batch_checker = BatchChecker(
                project_dir=self.project_dir,
                timeout=self.timeout,
                max_workers=self.max_workers,
            )
        return self._batch_checker

    def mine(
        self,
        pairs: list[dict],
        num_hard_per_positive: int = 5,
        max_pairs: int = 50000,
        seed: int = 42,
    ) -> list[dict]:
        """Mine hard negatives for a subset of proof-step pairs.

        For each pair, sample candidate wrong lemmas from other pairs
        (same domain preferred), check via Lean, and collect confirmed
        negatives (those Lean rejects).

        Args:
            pairs: List of {"goal", "lemma", "domain", ...} dicts.
            num_hard_per_positive: Target number of hard negatives per pair.
            max_pairs: Maximum number of pairs to process.
            seed: Random seed for reproducibility.

        Returns:
            List of {"goal", "positive_lemma", "hard_negatives": [...]} dicts.
        """
        random.seed(seed)
        rng = random.Random(seed)

        # Sample pairs to process
        if len(pairs) > max_pairs:
            indices = rng.sample(range(len(pairs)), max_pairs)
            sample_pairs = [pairs[i] for i in indices]
        else:
            sample_pairs = pairs

        # Build domain → lemma index for candidate sampling
        domain_to_lemmas: dict[str, list[str]] = defaultdict(list)
        all_lemmas: list[str] = []
        for p in pairs:
            domain = p.get("domain", "unknown")
            domain_to_lemmas[domain].append(p["lemma"])
            all_lemmas.append(p["lemma"])

        # Unique lemmas per domain for efficiency
        unique_lemmas = list(set(all_lemmas))
        domain_to_unique: dict[str, list[str]] = {}
        for domain, lemmas in domain_to_lemmas.items():
            domain_to_unique[domain] = list(set(lemmas))

        print(f"Mining hard negatives from {len(sample_pairs)} pairs")
        print(f"  Unique lemmas: {len(unique_lemmas)}")
        print(f"  Domains: {len(domain_to_unique)}")
        print(f"  Target: {num_hard_per_positive} hard negatives per pair")
        print(f"  Workers: {self.max_workers}, Timeout: {self.timeout}s")

        # Collect all candidate checks
        checks: list[dict] = []  # (pair_idx, pair, candidate_lemma)
        for pair_idx, pair in enumerate(sample_pairs):
            goal = pair["goal"]
            positive = pair["lemma"]
            domain = pair.get("domain", "unknown")

            # Get candidates from same domain (prefer) or all domains
            domain_candidates = domain_to_unique.get(domain, unique_lemmas)
            candidates = [
                c for c in domain_candidates
                if c != positive  # don't use positive as negative
            ]

            if len(candidates) < num_hard_per_positive:
                # Supplement with lemmas from other domains
                extra = [
                    c for c in unique_lemmas
                    if c != positive and c not in candidates
                ]
                candidates = candidates + extra

            # Sample candidates
            n_sample = min(num_hard_per_positive * 3, len(candidates))
            sampled = rng.sample(candidates, n_sample)

            for candidate in sampled:
                checks.append({
                    "pair_idx": pair_idx,
                    "pair": pair,
                    "candidate_lemma": candidate,
                })

        # Remove checks already cached
        uncached_checks = []
        for check in checks:
            cached = self.cache.get(check["pair"]["goal"], check["candidate_lemma"])
            if cached is None:
                uncached_checks.append(check)
            else:
                # Already cached - will be processed below
                pass

        print(f"  Total checks: {len(checks)}, Cached: {len(checks) - len(uncached_checks)}, "
              f"To run: {len(uncached_checks)}")

        # Run uncached checks in batches
        batch_size = 100  # submit 100 at a time to batch checker
        for batch_start in range(0, len(uncached_checks), batch_size):
            batch = uncached_checks[batch_start:batch_start + batch_size]

            # Build Lean proof scripts
            codes = []
            for check in batch:
                script = build_lemma_goal_proof_script(
                    check["pair"]["goal"],
                    check["candidate_lemma"],
                )
                codes.append(script)

            # Run batch check
            checker = self._get_checker()
            results = checker.check_batch(codes)

            # Store results
            for check, result in zip(batch, results):
                self.cache.set(
                    check["pair"]["goal"],
                    check["candidate_lemma"],
                    result.success,
                )

            # Progress
            done = min(batch_start + batch_size, len(uncached_checks))
            passed = sum(1 for r in results if r.success)
            print(f"  Checked {done}/{len(uncached_checks)} "
                  f"({passed} passed, {len(results) - passed} failed)", flush=True)

        # Save cache
        self.cache.save()

        # Build output: collect hard negatives per pair
        pair_to_hard_negs: dict[int, list[str]] = defaultdict(list)
        for check in checks:
            passed = self.cache.get(check["pair"]["goal"], check["candidate_lemma"])
            if passed is False:  # Confirmed hard negative
                pair_to_hard_negs[check["pair_idx"]].append(check["candidate_lemma"])

        # Assemble triples
        triples = []
        for pair_idx, pair in enumerate(sample_pairs):
            hard_negs = pair_to_hard_negs.get(pair_idx, [])
            if not hard_negs:
                continue  # No confirmed hard negatives found for this pair

            # Trim to target
            if len(hard_negs) > num_hard_per_positive:
                hard_negs = rng.sample(hard_negs, num_hard_per_positive)

            triples.append({
                "goal": pair["goal"],
                "positive_lemma": pair["lemma"],
                "hard_negatives": hard_negs,
                "domain": pair.get("domain", "unknown"),
            })

        print(f"\nMining complete: {len(triples)} triples with hard negatives")
        print(f"  Average hard negatives per triple: "
              f"{sum(len(t['hard_negatives']) for t in triples) / max(1, len(triples)):.1f}")

        return triples


# ---------------------------------------------------------------------------
# Hard negative data loading
# ---------------------------------------------------------------------------


def load_hard_negative_data(
    data_path: Path | str,
) -> list[dict]:
    """Load hard-negative triples from JSONL file.

    Expected format per line:
        {"goal": "...", "positive_lemma": "...", "hard_negatives": ["...", ...], "domain": "..."}
    """
    triples = []
    with open(data_path) as f:
        for line in f:
            triples.append(json.loads(line))
    return triples


def save_hard_negative_data(
    triples: list[dict],
    data_path: Path | str,
) -> None:
    """Save hard-negative triples to JSONL file."""
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w") as f:
        for triple in triples:
            json.dump(triple, f)
            f.write("\n")
