"""Parallel batch proof checking using ProcessPoolExecutor.

The proof checker runs on CPU and is the throughput bottleneck.
This module provides parallel checking across all available CPU cores.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from src.proof_checker.lean_interface import LeanProofChecker
from src.proof_checker.formats import ProofResult


def _check_single_worker(
    code: str, project_dir: Optional[str], timeout: float
) -> ProofResult:
    """Check a single proof. Called in a subprocess via ProcessPoolExecutor.

    Must be a module-level function (not a method) for pickling.
    Creates a fresh LeanProofChecker in the worker process.
    """
    checker = LeanProofChecker(
        project_dir=project_dir,
        timeout=timeout,
    )
    return checker.check(code)


class BatchChecker:
    """Parallel proof checker using a process pool.

    Each proof check is stateless and independent, enabling trivial
    parallelism across all CPU cores.

    When a project_dir is provided (or auto-detected), subprocess
    workers also use the Lake-managed Lean project for Mathlib4 access.
    """

    def __init__(
        self,
        project_dir: str | Path | None = None,
        timeout: float = 10.0,
        max_workers: int = 12,
        cache_size: int = 50000,
    ):
        self.checker = LeanProofChecker(
            project_dir=project_dir,
            timeout=timeout,
            cache_size=cache_size,
        )
        self.project_dir = str(self.checker.project_dir) if self.checker.project_dir else None
        self.timeout = timeout
        self.max_workers = max_workers

    def check_batch(self, codes: list[str]) -> list[ProofResult]:
        """Check multiple proofs in parallel.

        Uses ProcessPoolExecutor to distribute work across CPU cores.
        Results are returned in the same order as input codes.
        """
        if len(codes) <= 1:
            return self.checker.check_batch(codes)

        results: dict[int, ProofResult] = {}

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    _check_single_worker, code, self.project_dir, self.timeout
                ): i
                for i, code in enumerate(codes)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = ProofResult(
                        success=False,
                        errors=[f"Batch checker error: {str(e)}"],
                        num_tokens=0,
                    )

        return [results[i] for i in range(len(codes))]

    @property
    def cache_hit_rate(self) -> float:
        return self.checker.cache.hit_rate

    @property
    def cache_size(self) -> int:
        return self.checker.cache.size
