"""Parallel batch proof checking using ProcessPoolExecutor.

The proof checker runs on CPU and is the throughput bottleneck.
This module provides parallel checking across all available CPU cores.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.proof_checker.lean_interface import LeanProofChecker
from src.proof_checker.formats import ProofResult


class BatchChecker:
    """Parallel proof checker using a process pool.

    Each proof check is stateless and independent, enabling trivial
    parallelism across all CPU cores.
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
                executor.submit(self._check_single, code, i): i
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

    @staticmethod
    def _check_single(code: str, idx: int) -> ProofResult:
        """Check a single proof. Called in a subprocess.

        Note: Creates a fresh checker instance because LeanProofChecker
        is not pickleable (contains subprocess state).
        """
        checker = LeanProofChecker()
        return checker.check(code)

    @property
    def cache_hit_rate(self) -> float:
        return self.checker.cache.hit_rate

    @property
    def cache_size(self) -> int:
        return self.checker.cache.size
