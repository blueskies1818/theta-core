"""Parallel batch proof checking using ProcessPoolExecutor.

Each worker process creates one ``LeanProofChecker`` and reuses it for all
codes in its batch.  The checker writes code to temp .lean files (no stdin
pipes) and runs ``lean <file>`` per check — reliable, avoids SIGPIPE hangs.

Key reliability improvements over previous version:
- Temp-file backend (no stdin pipe issues)
- ``spawn`` start method (no fork+tokenizer deadlocks)
- Process group kills for timeout cleanup
- Reduced concurrency (4 workers) to avoid resource exhaustion
- Explicit garbage collection after each batch
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from src.proof_checker.lean_interface import LeanProofChecker
from src.proof_checker.formats import ProofResult


def _check_batch_worker(
    codes: list[str],
    project_dir: Optional[str],
    timeout: float,
    max_retries: int,
    worker_id: int,
) -> list[ProofResult]:
    """Check a batch of proofs in a worker process.

    Creates ONE LeanProofChecker and reuses it for all codes in the batch.
    The checker writes code to temp files and runs ``lean <file>`` per check.
    """
    checker = LeanProofChecker(
        project_dir=project_dir,
        timeout=timeout,
        max_retries=max_retries,
    )
    results = []
    for i, code in enumerate(codes):
        try:
            result = checker.check(code)
        except Exception as e:
            result = ProofResult(
                success=False,
                errors=[f"Worker {worker_id} check error: {str(e)}"],
                num_tokens=0,
            )
        results.append(result)

        # Periodic gc to avoid memory bloat from temp file handles
        if (i + 1) % 50 == 0:
            gc.collect()

    return results


# Use 'spawn' to avoid the fork+tokenizer deadlock issue.
# LeanProofChecker doesn't import tokenizers at module level, but
# PyTorch can be loaded by the eval process and fork+pytorch is unsafe.
_MP_CONTEXT = mp.get_context("spawn")


class BatchChecker:
    """Parallel proof checker using process pool with temp-file backend.

    Divides work into batches processed by ``max_workers`` worker processes.
    Each worker reuses one ``LeanProofChecker`` across multiple codes,
    eliminating the cost of creating a new checker per single-code invocation.
    """

    def __init__(
        self,
        project_dir: str | Path | None = None,
        timeout: float = 10.0,
        max_workers: int = 4,
        cache_size: int = 50000,
        max_retries: int = 3,
        min_batch_size: int = 1,
    ):
        self.checker = LeanProofChecker(
            project_dir=project_dir,
            timeout=timeout,
            cache_size=cache_size,
            max_retries=max_retries,
        )
        self.project_dir = (
            str(self.checker.project_dir) if self.checker.project_dir else None
        )
        self.timeout = timeout
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.min_batch_size = min_batch_size

        print(
            f"[BatchChecker] Ready: {max_workers} workers, "
            f"timeout={timeout}s, temp-file backend",
            file=sys.stderr,
        )

    def check_batch(self, codes: list[str]) -> list[ProofResult]:
        """Check multiple proofs in parallel.

        Codes are partitioned into ``max_workers`` batches.  Each batch is
        processed by a single worker process that reuses one LeanProofChecker.

        Results are returned in the same order as input codes.
        """
        n = len(codes)
        if n == 0:
            return []

        # Small batches: just use the main-thread checker (no pool overhead)
        if n <= self.min_batch_size:
            return self.checker.check_batch(codes)

        # Partition codes into roughly equal batches for each worker
        num_workers = min(self.max_workers, n)
        batch_size = (n + num_workers - 1) // num_workers
        batches: list[list[str]] = []
        for i in range(0, n, batch_size):
            batches.append(codes[i : i + batch_size])

        num_workers = len(batches)
        results_map: dict[int, list[ProofResult]] = {}

        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=_MP_CONTEXT
        ) as executor:
            futures = {}
            for worker_id, batch_codes in enumerate(batches):
                future = executor.submit(
                    _check_batch_worker,
                    batch_codes,
                    self.project_dir,
                    self.timeout,
                    self.max_retries,
                    worker_id,
                )
                futures[future] = worker_id

            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    results_map[worker_id] = future.result()
                except Exception as e:
                    # Worker crashed — mark all codes in that batch as failed
                    batch_codes = batches[worker_id]
                    results_map[worker_id] = [
                        ProofResult(
                            success=False,
                            errors=[f"Worker {worker_id} crashed: {str(e)}"],
                            num_tokens=0,
                        )
                        for _ in batch_codes
                    ]

        # Reassemble results in original order
        output: list[ProofResult] = []
        for worker_id in range(num_workers):
            output.extend(results_map[worker_id])
        return output

    @property
    def cache_hit_rate(self) -> float:
        return self.checker.cache.hit_rate

    @property
    def cache_size(self) -> int:
        return self.checker.cache.size
