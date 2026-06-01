"""Interface to Lean 4 for automated proof verification.

Core design:
1. Receives Lean 4 code strings from the model
2. Wraps them with necessary imports
3. Invokes `lean` as a subprocess to type-check the code
4. Parses exit code and stderr to determine success/failure
"""

import subprocess
import tempfile
import time
from pathlib import Path

from src.proof_checker.cache import ProofCache
from src.proof_checker.formats import ProofResult, parse_lean_error, wrap_lean_code


class LeanProofChecker:
    """Stateless interface to Lean 4 for proof verification.

    Each check is independent — no mutable state shared between checks.
    Uses subprocess invocation of `lean` for verification.
    """

    def __init__(
        self,
        project_dir: str | Path | None = None,
        timeout: float = 10.0,
        cache_size: int = 50000,
        lean_binary: str = "lean",
    ):
        self.timeout = timeout
        self.lean_binary = lean_binary
        self.project_dir = Path(project_dir) if project_dir else None
        self.cache = ProofCache(max_size=cache_size)

    def check(self, code: str, timeout: float | None = None) -> ProofResult:
        """Check if a Lean 4 code string type-checks.

        Args:
            code: The generated Lean 4 code (theorem + proof).
            timeout: Override default timeout in seconds.

        Returns:
            ProofResult with success status and error details.
        """
        timeout = timeout or self.timeout

        wrapped = wrap_lean_code(code)
        num_tokens = len(code.split())

        cached = self.cache.get(wrapped)
        if cached is not None:
            return cached

        start = time.time()
        try:
            result = self._run_lean_check(wrapped, timeout)
        except subprocess.TimeoutExpired:
            result = ProofResult(
                success=False,
                errors=["Proof check timed out"],
                num_tokens=num_tokens,
                timed_out=True,
            )
        except Exception as e:
            result = ProofResult(
                success=False,
                errors=[f"Proof checker error: {str(e)}"],
                num_tokens=num_tokens,
            )

        result.check_time_ms = (time.time() - start) * 1000
        self.cache.put(wrapped, result)
        return result

    def _run_lean_check(self, code: str, timeout: float) -> ProofResult:
        """Execute lean on the given code string."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            delete=True,
        ) as tmp:
            tmp.write(code)
            tmp.flush()

            cmd = [self.lean_binary, "--stdin"]
            env = {}

            if self.project_dir and self.project_dir.exists():
                cmd = ["lake", "env", "lean", "--stdin"]

            proc = subprocess.run(
                cmd,
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_dir) if self.project_dir else None,
            )

        success = proc.returncode == 0
        errors = []
        if not success:
            error_output = proc.stderr or proc.stdout or ""
            errors = parse_lean_error(error_output)

        return ProofResult(
            success=success,
            errors=errors,
            num_tokens=len(code.split()),
        )

    def check_batch(
        self, codes: list[str], timeout: float | None = None
    ) -> list[ProofResult]:
        """Check multiple proofs sequentially (parallelism via BatchChecker)."""
        return [self.check(code, timeout=timeout) for code in codes]
