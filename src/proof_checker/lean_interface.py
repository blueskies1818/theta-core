"""Interface to Lean 4 for automated proof verification.

Core design:
1. Receives Lean 4 code strings from the model
2. Wraps them with necessary imports
3. Invokes `lean` as a subprocess to type-check the code
4. Parses exit code and stderr to determine success/failure
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.proof_checker.cache import ProofCache
from src.proof_checker.formats import ProofResult, parse_lean_error, wrap_lean_code


def _find_project_dir() -> Path | None:
    """Auto-detect the proof_checker_env Lake project.

    Searches upward from this file's location and from CWD.
    Returns the path to the Lake project directory, or None.
    """
    candidates = []

    # Relative to this file: src/proof_checker/ -> ../../proof_checker_env
    this_dir = Path(__file__).resolve().parent
    candidates.append(this_dir.parent.parent / "proof_checker_env")

    # Relative to CWD
    candidates.append(Path.cwd() / "proof_checker_env")

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "lakefile.lean").exists():
            return candidate
    return None


class LeanProofChecker:
    """Stateless interface to Lean 4 for proof verification.

    Each check is independent — no mutable state shared between checks.
    Uses subprocess invocation of ``lean`` for verification.

    When a Lake-managed project directory is available (proof_checker_env/),
    proofs are checked with ``lake env lean`` which provides access to
    Mathlib4. Otherwise falls back to bare ``lean --stdin``.

    Pass ``project_dir=False`` to force bare lean even when a Lake project
    is available (useful for testing with core tactics only).
    """

    def __init__(
        self,
        project_dir: str | Path | None | bool = None,
        timeout: float = 10.0,
        cache_size: int = 50000,
        lean_binary: str = "lean",
    ):
        self.timeout = timeout
        self.lean_binary = lean_binary

        # Resolve project_dir:
        # - str/Path -> use that project
        # - None (default) -> auto-detect
        # - False -> force bare lean (no Lake project, even if available)
        if project_dir is False:
            self.project_dir = None
        elif project_dir is not None:
            self.project_dir = Path(project_dir)
        else:
            self.project_dir = _find_project_dir()

        self.cache = ProofCache(max_size=cache_size)

        if self.project_dir:
            print(f"[LeanProofChecker] Using Lake project: {self.project_dir}",
                  file=sys.stderr)
        else:
            print("[LeanProofChecker] Using bare lean --stdin (no Mathlib4)",
                  file=sys.stderr)

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
