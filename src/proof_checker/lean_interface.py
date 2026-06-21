"""Interface to Lean 4 for automated proof verification.

Core design:
1. Receives Lean 4 code strings from the model
2. Wraps them with necessary imports
3. Writes code to a temp .lean file in the project directory
4. Runs ``lean <temp_file>`` to type-check the code
5. Parses exit code and stderr to determine success/failure

Performance:
- Lake environment captured once and reused across all checks
- Temp-file approach avoids stdin pipe issues (SIGPIPE hangs)
- SHA-256 proof cache with configurable size

Reliability improvements over the old stdin-pipe design:
- No stdin pipe: code is written to a temp .lean file, lean reads from disk
- Process groups: each check runs in its own process group for clean kills
- Exponential backoff retry with automatic Lake env re-capture on failure
- Each temp file gets a unique name to avoid filesystem races
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.proof_checker.cache import ProofCache
from src.proof_checker.formats import (
    LEAN_PREAMBLE_LIGHT,
    LEAN_PREAMBLE_MATHLIB,
    ProofResult,
    parse_lean_error,
    wrap_lean_code,
)


def _find_project_dir() -> Path | None:
    """Auto-detect the proof_checker_env Lake project."""
    this_dir = Path(__file__).resolve().parent
    candidate = this_dir.parent.parent / "proof_checker_env"
    if candidate.is_dir() and (candidate / "lakefile.lean").exists():
        return candidate
    candidate = Path.cwd() / "proof_checker_env"
    if candidate.is_dir() and (candidate / "lakefile.lean").exists():
        return candidate
    return None


# ---------------------------------------------------------------------------
# Lake environment cache (thread-safe, shared across all checker instances)
# ---------------------------------------------------------------------------

_lake_env_cache: dict[str, str] | None = None
_lake_env_lock = None  # lazily initialized


def _capture_lake_env(project_dir: Path, timeout: float = 60.0) -> dict[str, str]:
    """Capture the full Lake-managed environment once."""
    global _lake_env_cache, _lake_env_lock

    if _lake_env_cache is not None:
        return _lake_env_cache

    import threading

    if _lake_env_lock is None:
        _lake_env_lock = threading.Lock()

    with _lake_env_lock:
        if _lake_env_cache is not None:
            return _lake_env_cache

        env = os.environ.copy()

        # Try lake env printenv
        try:
            result = subprocess.run(
                ["lake", "env", "printenv"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(project_dir),
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, _, v = line.partition("=")
                        env[k] = v
                _lake_env_cache = env
                return env
        except Exception:
            pass

        # Try lake env sh -c 'env'
        try:
            result = subprocess.run(
                ["lake", "env", "sh", "-c", "env"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(project_dir),
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, _, v = line.partition("=")
                        env[k] = v
                _lake_env_cache = env
                return env
        except Exception:
            pass

        _lake_env_cache = env
        return env


def _clear_lake_env_cache() -> None:
    """Clear the cached Lake environment (for testing)."""
    global _lake_env_cache
    _lake_env_cache = None


# ---------------------------------------------------------------------------
# LeanProofChecker
# ---------------------------------------------------------------------------


class LeanProofChecker:
    """Stateless interface to Lean 4 for proof verification.

    Each check writes the code to a temp .lean file in the project
    directory and runs ``lean <file>`` to verify it.  This avoids
    stdin pipe issues while keeping per-check isolation.

    When a Lake-managed project directory is available (proof_checker_env/),
    proofs are checked with the pre-captured Lake environment for Mathlib4
    access.  Falls back to bare ``lean`` otherwise.
    """

    def __init__(
        self,
        project_dir: str | Path | None | bool = None,
        timeout: float = 10.0,
        cache_size: int = 50000,
        lean_binary: str = "lean",
        max_retries: int = 3,
    ):
        self.timeout = timeout
        self.lean_binary = lean_binary
        self.max_retries = max_retries

        if project_dir is False:
            self.project_dir = None
        elif project_dir is not None:
            self.project_dir = Path(project_dir)
        else:
            self.project_dir = _find_project_dir()

        self.cache = ProofCache(max_size=cache_size)
        self._lake_env: dict[str, str] | None = None

        if self.project_dir:
            self._lake_env = _capture_lake_env(self.project_dir)

    def check(self, code: str, timeout: float | None = None) -> ProofResult:
        """Check if a Lean 4 code string type-checks.

        Args:
            code: The generated Lean 4 code (theorem + proof).
            timeout: Override default timeout in seconds.

        Returns:
            ProofResult with success status and error details.
        """
        timeout = timeout or self.timeout

        preamble = LEAN_PREAMBLE_MATHLIB if self.project_dir else LEAN_PREAMBLE_LIGHT
        wrapped = wrap_lean_code(code, preamble=preamble)
        num_tokens = len(code.split())

        # Cache lookup
        cached = self.cache.get(wrapped)
        if cached is not None:
            return cached

        start = time.time()
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                result = self._run_lean_check(wrapped, timeout)
                result.check_time_ms = (time.time() - start) * 1000
                result.num_tokens = num_tokens
                self.cache.put(wrapped, result)
                return result
            except subprocess.TimeoutExpired as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    backoff = 1.0 * (2 ** attempt)
                    time.sleep(backoff)
                    continue
                result = ProofResult(
                    success=False,
                    errors=[f"Proof check timed out after {self.max_retries} attempts"],
                    num_tokens=num_tokens,
                    timed_out=True,
                )
                result.check_time_ms = (time.time() - start) * 1000
                self.cache.put(wrapped, result)
                return result
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    backoff = 0.5 * (2 ** attempt)
                    time.sleep(backoff)
                    continue
                result = ProofResult(
                    success=False,
                    errors=[f"Proof checker error: {str(e)}"],
                    num_tokens=num_tokens,
                )
                result.check_time_ms = (time.time() - start) * 1000
                self.cache.put(wrapped, result)
                return result

        result = ProofResult(
            success=False,
            errors=[f"Proof checker error: {str(last_error)}"],
            num_tokens=num_tokens,
        )
        result.check_time_ms = (time.time() - start) * 1000
        self.cache.put(wrapped, result)
        return result

    def _run_lean_check(self, code: str, timeout: float) -> ProofResult:
        """Execute ``lean`` on code written to a temp file.

        Avoids stdin pipes entirely — code is written to a .lean file
        inside the project directory (or a system temp dir for bare lean),
        then ``lean <file>`` is invoked.  This eliminates SIGPIPE and
        stdin buffer hangs.
        """
        # Choose the temp directory: project dir (for Mathlib4 access)
        # or system temp.
        if self.project_dir and self.project_dir.is_dir():
            temp_dir = str(self.project_dir)
        else:
            temp_dir = None

        # Write code to temp file
        tf = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            prefix="theta_check_",
            dir=temp_dir,
            delete=False,
            encoding="utf-8",
        )
        try:
            tf.write(code)
            tf.flush()
            tf.close()  # Close so lean can read it

            # Build command
            if self._lake_env:
                cmd = [self.lean_binary, tf.name]
                env = self._lake_env
                cwd = str(self.project_dir)
            elif self.project_dir and self.project_dir.exists():
                cmd = ["lake", "env", "lean", tf.name]
                env = None
                cwd = str(self.project_dir)
            else:
                cmd = [self.lean_binary, tf.name]
                env = None
                cwd = None

            # Run with process group for clean kill on timeout
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=env,
                preexec_fn=os.setsid,  # new process group
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Kill the entire process group
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    proc.kill()
                proc.wait(timeout=5)
                raise

            success = proc.returncode == 0
            errors = []
            if not success:
                error_output = stderr or stdout or ""
                errors = parse_lean_error(error_output)

            return ProofResult(
                success=success,
                errors=errors,
                num_tokens=len(code.split()),
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(tf.name)
            except OSError:
                pass

    def check_batch_fast(
        self, codes: list[str], timeout: float | None = None
    ) -> list[ProofResult]:
        """Check multiple proofs efficiently.

        Strategy: write all codes to one temp file, run ``lean`` once.
        If exit code 0 → all proofs pass.
        If exit code != 0 → fall back to individual checks (cached).
        Common case (many passes): 1 lean invocation.
        Worst case (mixed): 1 batch run + N individual checks (cached).
        """
        import subprocess as _sp
        import os as _os
        import signal as _signal
        import tempfile

        n = len(codes)
        if n == 0:
            return []
        if n == 1:
            return [self.check(codes[0], timeout=timeout)]

        timeout_val = timeout or self.timeout
        preamble = LEAN_PREAMBLE_MATHLIB if self.project_dir else LEAN_PREAMBLE_LIGHT

        # Build a single file with all proofs
        lines: list[str] = [preamble, ""]
        for idx, code in enumerate(codes):
            wrapped = wrap_lean_code(code, include_preamble=False)
            lines.append(f"-- theta_proof_{idx}")
            for code_line in wrapped.split("\n"):
                lines.append(code_line)
            lines.append("")

        full_code = "\n".join(lines)
        num_tokens = sum(len(c.split()) for c in codes) // max(n, 1)

        # Write to temp file
        if self.project_dir and self.project_dir.is_dir():
            temp_dir = str(self.project_dir)
        else:
            temp_dir = None

        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", prefix="theta_batch_",
            dir=temp_dir, delete=False, encoding="utf-8",
        )
        tf.write(full_code)
        tf.flush()
        tf.close()

        try:
            # Single lean invocation for the whole batch
            if self._lake_env:
                cmd = [self.lean_binary, tf.name]
                env = self._lake_env
                cwd = str(self.project_dir)
            elif self.project_dir and self.project_dir.exists():
                cmd = ["lake", "env", "lean", tf.name]
                env = None
                cwd = str(self.project_dir)
            else:
                cmd = [self.lean_binary, tf.name]
                env = None
                cwd = None

            proc = _sp.Popen(
                cmd, stdout=_sp.PIPE, stderr=_sp.PIPE, text=True,
                cwd=cwd, env=env, preexec_fn=_os.setsid,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_val * max(1, n // 5))
            except _sp.TimeoutExpired:
                try:
                    _os.killpg(_os.getpgid(proc.pid), _signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    proc.kill()
                proc.wait(timeout=5)
                return [
                    ProofResult(success=False, errors=["Batch check timed out"],
                                num_tokens=num_tokens, timed_out=True)
                    for _ in codes
                ]

            if proc.returncode == 0:
                # All proofs pass — single lean invocation sufficed
                return [
                    ProofResult(success=True, errors=[], num_tokens=num_tokens)
                    for _ in codes
                ]

            # Mixed results — fall back to individual checks (cached, fast)
            results: list[ProofResult] = []
            for code in codes:
                results.append(self.check(code, timeout=timeout_val))
            return results

        finally:
            try:
                _os.unlink(tf.name)
            except OSError:
                pass

    def check_batch(
        self, codes: list[str], timeout: float | None = None
    ) -> list[ProofResult]:
        """Check multiple proofs using fast batch mode (single lean invocation).

        For batches of 2+ proofs, uses :meth:`check_batch_fast` which bundles
        all proofs into one temp file and runs ``lean --json`` once.
        Single-proof batches fall through to :meth:`check`.
        """
        if len(codes) > 1:
            return self.check_batch_fast(codes, timeout=timeout)
        return [self.check(code, timeout=timeout) for code in codes]
