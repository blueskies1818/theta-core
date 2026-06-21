"""Persistent Lean 4 proof checker via --server (LSP) protocol.

Replaces per-check subprocess spawning with a single long-lived ``lean --server``
process that handles all proof checks over JSON-RPC / LSP.  Eliminates the
~100ms-3s subprocess spawn + Lake env startup overhead per proof check.

Checks are performed by writing code to a temporary ``.lean`` file in the
project directory and sending a ``textDocument/didOpen`` notification with
the real file URI.  This is necessary because ``lean --server`` only fully
processes files that exist on disk within its workspace.
.
Usage::

    checker = LeanServerChecker(project_dir="proof_checker_env")
    result = checker.check(theorem_code)  # ProofResult
    results = checker.check_batch([code1, code2, ...])  # list[ProofResult]
    checker.shutdown()

Thread-safety: send/recv is protected by a lock so multiple threads can share
a single server process (ThreadPool pattern).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from src.proof_checker.cache import ProofCache
from src.proof_checker.formats import (
    LEAN_PREAMBLE_LIGHT,
    LEAN_PREAMBLE_MATHLIB,
    ProofResult,
    parse_lean_error,
    wrap_lean_code,
)


# ---------------------------------------------------------------------------
# LSP wire helpers
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(rb"Content-Length: (\d+)\r\n\r\n")


def _encode_lsp_message(payload: dict) -> bytes:
    """Encode a JSON dict as an LSP message with Content-Length header."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _read_lsp_message(pipe) -> Optional[dict]:
    """Read one LSP message from a binary pipe.  Returns the parsed JSON dict,
    or None on EOF / decode error."""
    # Read headers
    header_buf = b""
    while b"\r\n\r\n" not in header_buf:
        chunk = pipe.read(1)
        if not chunk:
            return None
        header_buf += chunk
    m = _HEADER_RE.search(header_buf)
    if not m:
        return None
    content_length = int(m.group(1))
    # Read exactly content_length bytes of body
    body = pipe.read(content_length)
    if body is None or len(body) < content_length:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


# ---------------------------------------------------------------------------
# LeanServerChecker
# ---------------------------------------------------------------------------


class LeanServerChecker:
    """Persistent Lean 4 proof checker using ``lean --server`` (LSP mode).

    One instance manages one ``lean --server`` subprocess.  All proof checks
    are routed through this process — zero per-check spawn overhead.

    Checks write code to a reusable temp ``.lean`` file inside the project
    directory and notify the server via ``textDocument/didChange``.  This
    is necessary because Lean only processes files that exist on disk
    within the project workspace.

    Thread-safe: send/recv is protected by ``_lock``.  Multiple threads can
    call :meth:`check` concurrently.
    """

    _INIT_TIMEOUT = 30.0

    def __init__(
        self,
        project_dir: str | Path | None = None,
        timeout: float = 15.0,
        cache_size: int = 50000,
        max_retries: int = 3,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = ProofCache(max_size=cache_size)
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._running = False
        self._doc_version = 0

        # Resolve project_dir
        if project_dir is not None:
            self.project_dir = Path(project_dir)
        else:
            # Auto-detect from src/proof_checker/ -> ../../proof_checker_env
            this_dir = Path(__file__).resolve().parent
            candidate = this_dir.parent.parent / "proof_checker_env"
            if candidate.is_dir() and (candidate / "lakefile.lean").exists():
                self.project_dir = candidate
            else:
                self.project_dir = None

        self._lake_env: dict[str, str] | None = None
        if self.project_dir:
            self._lake_env = self._capture_lake_env(self.project_dir)

        # Create a reusable temp file inside the project for LSP checks.
        # Lean --server only processes files that exist on disk.
        if self.project_dir:
            self._temp_file = tempfile.NamedTemporaryFile(
                suffix=".lean",
                prefix="theta_check_",
                dir=str(self.project_dir),
                delete=False,
                mode="w+",
                encoding="utf-8",
            )
            self._doc_uri = f"file://{self._temp_file.name}"
        else:
            self._temp_file = None
            self._doc_uri = "file:///theta_proof_check.lean"

        self._start_server()

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        """Launch ``lean --server`` as a persistent subprocess."""
        cmd = ["lean", "--server"]
        env = (self._lake_env or os.environ).copy()

        cwd = str(self.project_dir) if self.project_dir else None

        print(
            f"[LeanServerChecker] Starting lean --server (cwd={cwd})...",
            file=sys.stderr,
        )
        t0 = time.time()

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=cwd,
                bufsize=0,  # unbuffered binary
            )
        except FileNotFoundError:
            raise RuntimeError(
                "lean binary not found.  Install Lean 4 via elan: "
                "curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh"
            )

        # Initialize LSP session.  lean --server may not respond to
        # initialize with a proper result in bounded time (project
        # indexing), so we fire-and-forget and drain the startup flood.
        self._send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "processId": os.getpid(),
                    "rootUri": f"file://{cwd}" if cwd else None,
                    "capabilities": {},
                    "workspaceFolders": (
                        [{"uri": f"file://{cwd}", "name": "theta"}]
                        if cwd
                        else None
                    ),
                },
            }
        )
        # Drain until we see the initialize response or timeout
        deadline = time.time() + self._INIT_TIMEOUT
        initialized_ok = False
        while time.time() < deadline:
            msg = self._recv(timeout=min(1.0, deadline - time.time()))
            if msg is None:
                continue
            if msg.get("id") == 1 and "result" in msg:
                initialized_ok = True
                break
            # Also accept the server silently accepting initialize
            # (it may just start sending fileProgress without a result)

        # Send initialized notification (always — server is ready at this point)
        self._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})

        # Drain startup notifications (indexing progress, initial diagnostics)
        self._drain_notifications(timeout=3.0)

        elapsed = time.time() - t0
        self._running = True
        print(
            f"[LeanServerChecker] Server ready ({elapsed:.1f}s)",
            file=sys.stderr,
        )

    def shutdown(self) -> None:
        """Gracefully shut down the server."""
        if not self._running or self._proc is None:
            return
        try:
            self._send({"jsonrpc": "2.0", "id": 9999, "method": "shutdown"})
            self._recv(timeout=5.0)
            self._send({"jsonrpc": "2.0", "method": "exit"})
        except Exception:
            pass
        finally:
            proc = self._proc
            if proc is not None:
                try:
                    if proc.stdin is not None:
                        proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            self._running = False

    def restart(self) -> None:
        """Restart the server (e.g., after a crash)."""
        self.shutdown()
        time.sleep(0.5)
        self._start_server()

    # ------------------------------------------------------------------
    # Proof checking
    # ------------------------------------------------------------------

    def check(self, code: str, timeout: float | None = None) -> ProofResult:
        """Check a single Lean 4 code block.

        Args:
            code: The theorem + proof code to verify.
            timeout: Override default timeout.

        Returns:
            ProofResult with success/error details.
        """
        timeout = timeout or self.timeout
        preamble = (
            LEAN_PREAMBLE_MATHLIB if self.project_dir else LEAN_PREAMBLE_LIGHT
        )
        wrapped = wrap_lean_code(code, preamble=preamble)

        # Cache lookup
        cached = self.cache.get(wrapped)
        if cached is not None:
            return cached

        num_tokens = len(code.split())
        start = time.time()

        for attempt in range(self.max_retries):
            try:
                result = self._check_via_server(wrapped, timeout)
                result.check_time_ms = (time.time() - start) * 1000
                result.num_tokens = num_tokens
                self.cache.put(wrapped, result)
                return result
            except Exception as e:
                if attempt < self.max_retries - 1:
                    backoff = 1.0 * (2 ** attempt)
                    time.sleep(backoff)
                    # Restart server on persistent failures
                    if attempt >= 1:
                        try:
                            self.restart()
                        except Exception:
                            pass
                    continue
                result = ProofResult(
                    success=False,
                    errors=[f"Server check error: {str(e)}"],
                    num_tokens=num_tokens,
                )
                result.check_time_ms = (time.time() - start) * 1000
                self.cache.put(wrapped, result)
                return result

        # Unreachable, but satisfy type checker
        result = ProofResult(
            success=False,
            errors=["Proof checker exhausted retries"],
            num_tokens=num_tokens,
        )
        result.check_time_ms = (time.time() - start) * 1000
        self.cache.put(wrapped, result)
        return result

    def _check_via_server(
        self, wrapped_code: str, timeout: float
    ) -> ProofResult:
        """Write code to temp file, open it via LSP, wait for diagnostics, close."""
        self._doc_version += 1
        doc_uri = self._doc_uri

        with self._lock:
            # Write code to the temp file on disk so lean --server can process it
            if self._temp_file is not None:
                self._temp_file.seek(0)
                self._temp_file.write(wrapped_code)
                self._temp_file.truncate()
                self._temp_file.flush()

            # Open the document — server processes it because file exists on disk
            self._send(
                {
                    "jsonrpc": "2.0",
                    "method": "textDocument/didOpen",
                    "params": {
                        "textDocument": {
                            "uri": doc_uri,
                            "languageId": "lean4",
                            "version": self._doc_version,
                            "text": wrapped_code,
                        }
                    },
                }
            )

            # Wait for diagnostics — Lean sends empty diagnostic snapshots
            # during processing, then the real ones after fileProgress
            # indicates completion.  We use a quiet-period strategy: keep
            # reading until no new messages arrive for QUIET_PERIOD seconds,
            # then use the LAST publishDiagnostics for our URI.
            diagnostics = self._wait_for_diagnostics(doc_uri, timeout)

            # Close the document
            self._send(
                {
                    "jsonrpc": "2.0",
                    "method": "textDocument/didClose",
                    "params": {"textDocument": {"uri": doc_uri}},
                }
            )

        if diagnostics is None:
            return ProofResult(
                success=False,
                errors=["Timed out waiting for Lean diagnostics"],
                num_tokens=0,
                timed_out=True,
            )

        errors = []
        for diag in diagnostics:
            severity = diag.get("severity", 2)
            message = diag.get("message", "")
            # severity 1 = error; also catch explicit error keywords
            is_hard_error = severity == 1 or "error" in message.lower()
            if is_hard_error:
                errors.append(message[:500])

        if not errors:
            return ProofResult(success=True, errors=[], num_tokens=0)
        else:
            return ProofResult(
                success=False,
                errors=parse_lean_error("\n".join(errors)) or errors,
                num_tokens=0,
            )

    # How long to wait with no new messages before considering the
    # Lean server's response complete.
    _QUIET_PERIOD = 1.0
    _POLL_INTERVAL = 0.1

    def _wait_for_diagnostics(
        self, doc_uri: str, timeout: float
    ) -> Optional[list[dict]]:
        """Read LSP messages until a quiet period, then return the last
        publishDiagnostics for *doc_uri*.

        Lean sends multiple publishDiagnostics as file processing
        progresses — early snapshots may have zero diagnostics.  We
        keep the LAST one after the server falls silent.
        """
        deadline = time.time() + timeout
        last_diagnostics: list[dict] | None = None
        last_msg_time = time.time()

        while time.time() < deadline:
            remaining = deadline - time.time()
            msg = self._recv(timeout=min(self._POLL_INTERVAL, remaining))
            if msg is not None:
                last_msg_time = time.time()
                method = msg.get("method", "")
                if method == "textDocument/publishDiagnostics":
                    params = msg.get("params", {})
                    if params.get("uri") == doc_uri:
                        last_diagnostics = params.get("diagnostics", [])
                # Also check for fileProgress with 'kind': 1 (done) for
                # all processing items — this confirms processing is complete
                elif method == "$/lean/fileProgress":
                    params = msg.get("params", {})
                    processing = params.get("processing", [])
                    if processing and all(
                        p.get("kind") == 1 for p in processing
                    ):
                        # All processing stages are done — we can return now
                        if last_diagnostics is not None:
                            return last_diagnostics
            else:
                # No message received in this poll interval.
                # If we've been quiet long enough, return what we have.
                if last_diagnostics is not None and (
                    time.time() - last_msg_time >= self._QUIET_PERIOD
                ):
                    return last_diagnostics
                # If we have no diagnostics yet and quiet period has elapsed
                # since the last message, keep waiting (server may be slow).
                # But if we've been completely silent for half the timeout,
                # that's suspicious.
                if (
                    last_diagnostics is None
                    and time.time() - last_msg_time > timeout * 0.5
                ):
                    return []

        # Timeout: return last diagnostics if we have them
        return last_diagnostics

    def _drain_notifications(self, timeout: float = 2.0) -> None:
        """Read and discard pending notifications (non-blocking drain)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(0.1, deadline - time.time())
            msg = self._recv(timeout=min(remaining, 1.0))
            if msg is None:
                break

    def check_batch(
        self, codes: list[str], timeout: float | None = None
    ) -> list[ProofResult]:
        """Check multiple proofs sequentially against the same server."""
        return [self.check(code, timeout=timeout) for code in codes]

    # ------------------------------------------------------------------
    # Wire protocol (thread-safe)
    # ------------------------------------------------------------------

    def _send(self, msg: dict) -> None:
        """Send a JSON-RPC message to the server.  Must hold ``_lock``."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Lean server not running")
        data = _encode_lsp_message(msg)
        try:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
        except BrokenPipeError:
            self._running = False
            raise RuntimeError("Lean server pipe broken — process may have crashed")

    def _recv(self, timeout: float = 10.0) -> Optional[dict]:
        """Read one LSP message from the server.  Must hold ``_lock``.

        Returns the parsed JSON dict, or None on timeout/error.
        """
        if self._proc is None or self._proc.stdout is None:
            return None

        # Simple poll loop — read with timeout
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Check if data is available
            import select

            remaining = max(0.1, deadline - time.time())
            try:
                ready, _, _ = select.select(
                    [self._proc.stdout], [], [], remaining
                )
            except (ValueError, OSError):
                return None

            if not ready:
                continue

            try:
                return _read_lsp_message(self._proc.stdout)
            except Exception:
                return None

        return None

    # ------------------------------------------------------------------
    # Lake environment capture (same as lean_interface.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _capture_lake_env(project_dir: Path, timeout: float = 60.0) -> dict[str, str]:
        """Capture Lake-managed environment variables."""
        import threading as _thr

        _captured: dict[str, str] | None = None
        _cap_lock = _thr.Lock()

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
                return env
        except Exception:
            pass

        return env

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
