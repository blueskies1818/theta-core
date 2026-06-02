"""Lean 4 proof verifier interface.

Stateless, deterministic verification via subprocess invocation.
Includes parallel batch checking, result caching, and code formatting.

This is the system's source of ground-truth training signal — the
analog of the Go rule checker in AlphaGo Zero.

See mathematical_ai_system.md § Formal Proof System.
"""