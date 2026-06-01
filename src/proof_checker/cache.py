"""Proof checking result cache.

Avoids re-checking identical code strings by hashing and caching results.
"""

import hashlib
from collections import OrderedDict

from src.proof_checker.formats import ProofResult


class ProofCache:
    """LRU cache for proof checking results.

    Keyed by SHA-256 hash of the code string being checked.
    """

    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self._cache: OrderedDict[str, ProofResult] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _hash(code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def get(self, code: str) -> ProofResult | None:
        """Return cached result or None."""
        key = self._hash(code)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, code: str, result: ProofResult) -> None:
        """Store a result in the cache."""
        key = self._hash(code)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = result

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
