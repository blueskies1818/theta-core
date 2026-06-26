"""Unbounded mathematical invariance generator.

Generates mathematical identities — not physics formulas, but genuine
algebraic, trigonometric, exponential, and structural invariants that
hold for ALL values of their variables. The system learns mathematical
invariance from pure structure; physics is one application domain.

Design principle: the generator knows the ground-truth identity. It
generates noisy measurements of BOTH sides independently. The system
must discover that the two sides are equal — that the identity holds
despite measurement noise.

Symbol pool is virtually unbounded: random letter + digit combinations
generated on the fly (a0..z99, aa0..zz99, etc.), assigned to random
mathematical groups with specific transformation properties.
"""

from __future__ import annotations

import math as _math
import random
from dataclasses import dataclass, field
from typing import Callable


# ═══════════════════════════════════════════════════════════════════════════
# Symbol factory — unbounded, random-sampled for genuine cross-symbol diversity
# ═══════════════════════════════════════════════════════════════════════════

# Pre-build a limited reusable pool: ~100 symbols.
# The SAME symbols appear in DIFFERENT structural roles across variants,
# forcing the model to learn patterns, not symbol-to-formula mappings.
_SYMBOL_POOL: list[str] = []
for _base in range(4):  # a0..z3 = 104 symbols
    for _letter_idx in range(26):
        _letter = chr(ord('a') + _letter_idx)
        if _base == 0:
            _SYMBOL_POOL.append(_letter)
        else:
            _SYMBOL_POOL.append(f"{_letter}{_base}")
# Also include uppercase letters for physics symbols (E, P, V, etc.)
for _letter_idx in range(26):
    _SYMBOL_POOL.append(chr(ord('A') + _letter_idx))

_GEN_RNG: random.Random | None = None
_SHUFFLED_POOL: list[str] = []


def _random_symbol() -> str:
    """Return a random symbol. O(1). Reshuffles when exhausted."""
    global _SHUFFLED_POOL, _GEN_RNG
    if not _SHUFFLED_POOL:
        _SHUFFLED_POOL = list(_SYMBOL_POOL)
        if _GEN_RNG is None:
            _GEN_RNG = random.Random(0)
        _GEN_RNG.shuffle(_SHUFFLED_POOL)
    return _SHUFFLED_POOL.pop()


def reset_symbol_pool(seed: int = 0) -> None:
    """Reset for a new generation session. Different seeds produce
    different shuffled symbol sequences."""
    global _SHUFFLED_POOL, _GEN_RNG
    _SHUFFLED_POOL = []
    _GEN_RNG = random.Random(seed)


# ═══════════════════════════════════════════════════════════════════════════
# Invariant types — the mathematical identities we can generate
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MathInvariant:
    """A mathematical identity with ground truth.

    - expression: the invariant expression (what stays constant)
    - identity_type: category (algebraic, trigonometric, exponential, etc.)
    - variables: list of variable symbols used
    - complexity: 1-5 indicating structural depth
    - description: human-readable explanation
    - generate_values: function(num_configs) -> list[dict] of variable bindings
    """
    expression: str
    identity_type: str
    variables: list[str]
    complexity: int
    description: str
    generate_values: Callable[[int, random.Random], list[dict[str, float]]] = field(repr=False)


# ═══════════════════════════════════════════════════════════════════════════
# Algebraic invariants (commutativity, associativity, distributivity)
# ═══════════════════════════════════════════════════════════════════════════

def _make_algebraic_commutative_add() -> MathInvariant:
    a, b = _random_symbol(), _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(-100, 100), b: rng.uniform(-100, 100)}
                for _ in range(n)]
    return MathInvariant(
        expression=f"{a}+{b}",
        identity_type="algebraic_commutative",
        variables=[a, b],
        complexity=1,
        description=f"{a}+{b} = {b}+{a} (addition is commutative, so {a}+{b} is "
                     "invariant under swap of addends — but only if measured the same)",
        generate_values=values,
    )


def _make_algebraic_commutative_mul() -> MathInvariant:
    a, b = _random_symbol(), _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(0.1, 100), b: rng.uniform(0.1, 100)}
                for _ in range(n)]
    return MathInvariant(
        expression=f"{a}*{b}",
        identity_type="algebraic_commutative",
        variables=[a, b],
        complexity=1,
        description=f"{a}*{b} = {b}*{a} (multiplication is commutative)",
        generate_values=values,
    )


def _make_algebraic_associative() -> MathInvariant:
    a, b, c = _random_symbol(), _random_symbol(), _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(-50, 50), b: rng.uniform(-50, 50),
                 c: rng.uniform(-50, 50)} for _ in range(n)]
    return MathInvariant(
        expression=f"({a}+{b})+{c}",
        identity_type="algebraic_associative",
        variables=[a, b, c],
        complexity=2,
        description=f"({a}+{b})+{c} = {a}+({b}+{c}) (addition is associative, "
                     "so the grouped sum is invariant under regrouping)",
        generate_values=values,
    )


def _make_algebraic_distributive() -> MathInvariant:
    a, b, c = _random_symbol(), _random_symbol(), _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(-20, 20), b: rng.uniform(-20, 20),
                 c: rng.uniform(-20, 20)} for _ in range(n)]
    return MathInvariant(
        expression=f"{a}*({b}+{c})",
        identity_type="algebraic_distributive",
        variables=[a, b, c],
        complexity=2,
        description=f"{a}*({b}+{c}) = {a}*{b}+{a}*{c} (distributive law, "
                     "expression is equivalent to expanded form)",
        generate_values=values,
    )


def _make_algebraic_identity_add() -> MathInvariant:
    a = _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(-100, 100)} for _ in range(n)]
    return MathInvariant(
        expression=f"{a}+0",
        identity_type="algebraic_identity",
        variables=[a],
        complexity=1,
        description=f"{a}+0 = {a} (additive identity — the zero doesn't change anything)",
        generate_values=values,
    )


def _make_algebraic_identity_mul() -> MathInvariant:
    a = _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(0.1, 100)} for _ in range(n)]
    return MathInvariant(
        expression=f"{a}*1",
        identity_type="algebraic_identity",
        variables=[a],
        complexity=1,
        description=f"{a}*1 = {a} (multiplicative identity)",
        generate_values=values,
    )


def _make_algebraic_inverse() -> MathInvariant:
    a = _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(1, 100)} for _ in range(n)]
    return MathInvariant(
        expression=f"{a}/{a}",
        identity_type="algebraic_inverse",
        variables=[a],
        complexity=1,
        description=f"{a}/{a} = 1 for a ≠ 0 (multiplicative inverse)",
        generate_values=values,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Power invariants
# ═══════════════════════════════════════════════════════════════════════════

def _make_power_product_rule() -> MathInvariant:
    a, m, n = _random_symbol(), _random_symbol(), _random_symbol()
    def values(n_configs: int, rng: random.Random) -> list[dict[str, float]]:
        vals = []
        for _ in range(n_configs):
            a_val = rng.uniform(0.5, 5.0)
            m_val = float(rng.randint(1, 5))
            n_val = float(rng.randint(1, 5))
            # a^m * a^n is NOT constant across different a values.
            # We need: a^m * a^n / a^(m+n) = 1 — but that's a 3-variable identity.
            # Better: generate a, m, n fixed. Test: a^m * a^n
            vals.append({a: a_val, m: m_val, n: n_val})
        return vals
    return MathInvariant(
        expression=f"{a}^{m}*{a}^{n}",
        identity_type="power_rule",
        variables=[a, m, n],
        complexity=2,
        description=f"{a}^{m} * {a}^{n} = {a}^({m}+{n})",
        generate_values=values,
    )


def _make_power_power_rule() -> MathInvariant:
    a, m, n = _random_symbol(), _random_symbol(), _random_symbol()
    def values(n_configs: int, rng: random.Random) -> list[dict[str, float]]:
        vals = []
        for _ in range(n_configs):
            a_val = rng.uniform(0.5, 3.0)
            m_val = float(rng.randint(1, 3))
            n_val = float(rng.randint(1, 3))
            vals.append({a: a_val, m: m_val, n: n_val})
        return vals
    return MathInvariant(
        expression=f"({a}^{m})^{n}",
        identity_type="power_rule",
        variables=[a, m, n],
        complexity=2,
        description=f"({a}^{m})^{n} = {a}^({m}*{n})",
        generate_values=values,
    )


def _make_difference_of_squares() -> MathInvariant:
    a, b = _random_symbol(), _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(-50, 50), b: rng.uniform(-50, 50)}
                for _ in range(n)]
    return MathInvariant(
        expression=f"{a}^2-{b}^2",
        identity_type="algebraic_factorization",
        variables=[a, b],
        complexity=2,
        description=f"{a}^2-{b}^2 = ({a}+{b})({a}-{b}) — difference of squares",
        generate_values=values,
    )


def _make_square_of_sum() -> MathInvariant:
    a, b = _random_symbol(), _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(-20, 20), b: rng.uniform(-20, 20)}
                for _ in range(n)]
    return MathInvariant(
        expression=f"({a}+{b})^2",
        identity_type="algebraic_expansion",
        variables=[a, b],
        complexity=2,
        description=f"({a}+{b})^2 = {a}^2+2*{a}*{b}+{b}^2",
        generate_values=values,
    )


def _make_product_ratio() -> MathInvariant:
    """(a*b)/(c*d) — composition of multiplication and division across four symbols."""
    a, b, c, d = _random_symbol(), _random_symbol(), _random_symbol(), _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(1, 30), b: rng.uniform(1, 30),
                 c: rng.uniform(1, 30), d: rng.uniform(1, 30)} for _ in range(n)]
    return MathInvariant(
        expression=f"({a}*{b})/({c}*{d})",
        identity_type="algebraic_composition",
        variables=[a, b, c, d],
        complexity=3,
        description=f"({a}*{b})/({c}*{d}) — product ratio",
        generate_values=values,
    )


def _make_sum_product() -> MathInvariant:
    """(a+b)*(c+d) — distributive span across two sums."""
    a, b, c, d = _random_symbol(), _random_symbol(), _random_symbol(), _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(-30, 30), b: rng.uniform(-30, 30),
                 c: rng.uniform(-30, 30), d: rng.uniform(-30, 30)} for _ in range(n)]
    return MathInvariant(
        expression=f"({a}+{b})*({c}+{d})",
        identity_type="algebraic_composition",
        variables=[a, b, c, d],
        complexity=3,
        description=f"({a}+{b})*({c}+{d}) — sum product",
        generate_values=values,
    )


def _make_power_of_sum() -> MathInvariant:
    """a^(b+c) — power over sum. Composition of exponentiation and addition."""
    a, b, c = _random_symbol(), _random_symbol(), _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(1.1, 5), b: rng.uniform(0.5, 3),
                 c: rng.uniform(0.5, 3)} for _ in range(n)]
    return MathInvariant(
        expression=f"{a}^({b}+{c})",
        identity_type="algebraic_composition",
        variables=[a, b, c],
        complexity=3,
        description=f"{a}^({b}+{c}) — power of sum",
        generate_values=values,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Trigonometric invariants
# ═══════════════════════════════════════════════════════════════════════════

def _make_trig_pythagorean() -> MathInvariant:
    a = _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(-_math.pi, _math.pi)} for _ in range(n)]
    return MathInvariant(
        expression=f"sin({a})^2+cos({a})^2",
        identity_type="trigonometric",
        variables=[a],
        complexity=3,
        description=f"sin^2({a}) + cos^2({a}) = 1 for all {a}",
        generate_values=values,
    )


def _make_trig_tangent_identity() -> MathInvariant:
    a = _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        # Avoid cos(a) near zero
        vals = []
        for _ in range(n):
            av = rng.uniform(-_math.pi/4, _math.pi/4)
            vals.append({a: av})
        return vals
    return MathInvariant(
        expression=f"sin({a})/cos({a})",
        identity_type="trigonometric",
        variables=[a],
        complexity=3,
        description=f"sin({a})/cos({a}) = tan({a}) — ratio defines tangent",
        generate_values=values,
    )


def _make_trig_double_angle() -> MathInvariant:
    a = _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(-_math.pi/2, _math.pi/2)} for _ in range(n)]
    return MathInvariant(
        expression=f"2*sin({a})*cos({a})",
        identity_type="trigonometric",
        variables=[a],
        complexity=4,
        description=f"2*sin({a})*cos({a}) = sin(2*{a}) (double angle formula)",
        generate_values=values,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Exponential / logarithmic invariants
# ═══════════════════════════════════════════════════════════════════════════

def _make_exp_product_rule() -> MathInvariant:
    a, b = _random_symbol(), _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(-3, 3), b: rng.uniform(-3, 3)}
                for _ in range(n)]
    return MathInvariant(
        expression=f"exp({a}+{b})",
        identity_type="exponential",
        variables=[a, b],
        complexity=3,
        description=f"exp({a}+{b}) = exp({a})*exp({b})",
        generate_values=values,
    )


def _make_log_product_rule() -> MathInvariant:
    a, b = _random_symbol(), _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(0.1, 100), b: rng.uniform(0.1, 100)}
                for _ in range(n)]
    return MathInvariant(
        expression=f"log({a}*{b})",
        identity_type="logarithmic",
        variables=[a, b],
        complexity=3,
        description=f"log({a}*{b}) = log({a})+log({b})",
        generate_values=values,
    )


def _make_exp_log_inverse() -> MathInvariant:
    a = _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(0.1, 100)} for _ in range(n)]
    return MathInvariant(
        expression=f"exp(log({a}))",
        identity_type="exponential",
        variables=[a],
        complexity=2,
        description=f"exp(log({a})) = {a} (inverse functions)",
        generate_values=values,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Structural / symmetry invariants
# ═══════════════════════════════════════════════════════════════════════════

def _make_symmetry_swap_add() -> MathInvariant:
    """a+b is invariant under swapping a and b."""
    a, b = _random_symbol(), _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(-100, 100), b: rng.uniform(-100, 100),
                 f"{a}_swapped": rng.uniform(-100, 100),
                 f"{b}_swapped": rng.uniform(-100, 100)}
                for _ in range(n)]
    # This one is tricky — we need the same pair of values assigned to
    # (a,b) then to (b,a). Better: pre-compute a constant expression
    # and provide it directly.
    return MathInvariant(
        expression=f"{a}+{b}",
        identity_type="symmetry",
        variables=[a, b],
        complexity=1,
        description=f"{a}+{b} = {b}+{a} (symmetric under swap)",
        generate_values=values,
    )


def _make_symmetry_rearrangement() -> MathInvariant:
    a, b, c = _random_symbol(), _random_symbol(), _random_symbol()
    def values(n: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(-50, 50), b: rng.uniform(-50, 50),
                 c: rng.uniform(-50, 50)} for _ in range(n)]
    return MathInvariant(
        expression=f"({a}+{b})+{c}",
        identity_type="symmetry",
        variables=[a, b, c],
        complexity=2,
        description=f"({a}+{b})+{c} (invariant under regrouping — associativity)",
        generate_values=values,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Advanced invariants (calculus, linear algebra, number theory)
# ═══════════════════════════════════════════════════════════════════════════

def _make_calculus_derivative_power() -> MathInvariant:
    """For f(x) = x^n, the ratio f'(x)/n*x^(n-1) = 1."""
    # This is complex — skip for v1, placeholder for expansion
    a, n = _random_symbol(), _random_symbol()
    def values(nc: int, rng: random.Random) -> list[dict[str, float]]:
        return [{a: rng.uniform(0.5, 5), n: float(rng.randint(1, 4))}
                for _ in range(nc)]
    return MathInvariant(
        expression=f"{a}^{n}",
        identity_type="calculus_prep",
        variables=[a, n],
        complexity=5,
        description=f"{a}^{n} — prepare for derivative discovery",
        generate_values=values,
    )


def _make_linear_algebra_det_product() -> MathInvariant:
    """det(A*B) = det(A)*det(B) — 2x2 case."""
    a, b, c, d = _random_symbol(), _random_symbol(), _random_symbol(), _random_symbol()
    e, f, g, h = _random_symbol(), _random_symbol(), _random_symbol(), _random_symbol()
    def values(nc: int, rng: random.Random) -> list[dict[str, float]]:
        vals = []
        for _ in range(nc):
            # Generate 2x2 matrices
            A = [[rng.uniform(-10,10), rng.uniform(-10,10)],
                 [rng.uniform(-10,10), rng.uniform(-10,10)]]
            B = [[rng.uniform(-10,10), rng.uniform(-10,10)],
                 [rng.uniform(-10,10), rng.uniform(-10,10)]]
            # det(A*B) = a*e+b*g * c*f+d*h - (a*f+b*h)*(c*e+d*g)
            # but we provide the individual entries
            vals.append({
                a: A[0][0], b: A[0][1], c: A[1][0], d: A[1][1],
                e: B[0][0], f: B[0][1], g: B[1][0], h: B[1][1],
            })
        return vals
    # The expression det(A*B) is a*d*e*h - a*d*f*g - ... too complex.
    # Provide a simplified version: det(A)*det(B) = constant for fixed matrices
    return MathInvariant(
        expression=f"({a}*{d}-{b}*{c})*({e}*{h}-{f}*{g})",
        identity_type="linear_algebra",
        variables=[a, b, c, d, e, f, g, h],
        complexity=5,
        description="det(A)*det(B) = det(A*B) — determinant multiplication property",
        generate_values=values,
    )


def _make_number_theory_gcd_lcm() -> MathInvariant:
    """gcd(a,b) * lcm(a,b) = a*b."""
    a, b = _random_symbol(), _random_symbol()
    def values(nc: int, rng: random.Random) -> list[dict[str, float]]:
        vals = []
        for _ in range(nc):
            av = float(rng.randint(2, 100))
            bv = float(rng.randint(2, 100))
            g = float(_math.gcd(int(av), int(bv)))
            l = av * bv / g
            vals.append({a: av, b: bv, "gcd_val": g, "lcm_val": l})
        return vals
    return MathInvariant(
        expression=f"gcd_val*lcm_val",
        identity_type="number_theory",
        variables=[a, b],
        complexity=3,
        description="gcd(a,b) * lcm(a,b) = a*b — product invariance",
        generate_values=values,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Generator registry — all invariant types
# ═══════════════════════════════════════════════════════════════════════════

_INVARIANT_BUILDERS: dict[str, list[Callable[[], MathInvariant]]] = {
    "algebraic_commutative":  [_make_algebraic_commutative_add, _make_algebraic_commutative_mul],
    "algebraic_associative":  [_make_algebraic_associative],
    "algebraic_distributive": [_make_algebraic_distributive],
    "algebraic_identity":     [_make_algebraic_identity_add, _make_algebraic_identity_mul],
    "algebraic_inverse":      [_make_algebraic_inverse],
    "algebraic_expansion":    [_make_square_of_sum],
    "algebraic_factorization": [_make_difference_of_squares],
    "algebraic_composition":  [_make_product_ratio, _make_sum_product,
                               _make_power_of_sum],
    "power_rule":             [_make_power_product_rule, _make_power_power_rule],
    "trigonometric":          [_make_trig_pythagorean, _make_trig_tangent_identity, _make_trig_double_angle],
    "exponential":            [_make_exp_product_rule, _make_exp_log_inverse],
    "logarithmic":            [_make_log_product_rule],
    "symmetry":               [_make_symmetry_swap_add, _make_symmetry_rearrangement],
    "calculus_prep":          [_make_calculus_derivative_power],
    "linear_algebra":         [_make_linear_algebra_det_product],
    "number_theory":          [_make_number_theory_gcd_lcm],
}

# Flattened list for random selection, weighted by type
_ALL_BUILDERS: list[tuple[str, Callable[[], MathInvariant]]] = []
for _type, builders in _INVARIANT_BUILDERS.items():
    for b in builders:
        _ALL_BUILDERS.append((_type, b))


# ═══════════════════════════════════════════════════════════════════════════
# Main generator
# ═══════════════════════════════════════════════════════════════════════════

class MathInvariantGenerator:
    """Generate mathematical identities for self-play training.

    Unlike the physics expression generator, this produces pure mathematical
    identities — algebraic laws, trig identities, exponential properties,
    structural symmetries. The system learns mathematical invariance, not
    physics-specific formula patterns.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._generated_count = 0

    def generate(self, type_filter: str | None = None) -> MathInvariant:
        """Generate a single mathematical identity.

        Args:
            type_filter: Optional invariant type to restrict to.
                         None = random type.
        """
        if type_filter and type_filter in _INVARIANT_BUILDERS:
            builder = self._rng.choice(_INVARIANT_BUILDERS[type_filter])
        else:
            _, builder = self._rng.choice(_ALL_BUILDERS)

        reset_symbol_pool(seed=self._generated_count)
        inv = builder()
        self._generated_count += 1
        return inv

    def generate_batch(self, n: int,
                       types: list[str] | None = None) -> list[MathInvariant]:
        """Generate a batch of mathematical identities."""
        results = []
        for _ in range(n):
            if types:
                t = self._rng.choice(types)
                results.append(self.generate(type_filter=t))
            else:
                results.append(self.generate())
        return results

    @property
    def available_types(self) -> list[str]:
        return sorted(_INVARIANT_BUILDERS.keys())

    @property
    def total_builders(self) -> int:
        return len(_ALL_BUILDERS)


# ═══════════════════════════════════════════════════════════════════════════
# Demo / smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    gen = MathInvariantGenerator(seed=42)

    print(f"Available types: {gen.available_types}")
    print(f"Total builders: {gen.total_builders}")
    print()

    for i in range(15):
        inv = gen.generate()
        vals = inv.generate_values(3, random.Random(i))
        print(f"[{inv.identity_type:25s}] cplx={inv.complexity} "
              f"expr={inv.expression:30s} vars={inv.variables}")
        for v in vals[:2]:
            print(f"    values: {v}")
