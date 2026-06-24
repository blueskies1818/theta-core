"""Self-play expression generator with dimension validation.

Generates random physically-valid expressions at 4 complexity levels,
using only pre-1905 quantity symbols. Supports hidden variable injection
for proposer training.

Output: (expression_str, quantities_dict, domain_label, complexity_level)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.physics.dimensions import Dimension, DimensionError
from src.physics.grammar import Expression


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-1905 quantity pool
# ═══════════════════════════════════════════════════════════════════════════════

# Dimension definitions for pre-1905 quantity symbols.
# k = Force/Length (spring constant: F = -k*x)
# R = Energy (gas constant: PV = nRT, where n and T are scalar)
# T, n are scalar (Temperature and amount-of-substance not in MLT system)

_SPRING_CONSTANT_DIM = Dimension.named("Force") / Dimension.named("Length")

PRE_1905_QUANTITY_DIMS: dict[str, Dimension] = {
    "m": Dimension.named("Mass"),
    "g": Dimension.named("Accel"),
    "h": Dimension.named("Length"),
    "v": Dimension.named("Velocity"),
    "t": Dimension.named("Time"),
    "k": _SPRING_CONSTANT_DIM,
    "x": Dimension.named("Length"),
    "E": Dimension.named("Energy"),
    "P": Dimension.named("Pressure"),
    "V": Dimension.named("Volume"),
    "T": Dimension.scalar(),   # Temperature — scalar in simplified MLT system
    "n": Dimension.scalar(),   # amount of substance — scalar
    "R": Dimension.named("Energy"),  # gas constant
}

# Group quantities by dimension for same-dimension operations (Levels 2-3)
_QUANTITIES_BY_DIM: dict[Dimension, list[str]] = {}
for _q, _d in PRE_1905_QUANTITY_DIMS.items():
    _QUANTITIES_BY_DIM.setdefault(_d, []).append(_q)


# ═══════════════════════════════════════════════════════════════════════════════
# Post-1905 symbols (must NEVER appear in generated expressions)
# ═══════════════════════════════════════════════════════════════════════════════

POST_1905_SYMBOLS: set[str] = {
    "hbar", "c", "gamma", "p",  # p as relativistic momentum
    "lambda", "omega", "tau", "mu", "sigma", "pi",
    "alpha", "beta", "delta", "epsilon", "nu", "rho",
    "theta", "phi", "psi", "eta", "xi", "zeta", "kappa",
    "chi", "iota", "upsilon", "omicron",
}

# Extra guard: symbols that could leak via hidden variable names
_POST_1905_HIDDEN_GUARD: set[str] = {
    "hbar", "c", "gamma", "lambda", "omega", "tau",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Hidden variable types
# ═══════════════════════════════════════════════════════════════════════════════

HIDDEN_VAR_TYPES: list[str] = [
    "integer_n",      # n = 1, 2, 3, ...
    "half_integer",   # (n + 1/2)
    "squared_n",      # n^2
    "angular_m",      # m = -l, ..., +l
    "spin",           # s = 1/2, 1, ...
    "ratio",          # continuous ratio (dimensionless)
    "metric",         # metric coefficient (±1)
]

# Symbol names used for hidden variables in expressions
HIDDEN_VAR_SYMBOLS: dict[str, str] = {
    "integer_n":     "n_h",
    "half_integer":  "j_h",
    "squared_n":     "n2_h",
    "angular_m":     "m_h",
    "spin":          "s_h",
    "ratio":         "r_h",
    "metric":        "eta_h",
}

# Domain labels
DOMAIN_LABELS: list[str] = [
    "gravity", "spring", "thermal", "em",
    "classical", "mechanics", "gas_law",
]

# Domain key quantities (mirrors composer.py DOMAIN_QUANTITY_KEY, pre-1905 subset)
# Ordered by specificity: more specific domains first
DOMAIN_KEYS: dict[str, set[str]] = {
    "spring":   {"k"},
    "gravity":  {"g"},
    "thermal":  {"P", "V", "T"},
    "gas_law":  {"P", "V", "T", "n", "R"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Constants pool
# ═══════════════════════════════════════════════════════════════════════════════

# Scalar constants usable as powers or coefficients
SCALAR_CONSTANTS: dict[str, Dimension] = {
    "0.5": Dimension.scalar(),
    "2":   Dimension.scalar(),
    "-1":  Dimension.scalar(),
    "-2":  Dimension.scalar(),
    "3":   Dimension.scalar(),
    "4":   Dimension.scalar(),
}

# Allowed power exponents
POWER_EXPONENTS: list[str] = ["-2", "-1", "2"]


# ═══════════════════════════════════════════════════════════════════════════════
# Output data class
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeneratedExpression:
    """Output of the self-play expression generator.

    Attributes:
        expression_str: Parseable expression string (e.g. "m*g*h").
        quantities_dict: Mapping of quantity symbols to their dimensions.
        domain_label: Predicted physics domain.
        complexity_level: 1-4 indicating structural complexity.
        hidden_variables: Dict of hidden variable type → symbol name, or empty.
        ground_truth_expression: The full expression INCLUDING hidden vars
            (hidden vars stripped from expression_str for proposer training).
    """
    expression_str: str
    quantities_dict: dict[str, Dimension]
    domain_label: str
    complexity_level: int
    hidden_variables: dict[str, str] = field(default_factory=dict)
    ground_truth_expression: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Main generator class
# ═══════════════════════════════════════════════════════════════════════════════

class SelfPlayExpressionGenerator:
    """Generate random, dimensionally-valid physics expressions.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.
    include_hidden_vars : bool
        Whether to sometimes inject hidden variables into expressions.
    hidden_var_probability : float
        Probability (0.0–1.0) of including a hidden variable when
        include_hidden_vars is True.
    """

    def __init__(
        self,
        seed: int | None = None,
        include_hidden_vars: bool = True,
        hidden_var_probability: float = 0.3,
    ) -> None:
        self._rng = random.Random(seed)
        self._include_hidden_vars = include_hidden_vars
        self._hidden_var_prob = hidden_var_probability

        # Pre-compute same-dimension groups for efficient lookups
        self._same_dim_groups: dict[Dimension, list[str]] = {}
        for q, d in PRE_1905_QUANTITY_DIMS.items():
            self._same_dim_groups.setdefault(d, []).append(q)

        # Dimensions with at least 2 quantities (needed for sums and diffs)
        self._addable_dims: list[Dimension] = [
            d for d, qs in self._same_dim_groups.items()
            if len(qs) >= 2 and not d.is_scalar()
        ]

    # ── Public API ────────────────────────────────────────────────────────

    def generate(self, level: int | None = None) -> GeneratedExpression:
        """Generate a single expression.

        Args:
            level: Specific complexity level (1-4), or None for random.

        Returns:
            GeneratedExpression with the expression string, quantities,
            domain label, and metadata.
        """
        if level is None:
            level = self._rng.randint(1, 4)

        if level == 1:
            return self._generate_level1()
        elif level == 2:
            return self._generate_level2()
        elif level == 3:
            return self._generate_level3()
        elif level == 4:
            return self._generate_level4()
        else:
            raise ValueError(f"Invalid complexity level: {level}. Must be 1-4.")

    def generate_batch(
        self,
        n: int,
        levels: list[int] | None = None,
    ) -> list[GeneratedExpression]:
        """Generate a batch of expressions.

        Args:
            n: Number of expressions to generate.
            levels: Distribution of levels (e.g., [1,1,2,3] for 50% L1),
                or None for uniform random.

        Returns:
            List of GeneratedExpression objects.
        """
        results: list[GeneratedExpression] = []
        for _ in range(n):
            if levels:
                level = self._rng.choice(levels)
            else:
                level = self._rng.randint(1, 4)
            results.append(self.generate(level))
        return results

    # ── Level 1: Simple ratios, products, powers ──────────────────────────

    def _generate_level1(self) -> GeneratedExpression:
        """Generate a simple 2-3 variable expression.

        Forms: a*b, a/b, a^p, a*b/c, a^p * b, a^p / b^q
        """
        # Pick 2 or 3 variables
        n_vars = self._rng.choice([2, 3])
        vars_chosen = self._rng.sample(list(PRE_1905_QUANTITY_DIMS.keys()), n_vars)

        # Choose an operation pattern
        pattern = self._rng.choice(["product", "ratio", "power", "mixed"])

        try:
            if pattern == "product":
                expr_str = self._build_product(vars_chosen)
            elif pattern == "ratio":
                expr_str = self._build_ratio(vars_chosen)
            elif pattern == "power":
                expr_str = self._build_power_chain(vars_chosen)
            else:  # mixed
                expr_str = self._build_mixed(vars_chosen)
        except DimensionError:
            # Fallback: try a simple product (always dimensionally valid)
            expr_str = self._build_product(vars_chosen[:2])

        # Validate: expression must be parseable (non-empty, valid chars)
        self._validate_expression_str(expr_str)

        quantities = {q: PRE_1905_QUANTITY_DIMS[q] for q in vars_chosen}
        domain = self._assign_domain(vars_chosen)

        result = GeneratedExpression(
            expression_str=expr_str,
            quantities_dict=quantities,
            domain_label=domain,
            complexity_level=1,
            ground_truth_expression=expr_str,
        )

        # Optionally inject hidden variable
        if self._include_hidden_vars and self._rng.random() < self._hidden_var_prob:
            result = self._inject_hidden_variable(result)

        return result

    # ── Level 2: 2-term sums ──────────────────────────────────────────────

    def _generate_level2(self) -> GeneratedExpression:
        """Generate a 2-term sum of same-dimension expressions.

        Forms: term1 + term2 where both have identical dimensions.
        """
        # Pick a dimension that has multiple quantities for variety
        if not self._addable_dims:
            # Fallback: use any two quantities of same dim
            dim = self._rng.choice(list(self._same_dim_groups.keys()))
        else:
            dim = self._rng.choice(self._addable_dims)

        candidates = self._same_dim_groups[dim]
        if len(candidates) < 2:
            # Need at least 2 quantities of same dimension
            # Build compound expressions of same dimension instead
            return self._generate_level2_compound()

        # Build two sub-expressions of the same dimension
        term1_str, term1_vars = self._build_same_dim_term(dim)
        # Exclude the quantity used in term1 to avoid trivial a+0.5*a
        exclude = term1_vars[0] if len(term1_vars) == 1 else None
        term2_str, term2_vars = self._build_same_dim_term(dim, exclude_q=exclude)

        # Ensure terms are different
        attempts = 0
        while term1_str == term2_str and attempts < 10:
            term2_str, term2_vars = self._build_same_dim_term(dim)
            attempts += 1

        expr_str = f"{term1_str}+{term2_str}"

        # Validate: check that + is dimensionally valid
        self._validate_sum(term1_str, term2_str)

        all_vars = list(set(term1_vars + term2_vars))
        quantities = {q: PRE_1905_QUANTITY_DIMS[q] for q in all_vars}
        domain = self._assign_domain(all_vars)

        result = GeneratedExpression(
            expression_str=expr_str,
            quantities_dict=quantities,
            domain_label=domain,
            complexity_level=2,
            ground_truth_expression=expr_str,
        )

        if self._include_hidden_vars and self._rng.random() < self._hidden_var_prob:
            result = self._inject_hidden_variable(result)

        return result

    def _generate_level2_compound(self) -> GeneratedExpression:
        """Fallback: build same-dimension terms via products/ratios when
        no two quantities share a dimension naturally."""
        # Pick any two quantities and build compound expressions of matching dims
        all_qs = list(PRE_1905_QUANTITY_DIMS.keys())
        q1, q2 = self._rng.sample(all_qs, 2)

        # Make both produce Energy (or any shared dimension)
        # Strategy: multiply by appropriate companions to reach same target dim
        target_dim = Dimension.named("Energy")

        # Build term1: q1 * something → Energy
        term1_str, term1_qs = self._build_to_target_dim(q1, target_dim)
        term2_str, term2_qs = self._build_to_target_dim(q2, target_dim)

        expr_str = f"{term1_str}+{term2_str}"
        all_vars = list(set(term1_qs + term2_qs))
        quantities = {q: PRE_1905_QUANTITY_DIMS[q] for q in all_vars}
        domain = self._assign_domain(all_vars)

        return GeneratedExpression(
            expression_str=expr_str,
            quantities_dict=quantities,
            domain_label=domain,
            complexity_level=2,
            ground_truth_expression=expr_str,
        )

    # ── Level 3: Squared differences ──────────────────────────────────────

    def _generate_level3(self) -> GeneratedExpression:
        """Generate a squared-difference expression: q1^2 - q2^2.

        Both q1 and q2 must have the same dimension.
        """
        # Pick two quantities of the same dimension
        same_dim_candidates: list[tuple[str, str]] = []
        for dim, qs in self._same_dim_groups.items():
            if len(qs) >= 2:
                for i in range(len(qs)):
                    for j in range(i + 1, len(qs)):
                        same_dim_candidates.append((qs[i], qs[j]))

        if not same_dim_candidates:
            # Fallback: use two different quantities and make compound
            # expressions that produce same dimension when squared
            return self._generate_level3_compound()

        q1, q2 = self._rng.choice(same_dim_candidates)

        # Optionally add scalar coefficients
        use_coeff = self._rng.random() < 0.4
        if use_coeff:
            coeff1 = self._rng.choice(["0.5", "2"])
            coeff2 = self._rng.choice(["0.5", "2"])
            expr_str = f"{coeff1}*{q1}^2-{coeff2}*{q2}^2"
        else:
            expr_str = f"{q1}^2-{q2}^2"

        all_vars = [q1, q2]
        quantities = {q: PRE_1905_QUANTITY_DIMS[q] for q in all_vars}
        domain = self._assign_domain(all_vars)

        # Validate dimensionally
        self._validate_squared_diff(expr_str, q1, q2)

        result = GeneratedExpression(
            expression_str=expr_str,
            quantities_dict=quantities,
            domain_label=domain,
            complexity_level=3,
            ground_truth_expression=expr_str,
        )

        if self._include_hidden_vars and self._rng.random() < self._hidden_var_prob:
            result = self._inject_hidden_variable(result)

        return result

    def _generate_level3_compound(self) -> GeneratedExpression:
        """Fallback: build compound squared-difference when no two quantities
        share a dimension."""
        # Use two different quantities, scale to same dimension via companions
        all_qs = list(PRE_1905_QUANTITY_DIMS.keys())
        q1, q2 = self._rng.sample(all_qs, 2)

        target_dim = Dimension.named("Energy")
        term1_str, term1_qs = self._build_to_target_dim(q1, target_dim)
        term2_str, term2_qs = self._build_to_target_dim(q2, target_dim)

        expr_str = f"({term1_str})^2-({term2_str})^2"
        all_vars = list(set(term1_qs + term2_qs))
        quantities = {q: PRE_1905_QUANTITY_DIMS[q] for q in all_vars}
        domain = self._assign_domain(all_vars)

        return GeneratedExpression(
            expression_str=expr_str,
            quantities_dict=quantities,
            domain_label=domain,
            complexity_level=3,
            ground_truth_expression=expr_str,
        )

    # ── Level 4: Nested expressions ───────────────────────────────────────

    def _generate_level4(self) -> GeneratedExpression:
        """Generate a nested expression with parentheses.

        Forms: (a*b)/(c+d), (a+b)*(c+d), a/(b+c/d), etc.
        """
        pattern = self._rng.choice([
            "frac_of_sum",     # (a*b)/(c+d)
            "prod_of_sums",    # (a+b)*(c+d)
            "nested_frac",     # a/(b+c/d)
            "power_of_sum",    # (a+b)^2
        ])

        try:
            if pattern == "frac_of_sum":
                expr_str, all_vars = self._build_frac_of_sum()
            elif pattern == "prod_of_sums":
                expr_str, all_vars = self._build_prod_of_sums()
            elif pattern == "nested_frac":
                expr_str, all_vars = self._build_nested_frac()
            else:  # power_of_sum
                expr_str, all_vars = self._build_power_of_sum()
        except DimensionError:
            expr_str, all_vars = self._build_frac_of_sum()

        quantities = {q: PRE_1905_QUANTITY_DIMS[q] for q in all_vars}
        domain = self._assign_domain(all_vars)

        result = GeneratedExpression(
            expression_str=expr_str,
            quantities_dict=quantities,
            domain_label=domain,
            complexity_level=4,
            ground_truth_expression=expr_str,
        )

        if self._include_hidden_vars and self._rng.random() < self._hidden_var_prob:
            result = self._inject_hidden_variable(result)

        return result

    # ── Building helpers ──────────────────────────────────────────────────

    def _build_product(self, vars_chosen: list[str]) -> str:
        """Build product: a*b or a*b*c."""
        return "*".join(vars_chosen)

    def _build_ratio(self, vars_chosen: list[str]) -> str:
        """Build ratio: a/b or (a*b)/c."""
        if len(vars_chosen) == 2:
            return f"{vars_chosen[0]}/{vars_chosen[1]}"
        else:
            # 3 vars: pick numerator and denominator groups
            num_vars = vars_chosen[:2]
            den_var = vars_chosen[2]
            num_str = "*".join(num_vars)
            return f"({num_str})/{den_var}"

    def _build_power_chain(self, vars_chosen: list[str]) -> str:
        """Build power chain: a^2, a^-1, a^2*b, etc."""
        exp = self._rng.choice(POWER_EXPONENTS)
        base = vars_chosen[0]

        # Check: can only raise scalar or dimensionless to non-trivial powers
        # safely? Actually the grammar handles this: a^2 for any dimension is fine
        # as long as the exponent is scalar. Since our exponents are scalar
        # constants, this is always valid.

        if len(vars_chosen) == 1:
            return f"{base}^{exp}"

        # Multi-variable: e.g., a^2 * b
        parts: list[str] = []
        for i, q in enumerate(vars_chosen):
            if i == 0:
                parts.append(f"{q}^{exp}")
            else:
                parts.append(q)
        return "*".join(parts)

    def _build_mixed(self, vars_chosen: list[str]) -> str:
        """Build mixed: a^p / b or a^p * b / c."""
        if len(vars_chosen) == 2:
            exp = self._rng.choice(POWER_EXPONENTS)
            return f"{vars_chosen[0]}^{exp}/{vars_chosen[1]}"
        else:
            exp = self._rng.choice(POWER_EXPONENTS)
            return f"{vars_chosen[0]}^{exp}*{vars_chosen[1]}/{vars_chosen[2]}"

    def _build_same_dim_term(
        self, dim: Dimension, exclude_q: str | None = None,
    ) -> tuple[str, list[str]]:
        """Build a sub-expression of the given dimension.

        Returns (expression_str, list_of_variables_used).
        """
        candidates = self._same_dim_groups.get(dim, [])
        usable = [q for q in candidates if q != exclude_q]

        if len(usable) >= 1:
            # Use a single quantity (or product with scalar)
            q = self._rng.choice(usable)
            if self._rng.random() < 0.3:
                coeff = self._rng.choice(["0.5", "2"])
                return f"{coeff}*{q}", [q]
            return q, [q]

        # Build compound expression to reach target dim
        # Pick a random quantity (different from exclude_q) and multiply/divide
        all_qs = [q for q in PRE_1905_QUANTITY_DIMS if q != exclude_q]
        if not all_qs:
            all_qs = list(PRE_1905_QUANTITY_DIMS.keys())
        q = self._rng.choice(all_qs)
        return self._build_to_target_dim(q, dim)

    def _build_to_target_dim(
        self, start_q: str, target_dim: Dimension,
    ) -> tuple[str, list[str]]:
        """Build an expression starting from start_q that reaches target_dim.

        Uses multiplication/division by other quantities.
        """
        start_dim = PRE_1905_QUANTITY_DIMS[start_q]
        if start_dim == target_dim:
            return start_q, [start_q]

        # Compute what dimension we need to multiply by
        # start_dim * needed = target_dim → needed = target_dim / start_dim
        needed_dim = target_dim.__truediv__(start_dim)

        # Find a quantity (or compound) with the needed dimension
        for q, d in PRE_1905_QUANTITY_DIMS.items():
            if q == start_q:
                continue
            if d == needed_dim:
                return f"{start_q}*{q}", [start_q, q]

        # Try product of two quantities
        all_qs = [q for q in PRE_1905_QUANTITY_DIMS if q != start_q]
        for i, qa in enumerate(all_qs):
            for qb in all_qs[i + 1:]:
                combo_dim = PRE_1905_QUANTITY_DIMS[qa] * PRE_1905_QUANTITY_DIMS[qb]
                if combo_dim == needed_dim:
                    return f"{start_q}*{qa}*{qb}", [start_q, qa, qb]

        # Try ratio
        for qa in all_qs:
            for qb in all_qs:
                if qa == qb:
                    continue
                combo_dim = PRE_1905_QUANTITY_DIMS[qa] / PRE_1905_QUANTITY_DIMS[qb]
                if combo_dim == needed_dim:
                    return f"{start_q}*{qa}/{qb}", [start_q, qa, qb]

        # Ultimate fallback: use energy as universal target
        # Everything can be made energy somehow
        return f"{start_q}*{start_q}", [start_q]

    # ── Level 4 building helpers ──────────────────────────────────────────

    def _build_frac_of_sum(self) -> tuple[str, list[str]]:
        """Build (a*b)/(c+d) where numerator and denominator dimensions differ."""
        # Find two same-dimension quantities for denominator
        denom_dim = self._rng.choice(self._addable_dims) if self._addable_dims else \
            self._rng.choice(list(self._same_dim_groups.keys()))
        denom_qs = self._same_dim_groups[denom_dim]
        c = self._rng.choice(denom_qs)
        d = self._rng.choice([q for q in denom_qs if q != c]) if len(denom_qs) > 1 else c

        # Build numerator: must have same dimension as denominator
        a, b = self._rng.sample(list(PRE_1905_QUANTITY_DIMS.keys()), 2)
        num_dim = PRE_1905_QUANTITY_DIMS[a] * PRE_1905_QUANTITY_DIMS[b]

        # Check: numerator dim and denominator dim don't need to match for /
        # Division always works dimensionally
        expr_str = f"({a}*{b})/({c}+{d})"
        return expr_str, [a, b, c, d]

    def _build_prod_of_sums(self) -> tuple[str, list[str]]:
        """Build (a+b)*(c+d) where both sums are dimensionally valid."""
        # Find two pairs of same-dimension quantities
        dims_with_pairs = [
            d for d, qs in self._same_dim_groups.items()
            if len(qs) >= 2 and not d.is_scalar()
        ]
        if len(dims_with_pairs) < 2:
            # Fall back to frac_of_sum
            return self._build_frac_of_sum()

        dim1, dim2 = self._rng.sample(dims_with_pairs, 2)
        qs1 = self._same_dim_groups[dim1]
        qs2 = self._same_dim_groups[dim2]

        a, b = self._rng.sample(qs1, 2)
        c, d = self._rng.sample(qs2, 2)

        expr_str = f"({a}+{b})*({c}+{d})"
        return expr_str, [a, b, c, d]

    def _build_nested_frac(self) -> tuple[str, list[str]]:
        """Build a/(b+c/d) type nested fraction, ensuring dimension validity."""
        all_qs = list(PRE_1905_QUANTITY_DIMS.keys())
        for _ in range(50):
            a, b, c, d = self._rng.sample(all_qs, 4)
            # b and c/d must have same dimension for the sum
            dim_b = PRE_1905_QUANTITY_DIMS[b]
            dim_c_over_d = PRE_1905_QUANTITY_DIMS[c] / PRE_1905_QUANTITY_DIMS[d]
            if dim_b == dim_c_over_d:
                expr_str = f"{a}/({b}+{c}/{d})"
                return expr_str, [a, b, c, d]
        # Fallback: build with guaranteed compatible types
        # Use same-dimension pairs for inner sum
        for dim, qs in self._same_dim_groups.items():
            if len(qs) >= 2:
                b = qs[0]
                c, d = self._rng.sample(list(all_qs), 2)
                if (PRE_1905_QUANTITY_DIMS[c] / PRE_1905_QUANTITY_DIMS[d]) == dim:
                    a = self._rng.choice([q for q in all_qs if q not in (b, c, d)])
                    return f"{a}/({b}+{c}/{d})", [a, b, c, d]
        # Ultimate fallback: use two same-dimension quantities for inner sum
        # h and x are both Length, so h+x is valid
        a = self._rng.choice(all_qs)
        return f"{a}/(h+x)", [a, "h", "x"]

    def _build_power_of_sum(self) -> tuple[str, list[str]]:
        """Build (a+b)^2 where a and b have same dimension."""
        dim = self._rng.choice(self._addable_dims) if self._addable_dims else \
            self._rng.choice(list(self._same_dim_groups.keys()))
        qs = self._same_dim_groups[dim]
        if len(qs) >= 2:
            a, b = self._rng.sample(qs, 2)
        else:
            a = b = qs[0]

        expr_str = f"({a}+{b})^2"
        return expr_str, [a, b]

    # ── Hidden variable injection ─────────────────────────────────────────

    def _inject_hidden_variable(
        self, expr: GeneratedExpression,
    ) -> GeneratedExpression:
        """Inject a hidden variable into the expression.

        The hidden variable appears in the ground_truth_expression but is
        stripped from expression_str (what the proposer sees).
        """
        hv_type = self._rng.choice(HIDDEN_VAR_TYPES)
        hv_symbol = HIDDEN_VAR_SYMBOLS[hv_type]

        # Hidden variables are scalar, so they can multiply any expression
        # without changing its dimension.
        # Inject as a multiplier: expr * hv or hv * expr
        pos = self._rng.choice(["prefix", "suffix", "factor"])

        original_str = expr.expression_str
        if pos == "prefix":
            gt_expr = f"{hv_symbol}*{original_str}"
        elif pos == "suffix":
            gt_expr = f"{original_str}*{hv_symbol}"
        else:  # factor — multiply into a specific term
            gt_expr = f"{hv_symbol}*({original_str})"

        # The expression_str is what the proposer sees — hidden var REMOVED
        # The ground_truth_expression has the full expression
        hv_dict = {hv_type: hv_symbol}

        return GeneratedExpression(
            expression_str=original_str,  # proposer sees NO hidden var
            quantities_dict=dict(expr.quantities_dict),
            domain_label=expr.domain_label,
            complexity_level=expr.complexity_level,
            hidden_variables=hv_dict,
            ground_truth_expression=gt_expr,
        )

    # ── Domain assignment ─────────────────────────────────────────────────

    def _assign_domain(self, vars_used: list[str]) -> str:
        """Assign a domain label based on which quantities are present."""
        var_set = set(vars_used)

        # Check domain key quantities. gas_law requires 2+ keys to avoid
        # labeling single-scalar appearances (e.g. just 'n') as gas_law.
        for domain, keys in DOMAIN_KEYS.items():
            matched = keys & var_set
            if domain == "gas_law":
                if len(matched) >= 2:
                    return domain
            elif matched:
                return domain

        # Default domain based on quantity presence
        if "E" in var_set or "P" in var_set:
            return "mechanics"
        return "classical"

    # ── Validation ────────────────────────────────────────────────────────

    def _validate_expression_str(self, expr_str: str) -> None:
        """Validate that expression string is well-formed.

        Raises ValueError if the expression contains post-1905 symbols
        or is otherwise invalid.
        """
        if not expr_str or not expr_str.strip():
            raise ValueError("Empty expression string")

        # Check for post-1905 symbols
        tokens = set(expr_str.replace("(", " ").replace(")", " ")
                     .replace("+", " ").replace("-", " ")
                     .replace("*", " ").replace("/", " ")
                     .replace("^", " ").split())
        for token in tokens:
            # Skip numbers
            try:
                float(token)
                continue
            except ValueError:
                pass
            if token in POST_1905_SYMBOLS:
                raise ValueError(
                    f"Post-1905 symbol {token!r} leaked into expression {expr_str!r}"
                )

    def _validate_sum(self, term1_str: str, term2_str: str) -> None:
        """Validate that two terms can be added (same dimension)."""
        # Build Expression objects to verify
        try:
            expr1 = _parse_to_expression(term1_str)
            expr2 = _parse_to_expression(term2_str)
            expr1.dim.check_add(expr2.dim)
        except DimensionError as e:
            raise DimensionError(
                f"Invalid sum: {term1_str}+{term2_str}: {e}"
            ) from e

    def _validate_squared_diff(
        self, expr_str: str, q1: str, q2: str,
    ) -> None:
        """Validate squared-diff expression dimension."""
        dim1 = PRE_1905_QUANTITY_DIMS[q1]
        dim2 = PRE_1905_QUANTITY_DIMS[q2]

        # Both need same dimension for subtraction
        if dim1 != dim2:
            raise DimensionError(
                f"Cannot subtract different dimensions in {expr_str}: "
                f"{dim1} vs {dim2}"
            )

    def _validate_no_post_1905(self, expr_str: str) -> None:
        """Ensure no post-1905 symbols appear in the expression string."""
        for sym in _POST_1905_HIDDEN_GUARD:
            if sym in expr_str:
                raise ValueError(
                    f"Post-1905 symbol {sym!r} found in expression {expr_str!r}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Mini-parser for validation (avoids full evaluator dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_to_expression(expr_str: str) -> Expression:
    """Parse a simple expression string into an Expression tree for validation.

    This is a simplified parser that handles products, ratios, and powers
    for dimension-checking purposes.
    """
    # Strip whitespace
    expr_str = expr_str.strip()

    # Handle parenthesized expressions — only if the outer parens match
    if expr_str.startswith("(") and expr_str.endswith(")"):
        # Verify the opening ( and closing ) are matching
        depth = 0
        matched = True
        for i, ch in enumerate(expr_str):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i < len(expr_str) - 1:
                    matched = False
                    break
        if matched and depth == 0:
            return _parse_to_expression(expr_str[1:-1])

    # Try to split on + (lowest precedence, but need to handle parens)
    parts = _split_at_top_level(expr_str, "+")
    if len(parts) >= 2:
        left = _parse_to_expression(parts[0])
        right = _parse_to_expression("+".join(parts[1:]))
        return Expression.build("+", left, right)

    # Split on -
    parts = _split_at_top_level(expr_str, "-")
    if len(parts) >= 2:
        left = _parse_to_expression(parts[0])
        right = _parse_to_expression("-".join(parts[1:]))
        return Expression.build("-", left, right)

    # Split on *
    parts = _split_at_top_level(expr_str, "*")
    if len(parts) >= 2:
        left = _parse_to_expression(parts[0])
        right = _parse_to_expression("*".join(parts[1:]))
        return Expression.build("*", left, right)

    # Split on /
    parts = _split_at_top_level(expr_str, "/")
    if len(parts) >= 2:
        left = _parse_to_expression(parts[0])
        right = _parse_to_expression("/".join(parts[1:]))
        return Expression.build("/", left, right)

    # Split on ^
    if "^" in expr_str:
        base_str, exp_str = expr_str.rsplit("^", 1)
        base = _parse_to_expression(base_str)
        exp = _parse_to_expression(exp_str)
        return Expression.build("^", base, exp)

    # Leaf node: quantity or constant
    expr_str = expr_str.strip()
    if expr_str in PRE_1905_QUANTITY_DIMS:
        return Expression.quantity(expr_str, PRE_1905_QUANTITY_DIMS[expr_str])
    if expr_str in SCALAR_CONSTANTS:
        return Expression.constant(expr_str, SCALAR_CONSTANTS[expr_str])

    # Try parsing as a number
    try:
        _ = float(expr_str)
        return Expression.constant(expr_str, Dimension.scalar())
    except ValueError:
        pass

    # Handle negative numbers like "-1", "-2"
    if expr_str.startswith("-"):
        try:
            _ = float(expr_str)
            return Expression.constant(expr_str, Dimension.scalar())
        except ValueError:
            pass

    raise ValueError(f"Cannot parse leaf: {expr_str!r}")


def _split_at_top_level(expr_str: str, op: str) -> list[str]:
    """Split expression on operator, respecting parentheses.
    Handles leading/trailing operators gracefully (e.g. m^-2)."""
    parts: list[str] = []
    depth = 0
    current = ""
    i = 0
    prev_char = ""
    while i < len(expr_str):
        ch = expr_str[i]
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == op and depth == 0:
            # Don't split on '-' if it's part of '^-' (power of negative)
            if op == "-" and prev_char == "^":
                current += ch
            elif current.strip():
                parts.append(current)
                current = ""
            else:
                # Leading operator (e.g., "-2" at start)
                current += ch
            i += 1
            prev_char = ch
            continue
        else:
            current += ch
        prev_char = ch
        i += 1

    if current.strip():
        parts.append(current)
    elif parts and not parts[-1].endswith(op):
        # Trailing operator: merge with last part only if not already there
        parts[-1] = parts[-1] + op

    # Handle leading operator only case: return as-is
    if not parts:
        return [expr_str]

    return parts


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════════

def generate_single(
    level: int | None = None,
    seed: int | None = None,
    include_hidden: bool = True,
) -> GeneratedExpression:
    """Generate a single self-play expression.

    Args:
        level: Complexity level 1-4, or None for random.
        seed: Random seed for reproducibility.
        include_hidden: Whether to allow hidden variable injection.

    Returns:
        GeneratedExpression.
    """
    gen = SelfPlayExpressionGenerator(
        seed=seed,
        include_hidden_vars=include_hidden,
    )
    return gen.generate(level)


def generate_batch(
    n: int = 10,
    levels: list[int] | None = None,
    seed: int | None = None,
    include_hidden: bool = True,
) -> list[GeneratedExpression]:
    """Generate a batch of self-play expressions.

    Args:
        n: Number of expressions to generate.
        levels: Distribution of complexity levels.
        seed: Random seed.
        include_hidden: Whether to allow hidden variable injection.

    Returns:
        List of GeneratedExpression.
    """
    gen = SelfPlayExpressionGenerator(
        seed=seed,
        include_hidden_vars=include_hidden,
    )
    return gen.generate_batch(n, levels)
