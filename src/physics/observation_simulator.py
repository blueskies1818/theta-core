"""Observation simulator for self-play physics discovery.

Converts generated expressions into noisy observation data by sampling
random input values from physically-reasonable ranges, computing ground-truth
expression values, and applying independent Gaussian measurement noise.

Phase B of the self-play architecture: GENERATE → SIMULATE → CHALLENGE → SCORE.

Design: expressions are conserved quantities.  To produce configurations where
the expression is genuinely constant, one quantity is designated as "dependent"
— sampled independently, all others are solved for numerically to maintain
constant expression value.  Noise is then applied independently to all
measurements.
"""

from __future__ import annotations

import math
import random
import uuid
from typing import Any

from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import Observation


# ═══════════════════════════════════════════════════════════════════════════════
# Physical ranges for named dimension types
# ═══════════════════════════════════════════════════════════════════════════════

# Base dimension ranges (used for computing compound dimension ranges)
_BASE_RANGES: dict[str, tuple[float, float]] = {
    "Mass":   (0.1,    1000.0),
    "Length": (0.01,   1000.0),
    "Time":   (0.001,  1000.0),
}

# Named dimension ranges — directly defined for known dimension types.
_NAMED_RANGES: dict[str, tuple[float, float]] = {
    "Scalar":   (0.1,   10.0),
    "Mass":     (0.1,   1000.0),
    "Length":   (0.01,  1000.0),
    "Time":     (0.001, 1000.0),
    "Velocity": (0.1,   300.0),
    "Accel":    (0.1,   100.0),
    "Force":    (1.0,   1e6),
    "Momentum": (0.1,   1e5),
    "Energy":   (0.001, 1e6),
    "Pressure": (1.0,   1e6),
    "Volume":   (0.0001, 1000.0),
}


def _named_range_for_dimension(dim: Dimension) -> tuple[float, float]:
    """Return the physical range (min, max) for a given Dimension."""
    for name, named_dim in _NAMED_DIM_SINGLETONS.items():
        if dim == named_dim and name in _NAMED_RANGES:
            return _NAMED_RANGES[name]

    # Compound dimension: compute from base exponents.
    exponents: dict[str, float] = getattr(dim, '_exp', {})
    lo, hi = 1.0, 1.0
    for base, exp in exponents.items():
        base_lo, base_hi = _BASE_RANGES.get(base, (0.1, 10.0))
        if exp > 0:
            lo *= base_lo ** exp
            hi *= base_hi ** exp
        elif exp < 0:
            lo *= base_hi ** exp
            hi *= base_lo ** exp
    return (max(lo, 1e-12), min(hi, 1e12))


_NAMED_DIM_SINGLETONS: dict[str, Dimension] = {}
for _name in _NAMED_RANGES:
    try:
        _NAMED_DIM_SINGLETONS[_name] = Dimension.named(_name)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Quantity dimension resolution
# ═══════════════════════════════════════════════════════════════════════════════

_STANDARD_QUANTITY_DIMS: dict[str, Dimension] = {}
for _qname, _dim in [
    ("m", Dimension.named("Mass")),
    ("g", Dimension.named("Accel")),
    ("h", Dimension.named("Length")),
    ("v", Dimension.named("Velocity")),
    ("t", Dimension.named("Time")),
    ("x", Dimension.named("Length")),
    ("E", Dimension.named("Energy")),
    ("P", Dimension.named("Pressure")),
    ("V", Dimension.named("Volume")),
    ("T", Dimension.scalar()),
    ("n", Dimension.scalar()),
    ("R", Dimension.named("Energy")),
]:
    _STANDARD_QUANTITY_DIMS[_qname] = _dim

_SPRING_CONSTANT_DIM = Dimension.named("Force") / Dimension.named("Length")
_STANDARD_QUANTITY_DIMS["k"] = _SPRING_CONSTANT_DIM


def _resolve_quantity_dimension(
    qname: str,
    quantity_dims: dict[str, Dimension] | None = None,
) -> Dimension:
    """Resolve a quantity name to its Dimension."""
    if quantity_dims and qname in quantity_dims:
        return quantity_dims[qname]
    if qname in _STANDARD_QUANTITY_DIMS:
        return _STANDARD_QUANTITY_DIMS[qname]
    return Dimension.scalar()


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: sample value within a range
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_value(lo: float, hi: float, rng: random.Random) -> float:
    """Sample a value from [lo, hi], using log-uniform for wide ranges."""
    if hi / lo > 1000:
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    return rng.uniform(lo, hi)


# ═══════════════════════════════════════════════════════════════════════════════
# Core simulation function
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_observations(
    expression: str,
    quantities: dict[str, str],
    num_configs: int = 20,
    noise_frac: float = 0.03,
    seed: int | None = None,
) -> list[Observation]:
    """Convert a generated expression into noisy observation data.

    The expression is treated as a conserved quantity.  One variable is
    designated as *dependent* and is solved for numerically so the
    expression stays constant across all configurations.  All other
    variables are sampled independently from their physical ranges.

    Parameters
    ----------
    expression : str
        The ground-truth expression (e.g. ``\"m*g*h + 0.5*m*v^2\"``).
    quantities : dict[str, str]
        Mapping from quantity name to dimension name (e.g. ``{\"m\": \"Mass\", \"v\": \"Velocity\"}``).
    num_configs : int
        Number of random configurations (timesteps) to generate.
    noise_frac : float
        Relative noise standard deviation (e.g. 0.03 = 3% Gaussian noise).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    list[Observation]
        A single-element list containing one Observation with ``num_configs``
        noisy timesteps.  The ground-truth expression is stored in
        ``known_invariant``.

    Raises
    ------
    ValueError
        If validation fails (no variation, poor constancy, bad values, or
        dependent variable cannot be solved for within range).
    """
    if num_configs < 2:
        raise ValueError(f"Need at least 2 configurations, got {num_configs}")
    if noise_frac < 0:
        raise ValueError(f"noise_frac must be >= 0, got {noise_frac}")

    rng = random.Random(seed)
    evaluator = ExpressionEvaluator()

    # ── Resolve quantity dimensions ───────────────────────────────────────
    quantity_dim_objects: dict[str, Dimension] = {}
    for qname, dim_name in quantities.items():
        try:
            quantity_dim_objects[qname] = Dimension.named(dim_name)
        except Exception:
            quantity_dim_objects[qname] = Dimension.scalar()

    # ── Extract variables from expression ─────────────────────────────────
    try:
        ast = evaluator.parse(expression)
    except Exception as e:
        raise ValueError(f"Cannot parse expression {expression!r}: {e}") from e

    from src.physics.evaluator import _collect_var_names
    var_names = _collect_var_names(ast)

    # All variables that appear in expression or quantities
    all_vars: list[str] = sorted(set(var_names) | set(quantities.keys()))
    if len(all_vars) < 1:
        raise ValueError("Expression has no variables")

    # ── Compute ranges for each variable ──────────────────────────────────
    var_ranges: dict[str, tuple[float, float]] = {}
    for vname in all_vars:
        dim = _resolve_quantity_dimension(vname, quantity_dim_objects)
        var_ranges[vname] = _named_range_for_dimension(dim)

    # ── Designate dependent variable ──────────────────────────────────────
    # Choose the LAST variable in sorted order as dependent.
    # This guarantees at least one independent variable for variation.
    if len(all_vars) == 1:
        # Single-variable expression (e.g., "v^2"): expression can't be
        # constant while the variable varies.  We vary the variable and
        # accept that the expression value changes — best we can do.
        dependent_var = None
        independent_vars = list(all_vars)
    else:
        dependent_var = all_vars[-1]
        independent_vars = all_vars[:-1]

    # ── Seed configuration: all variables at mid-range ────────────────────
    seed_cfg: dict[str, float] = {}
    for vname in all_vars:
        lo, hi = var_ranges[vname]
        seed_cfg[vname] = (lo + hi) / 2.0

    # Compute constant expression value C from seed configuration
    try:
        C = float(evaluator.evaluate(expression, seed_cfg))
    except Exception as e:
        raise ValueError(
            f"Cannot evaluate {expression!r} at seed config: {e}"
        ) from e

    if abs(C) < 1e-300:
        # Expression evaluates to near-zero at seed — shift seed slightly
        for vname in all_vars:
            lo, hi = var_ranges[vname]
            seed_cfg[vname] = (lo + 3 * hi) / 4.0  # shift toward upper range
        try:
            C = float(evaluator.evaluate(expression, seed_cfg))
        except Exception as e:
            raise ValueError(
                f"Cannot evaluate {expression!r} after seed shift: {e}"
            ) from e

    # ── Generate configurations with constant expression value ────────────
    configs: list[dict[str, float]] = []
    true_expr_values: list[float] = []

    max_attempts_per_config = 200

    for _ in range(num_configs):
        for attempt in range(max_attempts_per_config):
            cfg: dict[str, float] = {}

            # Sample independent variables
            for vname in independent_vars:
                lo, hi = var_ranges[vname]
                cfg[vname] = _sample_value(lo, hi, rng)

            if dependent_var is None:
                # Single-variable case: just sample it
                lo, hi = var_ranges[all_vars[0]]
                cfg[all_vars[0]] = _sample_value(lo, hi, rng)
                configs.append(dict(cfg))
                try:
                    true_expr_values.append(
                        float(evaluator.evaluate(expression, cfg))
                    )
                except Exception:
                    continue
                break

            # Solve for dependent variable numerically
            dep_lo, dep_hi = var_ranges[dependent_var]
            # Initial guess: mid-range
            dep_val = (dep_lo + dep_hi) / 2.0
            cfg[dependent_var] = dep_val

            # Newton iteration to find dep_val such that eval(expr, cfg) ≈ C
            converged = False
            for newton_iter in range(50):
                try:
                    val = float(evaluator.evaluate(expression, cfg))
                except Exception:
                    break

                error = val - C
                if abs(error) < max(1e-10 * abs(C), 1e-12):
                    converged = True
                    break

                # Numeric derivative: dE/d(dep)
                eps = max(abs(dep_val) * 1e-6, 1e-10)
                cfg_pert = dict(cfg)
                cfg_pert[dependent_var] = dep_val + eps
                try:
                    val_pert = float(evaluator.evaluate(expression, cfg_pert))
                except Exception:
                    break
                derivative = (val_pert - val) / eps

                if abs(derivative) < 1e-15:
                    # Flat derivative — can't solve; try different independent values
                    break

                # Newton step
                delta = -error / derivative
                # Damping for stability
                delta = max(min(delta, abs(dep_val) * 0.5), -abs(dep_val) * 0.5)
                dep_val += delta

                # Clamp to valid range
                if dep_val < dep_lo * 0.01 or dep_val > dep_hi * 100:
                    break
                dep_val = max(dep_lo * 0.01, min(dep_hi * 100, dep_val))
                cfg[dependent_var] = dep_val

            if not converged:
                if attempt < max_attempts_per_config - 1:
                    continue  # Try different independent values
                raise ValueError(
                    f"Could not solve for dependent variable {dependent_var!r} "
                    f"to make {expression!r} = {C:.6g} after {max_attempts_per_config} attempts"
                )

            # Final clamp to exact range and verify
            dep_val = max(dep_lo, min(dep_hi, dep_val))
            cfg[dependent_var] = dep_val

            try:
                final_val = float(evaluator.evaluate(expression, cfg))
            except Exception:
                continue

            if abs(final_val - C) > max(1e-8 * abs(C), 1e-8):
                continue  # Not close enough, retry

            configs.append(dict(cfg))
            true_expr_values.append(final_val)
            break
        else:
            raise ValueError(
                f"Could not generate config {len(configs)} for {expression!r} "
                f"after {max_attempts_per_config} attempts"
            )

    # ── Validation on noise-free data ─────────────────────────────────────
    _validate_noise_free(
        expression=expression,
        configs=configs,
        true_expr_values=true_expr_values,
        evaluator=evaluator,
    )

    # ── Apply independent Gaussian noise ──────────────────────────────────
    noisy_timesteps: list[dict[str, float]] = []
    for i, cfg in enumerate(configs):
        noisy_ts: dict[str, float] = {"t": float(i)}
        for vname in all_vars:
            true_val = cfg[vname]
            noise = rng.gauss(0.0, noise_frac) if noise_frac > 0 else 0.0
            noisy_ts[vname] = true_val * (1.0 + noise)

        # Expression value also gets independent noise
        expr_noise = rng.gauss(0.0, noise_frac) if noise_frac > 0 else 0.0
        raw_expr = true_expr_values[i]
        noisy_ts["expr_value"] = raw_expr * (1.0 + expr_noise)
        noisy_ts["expr_true"] = raw_expr  # stored for debugging

        noisy_timesteps.append(noisy_ts)

    # ── Post-noise validation ─────────────────────────────────────────────
    _validate_post_noise(noisy_timesteps, quantities)

    # ── Build Observation ─────────────────────────────────────────────────
    obs_id = f"selfplay_{uuid.uuid4().hex[:12]}"
    obs = Observation(
        id=obs_id,
        name=f"Self-play: {expression}",
        description=(
            f"Generated from expression {expression} with {num_configs} configs, "
            f"noise_frac={noise_frac}"
        ),
        quantities=dict(quantities),
        parameters={},
        timesteps=noisy_timesteps,
        known_invariant=expression,
        lean_theorem="",
    )

    return [obs]


# ═══════════════════════════════════════════════════════════════════════════════
# Validation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_noise_free(
    expression: str,
    configs: list[dict[str, float]],
    true_expr_values: list[float],
    evaluator: ExpressionEvaluator,
) -> None:
    """Validate noise-free data before applying noise.

    Checks:
    1. At least one quantity genuinely varies across configurations.
    2. The ground-truth expression scores >= 0.95 constancy.
    """
    if len(configs) < 2:
        raise ValueError("Need at least 2 configurations to check variation")

    # Check 1: at least one quantity varies.
    any_varies = False
    for vname in configs[0]:
        vals = [cfg[vname] for cfg in configs]
        if max(vals) - min(vals) > 1e-12:
            any_varies = True
            break
    if not any_varies:
        raise ValueError(
            "No quantity varies across configurations — "
            "cannot produce meaningful observations"
        )

    # Check 2: constancy of ground-truth on noise-free data.
    # Compute constancy directly from true_expr_values.
    n = len(true_expr_values)
    mean_val = sum(true_expr_values) / n
    if abs(mean_val) < 1e-300:
        scale = max(abs(v) for v in true_expr_values)
        if scale < 1e-300:
            return  # All values essentially zero — trivially constant
        variance = sum((v - mean_val) ** 2 for v in true_expr_values) / n
        std_val = math.sqrt(max(variance, 0.0))
        constancy = 1.0 / (1.0 + std_val / scale)
    else:
        variance = sum((v - mean_val) ** 2 for v in true_expr_values) / n
        std_val = math.sqrt(max(variance, 0.0))
        constancy = 1.0 / (1.0 + std_val / abs(mean_val))

    if constancy < 0.95:
        raise ValueError(
            f"Ground-truth expression {expression!r} constancy={constancy:.4f} < 0.95 "
            f"on noise-free data — expression may not actually be constant. "
            f"Values: mean={mean_val:.6g}, std={std_val:.6g}"
        )


def _validate_post_noise(
    timesteps: list[dict[str, float]],
    quantities: dict[str, str],
) -> None:
    """Validate constraints after noise is applied.

    Checks:
    - No timestep has zero or negative values for mass/length/time base quantities.
    """
    _MASS_DIM_NAMES = {"Mass", "mass"}
    _LENGTH_DIM_NAMES = {"Length", "length"}
    _TIME_DIM_NAMES = {"Time", "time"}

    mass_quantities: set[str] = set()
    length_quantities: set[str] = set()
    time_quantities: set[str] = set()

    for qname, dim_name in quantities.items():
        if dim_name in _MASS_DIM_NAMES:
            mass_quantities.add(qname)
        elif dim_name in _LENGTH_DIM_NAMES:
            length_quantities.add(qname)
        elif dim_name in _TIME_DIM_NAMES:
            time_quantities.add(qname)

    for i, ts in enumerate(timesteps):
        for qname in mass_quantities | length_quantities | time_quantities:
            if qname in ts:
                val = ts[qname]
                if val <= 0:
                    raise ValueError(
                        f"Timestep {i}: quantity {qname!r} = {val} <= 0 "
                        f"(dimension type: {quantities.get(qname, 'unknown')})"
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: batch simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_batch(
    expressions: list[str],
    quantities_list: list[dict[str, str]],
    num_configs: int = 20,
    noise_frac: float = 0.03,
    seed: int | None = None,
) -> list[Observation]:
    """Simulate observations for multiple expressions.

    Each expression gets its own independent seed offset.

    Parameters
    ----------
    expressions : list[str]
        Ground-truth expressions.
    quantities_list : list[dict[str, str]]
        Quantity dimension mapping for each expression (must match length of expressions).
    num_configs : int
        Configurations per expression.
    noise_frac : float
        Relative noise level.
    seed : int or None
        Base random seed.

    Returns
    -------
    list[Observation]
        Flattened list of all generated observations.
    """
    if len(expressions) != len(quantities_list):
        raise ValueError(
            f"Length mismatch: {len(expressions)} expressions vs "
            f"{len(quantities_list)} quantity mappings"
        )

    all_obs: list[Observation] = []
    for i, (expr, quants) in enumerate(zip(expressions, quantities_list)):
        expr_seed = (seed or 0) + i * 10000
        obs_list = simulate_observations(
            expression=expr,
            quantities=quants,
            num_configs=num_configs,
            noise_frac=noise_frac,
            seed=expr_seed,
        )
        all_obs.extend(obs_list)

    return all_obs
