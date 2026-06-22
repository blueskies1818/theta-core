"""Hidden variable discovery v2 — continuous ratios, multi-variable groups, era-gated.

v2 additions over v1:
  - 3 new error shapes: linear_ratio, power_law, multi_var
  - 2 new variable types: continuous_ratio, continuous_additive
  - 2 new transforms: ratio, offset
  - ~150 training examples covering integer/quantum + continuous ratio + exponent + multi-var
  - Smart continuous variable injection in _augment_observations
  - Rule-based proposer updated for relativistic and photoelectric patterns
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.physics.dimensions import Dimension
from src.physics.evaluator import ExpressionEvaluator, EvalError, evaluate_node
from src.physics.observations import Observation


# Threshold for detecting grouped quantities
GROUPED_CORRELATION_THRESHOLD = 0.65  # Pearson r for "quantities co-vary"
GROUPED_MIN_PAIRS = 1  # Minimum number of grouped pairs to trigger detection


# =============================================================================
# Constants (expanded for v2)
# =============================================================================

# Error shapes
SHAPE_LINEAR = "linear"
SHAPE_QUADRATIC = "quadratic"
SHAPE_INVERSE_SQUARE = "inverse_square"
SHAPE_EXPONENTIAL = "exponential"
SHAPE_PERIODIC = "periodic"
SHAPE_RANDOM = "random"
SHAPE_CONSTANT = "constant"
SHAPE_LINEAR_RATIO = "linear_ratio"       # v2: errors scale linearly with one quantity → ratio
SHAPE_POWER_LAW = "power_law"            # v2: errors follow power law → exponent
SHAPE_MULTI_VAR = "multi_var"            # v2: errors correlate with 2+ variables → group
SHAPE_GROUPED = "grouped"                # v4: quantities co-vary as a group → metric relationship

ALL_SHAPES = [
    SHAPE_LINEAR, SHAPE_QUADRATIC, SHAPE_INVERSE_SQUARE,
    SHAPE_EXPONENTIAL, SHAPE_PERIODIC, SHAPE_RANDOM, SHAPE_CONSTANT,
    SHAPE_LINEAR_RATIO, SHAPE_POWER_LAW, SHAPE_MULTI_VAR,
    SHAPE_GROUPED,
]
NUM_SHAPES = len(ALL_SHAPES)
SHAPE_TO_IDX = {s: i for i, s in enumerate(ALL_SHAPES)}
IDX_TO_SHAPE = {i: s for s, i in SHAPE_TO_IDX.items()}

# Variable types
VAR_INTEGER = "integer_n"
VAR_HALF_INTEGER = "half_integer"
VAR_ANGULAR_M = "angular_m"
VAR_SPIN = "spin_s"
VAR_CONTINUOUS = "continuous"
VAR_CONTINUOUS_RATIO = "continuous_ratio"      # v2: μ, γ, drag coefficient
VAR_CONTINUOUS_ADDITIVE = "continuous_additive"  # v2: φ, work function, offset
VAR_GROUPED = "grouped_quantity"                  # v4: groups of co-varying quantities → metric

VAR_TYPES = [
    VAR_INTEGER, VAR_HALF_INTEGER, VAR_ANGULAR_M, VAR_SPIN, VAR_CONTINUOUS,
    VAR_CONTINUOUS_RATIO, VAR_CONTINUOUS_ADDITIVE,
    VAR_GROUPED,
]
NUM_VAR_TYPES = len(VAR_TYPES)
VAR_TYPE_TO_IDX = {t: i for i, t in enumerate(VAR_TYPES)}
IDX_TO_VAR_TYPE = {i: t for t, i in VAR_TYPE_TO_IDX.items()}

# Transforms
TRANSFORM_IDENTITY = "identity"
TRANSFORM_SQUARED = "squared"
TRANSFORM_INV_SQUARED = "inv_squared"
TRANSFORM_SQRT = "sqrt"
TRANSFORM_RATIO = "ratio"       # v2: var = a/b (for continuous ratios like γ=v/c)
TRANSFORM_OFFSET = "offset"     # v2: var = constant additive offset (like φ)
TRANSFORM_METRIC = "metric"     # v4: squared-difference metric invariant (ct)²-x²

TRANSFORMS = [
    TRANSFORM_IDENTITY, TRANSFORM_SQUARED, TRANSFORM_INV_SQUARED,
    TRANSFORM_SQRT, TRANSFORM_RATIO, TRANSFORM_OFFSET,
    TRANSFORM_METRIC,
]
NUM_TRANSFORMS = len(TRANSFORMS)
TRANSFORM_TO_IDX = {t: i for i, t in enumerate(TRANSFORMS)}
IDX_TO_TRANSFORM = {i: t for t, i in TRANSFORM_TO_IDX.items()}

# Domains
HIDDEN_VAR_DOMAINS = [
    "gravity", "spring", "em", "thermal", "quantum", "relativistic", "unknown",
]
NUM_HV_DOMAINS = len(HIDDEN_VAR_DOMAINS)
HV_DOMAIN_TO_IDX = {d: i for i, d in enumerate(HIDDEN_VAR_DOMAINS)}

# Expression Templates (v3 — proposer emits expression fragments, not just names)
# Each template defines a hidden variable with its relationship to known quantities.
_EXPRESSION_TEMPLATES = [
    # Integer/quantum archetypes
    "n = hbar*omega/E",
    "n = sqrt(hbar*c/(E*lambda))",
    "n = L*sqrt(2*m*E)/(pi*hbar)",
    "j = (E/(hbar*omega)) - 0.5",
    "m_l = delta_E/(mu_B*B)",
    "s = (delta_E/(g*mu_B*B) - 1)/2",
    # Continuous ratio archetypes
    "mu = F/N",
    "C_d = 2*F/(rho*A*v^2)",
    "n_refr = c/v",
    "gamma = 1/sqrt(1 - beta^2)",
    "beta = v/c",
    "k_gas = P*V/(n*T)",
    "R_gas = P*V/(n*T)",
    "k_kepler = T^2/a^3",
    "k_spring = 2*E/x^2",
    "k_invsq = F*r^2/(m1*m2)",
    "G_invsq = F*r^2/(m1*m2)",
    "T_factor = 2*pi*sqrt(L/g)",
    # Continuous additive archetypes
    "phi = h*f - K_max",
    "eta = 1 - Q_out/Q_in",
    "Q_loss = Q_in - W_out",
    # Power law / exponent archetypes
    "alpha = log(E/E0)/log(x)",
    "p_coeff = (E1/E2)^(1/3)",
    # Generic / fallback
    "alpha = 1/2",
    "n_scale = E/hbar",
    # v4: Grouped quantity metric forms (spacetime interval archetypes)
    "s2 = (c*t)^2 - x^2",       # Spacetime interval (Minkowski metric)
    "s2 = (c*t)^2 + x^2",       # Euclidean metric candidate
    "s2 = t^2 - (x/c)^2",       # Alternative scaling
    "s2 = (c*tau)^2 - x^2",     # Proper time variant
    "s2 = (c*t)^2 - r^2",       # Radial variant
    "s2 = x^2 + y^2 + z^2 - (c*t)^2",  # Full 4D Minkowski
]
# Add "none" (no defining expression) as index 0
_EXPRESSION_TEMPLATES.insert(0, "none")
NUM_EXPR_TEMPLATES = len(_EXPRESSION_TEMPLATES)
EXPR_TEMPLATE_TO_IDX = {e: i for i, e in enumerate(_EXPRESSION_TEMPLATES)}
IDX_TO_EXPR_TEMPLATE = {i: e for e, i in EXPR_TEMPLATE_TO_IDX.items()}

# Quantity vocabulary (expanded for v2 — friction, optics, gas, oscillation terms)
_HV_QUANTITY_VOCAB = [
    "m", "g", "h", "v", "t", "k", "L", "q", "E", "x", "y", "r",
    "P", "V", "T", "S", "n", "R", "B", "W", "Q", "c", "p",
    "m1", "v1", "m2", "v2", "x1", "x2", "epsilon",
    "hbar", "omega", "gamma", "lambda", "tau",
    "vx", "vy", "theta", "delta_x", "delta_p",
    "n_i", "n_f", "f", "phi", "a", "e", "delta_phi_obs",
    # v2 additions for new archetypes
    "mu", "N", "F", "rho", "A", "C_d", "nu", "n_refr",
    "T_period", "alpha", "beta", "K_max",
    "x_0", "v_0", "delta_t", "delta_tau", "u",
    "Q_in", "Q_out", "W_out", "eta",
    # v4: Spacetime / metric detection quantities
    "tau", "s2", "ds2", "t_lab", "x_lab",
]
HV_QTY_TO_IDX = {q: i for i, q in enumerate(_HV_QUANTITY_VOCAB)}
NUM_HV_QUANTITIES = len(_HV_QUANTITY_VOCAB)


# =============================================================================
# Data Classes (unchanged from v1, except docstrings)
# =============================================================================

@dataclass
class CurveFitResult:
    shape: str
    r_squared: float
    params: list[float]
    fitted_values: list[float]


@dataclass
class ErrorShapeAnalysis:
    shape: str
    shape_confidence: float
    per_shape_scores: dict[str, float]
    top_expressions: list[str]
    expression_scores: list[float]
    per_obs_values: list[list[float]]
    observation_count: int
    sample_expression_count: int
    mean_cv: float


@dataclass
class HiddenVariableProposal:
    variable_type: str
    variable_name: str
    transform: str
    rationale: str
    confidence: float
    expression_patch: str
    expression_template: str = "none"  # v3: defining expression fragment
    dimension_hint: str = "Scalar"
    expression_fragment: str = ""  # v3: the defining expression e.g. "gamma = 1/sqrt(1 - beta^2)"


@dataclass
class DiscoveryResult:
    discovered: bool
    hidden_variable: str | None
    transform: str | None
    best_expression: str | None
    best_score: float
    baseline_score: float
    num_proposals_tried: int
    proposals: list[HiddenVariableProposal] = field(default_factory=list)
    error_analysis: ErrorShapeAnalysis | None = None
    metadata: dict = field(default_factory=dict)


# =============================================================================
# 1. Error Shape Detector (v2 — expanded fitting)
# =============================================================================

class ErrorShapeDetector:
    """Analyze residuals from failed beam search to detect hidden structure.

    v2 adds: linear_ratio (correlation with a single quantity), power_law (log-log fit),
    and multi_var (residuals correlate with 2+ variables).
    """

    def __init__(self, *, r_squared_threshold: float = 0.7) -> None:
        self._evaluator = ExpressionEvaluator()
        self.r_squared_threshold = r_squared_threshold

    def analyze(
        self,
        scored_expressions: dict[str, float],
        observations: list[Observation],
        *,
        top_k: int = 100,
    ) -> ErrorShapeAnalysis:
        if not scored_expressions:
            return ErrorShapeAnalysis(
                shape=SHAPE_RANDOM, shape_confidence=0.0,
                per_shape_scores={}, top_expressions=[], expression_scores=[],
                per_obs_values=[], observation_count=len(observations),
                sample_expression_count=0, mean_cv=1.0,
            )

        sorted_exprs = sorted(scored_expressions.items(), key=lambda x: -x[1])[:top_k]
        top_exprs = [e for e, _ in sorted_exprs]
        top_scores = [s for _, s in sorted_exprs]

        per_obs_vals: list[list[float]] = []
        for expr_str in top_exprs:
            vals = self._evaluate_across_observations(expr_str, observations)
            if vals:
                per_obs_vals.append(vals)
            if len(per_obs_vals) >= top_k:
                break

        if not per_obs_vals:
            return ErrorShapeAnalysis(
                shape=SHAPE_RANDOM, shape_confidence=0.0,
                per_shape_scores={}, top_expressions=top_exprs[:10],
                expression_scores=top_scores[:10],
                per_obs_values=[], observation_count=len(observations),
                sample_expression_count=0, mean_cv=1.0,
            )

        shape_scores: dict[str, list[float]] = defaultdict(list)
        for vals in per_obs_vals:
            fits = self._fit_all_shapes(vals)
            for shape, fit in fits.items():
                shape_scores[shape].append(fit.r_squared)

        avg_shape_scores: dict[str, float] = {}
        for shape, scores in shape_scores.items():
            avg_shape_scores[shape] = sum(scores) / len(scores) if scores else 0.0

        # v2: also check for ratio/power/multi-var patterns
        ratio_score = self._detect_linear_ratio(per_obs_vals, observations)
        power_score = self._detect_power_law(per_obs_vals)
        multi_score = self._detect_multi_var(per_obs_vals, observations)

        avg_shape_scores[SHAPE_LINEAR_RATIO] = ratio_score
        avg_shape_scores[SHAPE_POWER_LAW] = power_score
        avg_shape_scores[SHAPE_MULTI_VAR] = multi_score

        # v4: detect grouped quantities that co-vary across observations
        grouped_score = self._detect_grouped(per_obs_vals, observations)
        avg_shape_scores[SHAPE_GROUPED] = grouped_score

        if avg_shape_scores:
            best_shape = max(avg_shape_scores, key=lambda s: avg_shape_scores[s])
            best_conf = avg_shape_scores[best_shape]
        else:
            best_shape = SHAPE_RANDOM
            best_conf = 0.0

        cvs: list[float] = []
        for vals in per_obs_vals:
            mean_v = sum(vals) / len(vals) if vals else 0.0
            if abs(mean_v) < 1e-12:
                cvs.append(0.0)
            else:
                var = sum((v - mean_v) ** 2 for v in vals) / len(vals)
                cvs.append(math.sqrt(max(var, 0.0)) / abs(mean_v))
        mean_cv = sum(cvs) / len(cvs) if cvs else 1.0

        return ErrorShapeAnalysis(
            shape=best_shape, shape_confidence=best_conf,
            per_shape_scores=avg_shape_scores,
            top_expressions=top_exprs[:20], expression_scores=top_scores[:20],
            per_obs_values=per_obs_vals, observation_count=len(observations),
            sample_expression_count=len(per_obs_vals), mean_cv=mean_cv,
        )

    def _detect_linear_ratio(
        self, per_obs_vals: list[list[float]], observations: list[Observation],
    ) -> float:
        """Detect if expression values scale linearly with a quantity — suggests hidden ratio."""
        if not per_obs_vals or not observations:
            return 0.0
        # Check if expression values correlate with any quantity in observations
        scores: list[float] = []
        primary_obs = observations[0]
        if not primary_obs.timesteps:
            return 0.0

        # For each quantity in the observations, check correlation with expression values
        for qname in primary_obs.quantities:
            qvals: list[float] = []
            for ts in primary_obs.timesteps:
                val = ts.get(qname, ts.get("t", 0.0))
                if isinstance(val, (int, float)):
                    qvals.append(float(val))
                else:
                    qvals.append(0.0)
            if not qvals or len(qvals) < 2:
                continue
            # Check if ANY expression's values correlate with this quantity
            for vals in per_obs_vals:
                n = min(len(vals), len(qvals))
                if n < 3:
                    continue
                corr = self._pearson_r(vals[:n], qvals[:n])
                if abs(corr) > 0.8:
                    scores.append(abs(corr))
        if scores:
            return sum(scores) / len(scores)
        return 0.0

    def _detect_power_law(self, per_obs_vals: list[list[float]]) -> float:
        """Detect power-law patterns (log-log linear)."""
        if not per_obs_vals:
            return 0.0
        scores: list[float] = []
        for vals in per_obs_vals:
            n = len(vals)
            if n < 4:
                continue
            x = list(range(1, n + 1))
            # Log-log fit
            log_x = [math.log(max(xi, 1e-10)) for xi in x]
            log_y = [math.log(max(v, 1e-10)) for v in vals]
            r2 = self._linear_fit_r2(log_x, log_y)
            # Only count as power_law if it beats linear by significant margin
            lin_r2 = self._linear_fit_r2([float(xi) for xi in x], vals)
            if r2 > 0.7 and r2 > lin_r2 + 0.1:
                scores.append(r2)
        if scores:
            return sum(scores) / len(scores)
        return 0.0

    def _detect_multi_var(
        self, per_obs_vals: list[list[float]], observations: list[Observation],
    ) -> float:
        """Detect if residuals depend on 2+ variables — suggests multi-var group."""
        if not observations or not observations[0].timesteps:
            return 0.0
        primary_obs = observations[0]
        qnames = list(primary_obs.quantities.keys())
        if len(qnames) < 2:
            return 0.0

        # For each pair of quantities, check if expressions correlate with their product/ratio
        scores: list[float] = []
        for i in range(len(qnames)):
            for j in range(i + 1, len(qnames)):
                qa, qb = qnames[i], qnames[j]
                product_vals: list[float] = []
                for ts in primary_obs.timesteps:
                    va = ts.get(qa, 1.0)
                    vb = ts.get(qb, 1.0)
                    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                        product_vals.append(float(va) * float(vb))
                    else:
                        product_vals.append(1.0)
                if not product_vals or len(product_vals) < 3:
                    continue
                for vals in per_obs_vals:
                    n = min(len(vals), len(product_vals))
                    if n < 3:
                        continue
                    corr = self._pearson_r(vals[:n], product_vals[:n])
                    if abs(corr) > 0.6:
                        scores.append(abs(corr))
        if scores:
            return sum(scores) / len(scores)
        return 0.0

    def _detect_grouped(
        self, per_obs_vals: list[list[float]], observations: list[Observation],
    ) -> float:
        """v4: Detect if pairs of quantities co-vary across observations — they form a group.

        A group is 2+ quantities whose values move together (high correlation)
        OR whose product/ratio/difference is near-constant across observations.
        This is the spacetime detection signal: if t and x move together across
        frames, they form a group → propose a metric relationship.

        Returns a score [0, 1] — higher means stronger group evidence.
        """
        if not observations or not observations[0].timesteps:
            return 0.0
        if len(observations) < 2:
            return 0.0

        primary_obs = observations[0]
        qnames = list(primary_obs.quantities.keys())
        num_qnames: list[str] = []
        for q in qnames:
            for ts in primary_obs.timesteps:
                if q in ts and isinstance(ts[q], (int, float)):
                    num_qnames.append(q)
                    break
        if len(num_qnames) < 2:
            return 0.0

        q_vals: dict[str, list[float]] = {q: [] for q in num_qnames}
        for obs in observations:
            if not obs.timesteps:
                continue
            ts = obs.timesteps[0]
            for q in num_qnames:
                if q in ts:
                    v = ts[q]
                    if isinstance(v, (int, float)):
                        q_vals[q].append(float(v))

        scores: list[float] = []
        for i in range(len(num_qnames)):
            for j in range(i + 1, len(num_qnames)):
                qa, qb = num_qnames[i], num_qnames[j]
                va = q_vals[qa]
                vb = q_vals[qb]
                n = min(len(va), len(vb))
                if n < 3:
                    continue
                corr = self._pearson_r(va[:n], vb[:n])
                prod = [va[k] * vb[k] for k in range(n)]
                ratio = [va[k] / max(vb[k], 1e-10) for k in range(n)]
                sq_diff = [va[k]**2 - vb[k]**2 for k in range(n)]
                sq_sum = [va[k]**2 + vb[k]**2 for k in range(n)]

                def _cv(vals: list[float]) -> float:
                    m = sum(vals) / len(vals)
                    if abs(m) < 1e-12:
                        return 0.0
                    var = sum((v - m)**2 for v in vals) / len(vals)
                    return math.sqrt(max(var, 0.0)) / abs(m)

                prod_cv = _cv(prod)
                ratio_cv = _cv(ratio)
                sq_diff_cv = _cv(sq_diff)
                sq_sum_cv = _cv(sq_sum)
                min_cv = min(prod_cv, ratio_cv, sq_diff_cv, sq_sum_cv)

                corr_score = max(0.0, min(1.0, abs(corr) / GROUPED_CORRELATION_THRESHOLD))
                const_score = max(0.0, 1.0 - min_cv / 0.15) if min_cv < 1.0 else 0.0
                pair_score = max(corr_score, const_score)

                if pair_score > 0.4:
                    scores.append(pair_score)

        if not scores:
            return 0.0
        top_scores = sorted(scores, reverse=True)[:max(GROUPED_MIN_PAIRS, len(scores) // 2 + 1)]
        return sum(top_scores) / len(top_scores)

    @staticmethod
    def _pearson_r(x: list[float], y: list[float]) -> float:
        n = len(x)
        if n < 2:
            return 0.0
        mx = sum(x) / n
        my = sum(y) / n
        cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
        sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
        sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
        if sx < 1e-12 or sy < 1e-12:
            return 0.0
        return max(-1.0, min(1.0, cov / (sx * sy)))

    @staticmethod
    def _linear_fit_r2(x: list[float], y: list[float]) -> float:
        n = len(x)
        if n < 2:
            return 0.0
        sx, sy = sum(x), sum(y)
        sxx = sum(xi * xi for xi in x)
        sxy = sum(x[i] * y[i] for i in range(n))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12:
            a, b = 0.0, sy / n
        else:
            a = (n * sxy - sx * sy) / denom
            b = (sy - a * sx) / n
        fitted = [a * xi + b for xi in x]
        mean_y = sum(y) / n
        ss_res = sum((y[i] - fitted[i]) ** 2 for i in range(n))
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        if ss_tot < 1e-15:
            return 1.0 if ss_res < 1e-15 else 0.0
        return max(0.0, min(1.0, 1.0 - ss_res / ss_tot))

    def _evaluate_across_observations(
        self, expr_str: str, observations: list[Observation],
    ) -> list[float]:
        values: list[float] = []
        try:
            ast = self._evaluator.parse(expr_str)
        except Exception:
            return []

        if len(observations) == 1 and len(observations[0].timesteps) > 1:
            obs = observations[0]
            for ts in obs.timesteps:
                context = {**obs.parameters, **ts}
                try:
                    val = evaluate_node(ast, context)
                    if isinstance(val, (int, float)) and not math.isnan(val):
                        values.append(float(val))
                except (EvalError, ZeroDivisionError, ValueError, OverflowError):
                    values.append(0.0)
            return values

        for obs in observations:
            if not obs.timesteps:
                continue
            ts = obs.timesteps[0]
            context = {**obs.parameters, **ts}
            try:
                val = evaluate_node(ast, context)
                if isinstance(val, (int, float)) and not math.isnan(val):
                    values.append(float(val))
            except (EvalError, ZeroDivisionError, ValueError, OverflowError):
                values.append(0.0)
        return values

    def _fit_all_shapes(self, values: list[float]) -> dict[str, CurveFitResult]:
        n = len(values)
        if n < 3:
            return {}
        x = [float(i) for i in range(1, n + 1)]
        results: dict[str, CurveFitResult] = {}
        results[SHAPE_LINEAR] = self._fit_linear(x, values)
        results[SHAPE_QUADRATIC] = self._fit_quadratic(x, values)
        results[SHAPE_INVERSE_SQUARE] = self._fit_inverse_square(x, values)
        results[SHAPE_EXPONENTIAL] = self._fit_exponential(x, values)
        results[SHAPE_PERIODIC] = self._fit_periodic(x, values)
        results[SHAPE_CONSTANT] = self._fit_constant(values)
        return results

    def _fit_linear(self, x: list[float], y: list[float]) -> CurveFitResult:
        n = len(x)
        sx, sy = sum(x), sum(y)
        sxx = sum(xi * xi for xi in x)
        sxy = sum(x[i] * y[i] for i in range(n))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12:
            a, b = 0.0, sy / n
        else:
            a = (n * sxy - sx * sy) / denom
            b = (sy - a * sx) / n
        fitted = [a * xi + b for xi in x]
        r2 = self._r_squared(y, fitted)
        return CurveFitResult(shape=SHAPE_LINEAR, r_squared=r2, params=[a, b], fitted_values=fitted)

    def _fit_quadratic(self, x: list[float], y: list[float]) -> CurveFitResult:
        n = len(x)
        sx = sum(x)
        sx2 = sum(xi * xi for xi in x)
        sx3 = sum(xi ** 3 for xi in x)
        sx4 = sum(xi ** 4 for xi in x)
        sy = sum(y)
        sxy = sum(x[i] * y[i] for i in range(n))
        sx2y = sum(x[i] * x[i] * y[i] for i in range(n))
        A = [[float(n), sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]]
        B = [sy, sxy, sx2y]
        try:
            c_, b_, a_ = _solve_linear_3x3(A, B)
        except ValueError:
            a_, b_, c_ = 0.0, 0.0, sy / n if n else 0.0
        fitted = [a_ * xi * xi + b_ * xi + c_ for xi in x]
        r2 = self._r_squared(y, fitted)
        return CurveFitResult(shape=SHAPE_QUADRATIC, r_squared=r2, params=[a_, b_, c_], fitted_values=fitted)

    def _fit_inverse_square(self, x: list[float], y: list[float]) -> CurveFitResult:
        n = len(x)
        inv_x2 = [1.0 / (xi * xi) for xi in x]
        sx, sy = sum(inv_x2), sum(y)
        sxx = sum(v * v for v in inv_x2)
        sxy = sum(inv_x2[i] * y[i] for i in range(n))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12:
            a, b = 0.0, sy / n
        else:
            a = (n * sxy - sx * sy) / denom
            b = (sy - a * sx) / n
        fitted = [a * inv_x2[i] + b for i in range(n)]
        r2 = self._r_squared(y, fitted)
        return CurveFitResult(shape=SHAPE_INVERSE_SQUARE, r_squared=r2, params=[a, b], fitted_values=fitted)

    def _fit_exponential(self, x: list[float], y: list[float]) -> CurveFitResult:
        y_min = min(y)
        shifted = [max(v - y_min + 1e-10, 1e-10) for v in y]
        log_y = [math.log(v) for v in shifted]
        n = len(x)
        sx, sy = sum(x), sum(log_y)
        sxx = sum(xi * xi for xi in x)
        sxy = sum(x[i] * log_y[i] for i in range(n))
        denom = n * sxx - sx * sx
        b_ = 0.0 if abs(denom) < 1e-12 else (n * sxy - sx * sy) / denom
        log_a = (sy - b_ * sx) / n
        a_ = math.exp(log_a)
        fitted = [a_ * math.exp(b_ * xi) + y_min for xi in x]
        r2 = self._r_squared(y, fitted)
        return CurveFitResult(shape=SHAPE_EXPONENTIAL, r_squared=r2, params=[a_, b_, y_min], fitted_values=fitted)

    def _fit_periodic(self, x: list[float], y: list[float]) -> CurveFitResult:
        n = len(y)
        mean_y = sum(y) / n
        centered = [v - mean_y for v in y]
        period = self._detect_period(centered)
        if period is None or period < 2:
            fitted = [mean_y] * n
            r2 = self._r_squared(y, fitted)
            return CurveFitResult(shape=SHAPE_PERIODIC, r_squared=r2, params=[0.0, 0.0, 0.0, mean_y], fitted_values=fitted)
        b = 2 * math.pi / period
        sin_sum = sum(math.sin(b * x[i]) * centered[i] for i in range(n))
        cos_sum = sum(math.cos(b * x[i]) * centered[i] for i in range(n))
        amp = math.sqrt(sin_sum**2 + cos_sum**2) * 2.0 / n
        phase = math.atan2(cos_sum, sin_sum) if abs(sin_sum) > 1e-12 else 0.0
        fitted = [amp * math.sin(b * xi + phase) + mean_y for xi in x]
        r2 = self._r_squared(y, fitted)
        return CurveFitResult(shape=SHAPE_PERIODIC, r_squared=r2, params=[amp, b, phase, mean_y], fitted_values=fitted)

    def _fit_constant(self, values: list[float]) -> CurveFitResult:
        mean_val = sum(values) / len(values) if values else 0.0
        fitted = [mean_val] * len(values)
        r2 = self._r_squared(values, fitted)
        return CurveFitResult(shape=SHAPE_CONSTANT, r_squared=r2, params=[mean_val], fitted_values=fitted)

    @staticmethod
    def _r_squared(y_true: list[float], y_pred: list[float]) -> float:
        n = len(y_true)
        if n < 2:
            return 0.0
        mean_y = sum(y_true) / n
        ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((yi - mean_y) ** 2 for yi in y_true)
        if ss_tot < 1e-15:
            return 1.0 if ss_res < 1e-15 else 0.0
        return max(0.0, min(1.0, 1.0 - ss_res / ss_tot))

    @staticmethod
    def _detect_period(centered: list[float], max_lag: int | None = None) -> int | None:
        n = len(centered)
        if n < 4:
            return None
        max_lag = max_lag or n // 2
        best_lag, best_corr = None, 0.0
        for lag in range(2, min(max_lag, n - 1)):
            corr = 0.0
            count = 0
            for i in range(n - lag):
                corr += centered[i] * centered[i + lag]
                count += 1
            if count > 0:
                corr /= count
            if corr > best_corr:
                best_corr, best_lag = corr, lag
        return best_lag if best_corr > 0.3 else None


# =============================================================================
# 2. Hidden Variable Proposer v2 (MLP — expanded)
# =============================================================================

class HiddenVariableProposer(nn.Module):
    """Small MLP that proposes hidden variables from error shape + context.

    v3: adds expression template prediction — model learns to output the
    RELATIONSHIP (e.g. "gamma = 1/sqrt(1 - beta^2)"), not just the name.
    Input: [shape_encoding(NUM_SHAPES) + quantity_vector(NUM_HV_QUANTITIES) + domain_onehot(NUM_HV_DOMAINS)]
    Hidden: 64 -> 48 -> 32
    Output: [var_type_logits(NUM_VAR_TYPES) + transform_logits(NUM_TRANSFORMS) + confidence
             + expr_template_logits(NUM_EXPR_TEMPLATES)]
    ~16K parameters.
    """

    def __init__(self, *, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        input_dim = NUM_SHAPES + NUM_HV_QUANTITIES + NUM_HV_DOMAINS
        output_dim = NUM_VAR_TYPES + NUM_TRANSFORMS + 1 + NUM_EXPR_TEMPLATES

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        return self.fc3(h)

    def propose(
        self,
        shape_encoding: torch.Tensor,
        quantity_vector: torch.Tensor,
        domain_onehot: torch.Tensor,
        *,
        temperature: float = 0.15,
    ) -> list[HiddenVariableProposal]:
        x = torch.cat([shape_encoding, quantity_vector, domain_onehot], dim=-1)
        output = self.forward(x)

        var_logits = output[:, :NUM_VAR_TYPES]
        transform_logits = output[:, NUM_VAR_TYPES:NUM_VAR_TYPES + NUM_TRANSFORMS]
        confidence_logit = output[:, NUM_VAR_TYPES + NUM_TRANSFORMS]
        expr_logits = output[:, NUM_VAR_TYPES + NUM_TRANSFORMS + 1:]

        var_probs = F.softmax(var_logits / max(temperature, 1e-8), dim=-1)
        transform_probs = F.softmax(transform_logits / max(temperature, 1e-8), dim=-1)
        confidence = torch.sigmoid(confidence_logit)
        expr_probs = F.softmax(expr_logits / max(temperature, 1e-8), dim=-1)

        batch_size = x.size(0)
        qty_indices = quantity_vector.nonzero(as_tuple=False)
        name_map = {
            VAR_INTEGER: "n", VAR_HALF_INTEGER: "j",
            VAR_ANGULAR_M: "m_l", VAR_SPIN: "s", VAR_CONTINUOUS: "alpha",
            VAR_CONTINUOUS_RATIO: "gamma", VAR_CONTINUOUS_ADDITIVE: "phi",
            VAR_GROUPED: "s2",
        }
        patch_map = {
            TRANSFORM_IDENTITY: "*{name}", TRANSFORM_SQUARED: "*{name}^2",
            TRANSFORM_INV_SQUARED: "/{name}^2", TRANSFORM_SQRT: "*sqrt({name})",
            TRANSFORM_RATIO: "*{name}", TRANSFORM_OFFSET: "+{name}",
            TRANSFORM_METRIC: "*{name}",
        }

        results: list[HiddenVariableProposal] = []
        for b in range(batch_size):
            var_idx = var_probs[b].argmax().item()
            transform_idx = transform_probs[b].argmax().item()
            conf_val = confidence[b].item()
            expr_idx = expr_probs[b].argmax().item()
            var_type = IDX_TO_VAR_TYPE.get(var_idx, VAR_INTEGER)
            transform = IDX_TO_TRANSFORM.get(transform_idx, TRANSFORM_SQUARED)
            expr_fragment = IDX_TO_EXPR_TEMPLATE.get(expr_idx, "n = E/hbar")

            # v3: derive variable_name from expression_fragment for consistency
            # e.g. "gamma = 1/sqrt(1 - beta^2)" → var_name="gamma"
            if " = " in expr_fragment:
                var_name = expr_fragment.split(" = ")[0].strip()
            else:
                var_name = name_map.get(var_type, "n")
            expr_patch = patch_map.get(transform, "*{name}").format(name=var_name)

            b_qty_indices = qty_indices[qty_indices[:, 0] == b]
            present_qtys = []
            for idx_tuple in b_qty_indices:
                q_idx = idx_tuple[1].item()
                if q_idx < len(_HV_QUANTITY_VOCAB):
                    present_qtys.append(_HV_QUANTITY_VOCAB[q_idx])
            qty_str = ", ".join(present_qtys[:6]) if present_qtys else "unknown"
            domain_idx = domain_onehot[b].argmax().item()
            domain = HIDDEN_VAR_DOMAINS[domain_idx] if domain_idx < NUM_HV_DOMAINS else "unknown"

            results.append(HiddenVariableProposal(
                variable_type=var_type, variable_name=var_name,
                transform=transform,
                rationale=f"Error shape suggests {var_type} with {transform} in {domain} domain (quantities: {qty_str})",
                confidence=conf_val, expression_patch=expr_patch,
                expression_fragment=expr_fragment,
            ))
        return results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Feature Construction (v2 — updated for expanded vocab)
# =============================================================================

def shape_to_encoding(shape: str, *, soft: bool = True) -> torch.Tensor:
    enc = torch.zeros(NUM_SHAPES)
    if shape not in SHAPE_TO_IDX:
        return enc
    if soft:
        idx = SHAPE_TO_IDX[shape]
        enc[idx] = 0.7
        remaining = 0.3
        neighbors = _shape_neighbors(shape)
        if neighbors:
            weight = remaining / len(neighbors)
            for n_shape in neighbors:
                if n_shape in SHAPE_TO_IDX:
                    enc[SHAPE_TO_IDX[n_shape]] += weight
    else:
        enc[SHAPE_TO_IDX[shape]] = 1.0
    return enc


def _shape_neighbors(shape: str) -> list[str]:
    neighbors = {
        SHAPE_QUADRATIC: [SHAPE_INVERSE_SQUARE, SHAPE_LINEAR],
        SHAPE_INVERSE_SQUARE: [SHAPE_QUADRATIC, SHAPE_CONSTANT],
        SHAPE_LINEAR: [SHAPE_QUADRATIC, SHAPE_LINEAR_RATIO, SHAPE_CONSTANT],
        SHAPE_EXPONENTIAL: [SHAPE_QUADRATIC, SHAPE_POWER_LAW],
        SHAPE_PERIODIC: [SHAPE_RANDOM],
        SHAPE_CONSTANT: [SHAPE_LINEAR, SHAPE_INVERSE_SQUARE],
        SHAPE_RANDOM: [SHAPE_PERIODIC],
        SHAPE_LINEAR_RATIO: [SHAPE_LINEAR, SHAPE_CONSTANT],
        SHAPE_POWER_LAW: [SHAPE_EXPONENTIAL, SHAPE_QUADRATIC],
        SHAPE_MULTI_VAR: [SHAPE_LINEAR_RATIO, SHAPE_QUADRATIC],
        SHAPE_GROUPED: [SHAPE_MULTI_VAR, SHAPE_LINEAR_RATIO, SHAPE_CONSTANT],
    }
    return neighbors.get(shape, [])


def quantities_to_proposer_vector(quantity_names: list[str]) -> torch.Tensor:
    vec = torch.zeros(NUM_HV_QUANTITIES)
    for name in quantity_names:
        if name in HV_QTY_TO_IDX:
            vec[HV_QTY_TO_IDX[name]] = 1.0
    return vec


def domain_to_proposer_onehot(domain: str) -> torch.Tensor:
    vec = torch.zeros(NUM_HV_DOMAINS)
    if domain in HV_DOMAIN_TO_IDX:
        vec[HV_DOMAIN_TO_IDX[domain]] = 1.0
    else:
        vec[HV_DOMAIN_TO_IDX["unknown"]] = 1.0
    return vec


# =============================================================================
# 3. Hidden Variable Discovery (v2 — expanded rule-based proposer)
# =============================================================================

class HiddenVariableDiscovery:
    """Orchestrate the hidden variable discovery pipeline.

    v2: rule-based proposer now handles continuous ratios, additive offsets,
    and multi-variable groups in addition to the v1 quantum/integer patterns.
    """

    def __init__(
        self,
        *,
        proposer: HiddenVariableProposer | None = None,
        detector: ErrorShapeDetector | None = None,
        max_proposals: int = 5,
        score_improvement_threshold: float = 0.15,
        discovery_threshold: float = 0.95,
    ) -> None:
        self.proposer = proposer or HiddenVariableProposer()
        self.detector = detector or ErrorShapeDetector()
        self.max_proposals = max_proposals
        self.score_improvement_threshold = score_improvement_threshold
        self.discovery_threshold = discovery_threshold
        self._evaluator = ExpressionEvaluator()

    def discover(
        self,
        quantities: dict[str, Dimension],
        observations: list[Observation],
        beam_search_fn: Callable,
        *,
        domain: str = "unknown",
        quantity_names: list[str] | None = None,
    ) -> DiscoveryResult:
        qty_names = quantity_names or list(quantities.keys())

        search_result = beam_search_fn(quantities, observations)
        baseline_score = getattr(search_result, "best_score", 0.0)
        baseline_expr = getattr(search_result, "best_expression", "")

        if getattr(search_result, "discovered", False):
            return DiscoveryResult(
                discovered=True, hidden_variable=None, transform=None,
                best_expression=baseline_expr, best_score=baseline_score,
                baseline_score=baseline_score, num_proposals_tried=0,
                metadata={"source": "known_quantities"},
            )

        scored_exprs = getattr(search_result, "_scored", {})
        if not scored_exprs:
            scored_exprs = self._collect_all_scored(quantities, observations)

        analysis = self.detector.analyze(scored_exprs, observations)
        proposals = self._propose(analysis, qty_names, domain, observations)

        best_result_score = baseline_score
        best_result_expr = baseline_expr
        best_var = None
        best_transform = None

        for i, proposal in enumerate(proposals[:self.max_proposals]):
            new_quantities = dict(quantities)
            new_quantities[proposal.variable_name] = Dimension.scalar()
            aug_obs = self._augment_observations(observations, proposal, new_quantities)
            try:
                new_result = beam_search_fn(new_quantities, aug_obs)
            except Exception:
                continue
            new_score = getattr(new_result, "best_score", 0.0)
            new_expr = getattr(new_result, "best_expression", "")
            if new_score > best_result_score:
                best_result_score = new_score
                best_result_expr = new_expr
                best_var = proposal.variable_name
                best_transform = proposal.transform

        score_improved = best_result_score - baseline_score > self.score_improvement_threshold
        discovered = best_result_score >= self.discovery_threshold and score_improved

        return DiscoveryResult(
            discovered=discovered,
            hidden_variable=best_var if discovered else None,
            transform=best_transform if discovered else None,
            best_expression=best_result_expr if discovered else baseline_expr,
            best_score=best_result_score,
            baseline_score=baseline_score,
            num_proposals_tried=min(len(proposals), self.max_proposals),
            proposals=proposals,
            error_analysis=analysis,
            metadata={
                "score_improvement": best_result_score - baseline_score,
                "domain": domain, "quantity_count": len(quantities),
                "observation_count": len(observations),
            },
        )

    def _propose(
        self, analysis: ErrorShapeAnalysis, quantity_names: list[str],
        domain: str, observations: list[Observation] | None = None,
    ) -> list[HiddenVariableProposal]:
        rule_proposals = self._rule_based_proposals(analysis, quantity_names, domain, observations)
        try:
            shape_enc = shape_to_encoding(analysis.shape).unsqueeze(0)
            qty_vec = quantities_to_proposer_vector(quantity_names).unsqueeze(0)
            domain_oh = domain_to_proposer_onehot(domain).unsqueeze(0)
            with torch.no_grad():
                mlp_proposals = self.proposer.propose(shape_enc, qty_vec, domain_oh)
        except Exception:
            mlp_proposals = []

        all_proposals = []
        added_names: set[str] = set()
        for p in mlp_proposals + rule_proposals:
            key = f"{p.variable_type}:{p.transform}"
            if key not in added_names:
                all_proposals.append(p)
                added_names.add(key)
        return all_proposals

    def _detect_observable_keys(self, observations: list[Observation]) -> set[str]:
        """Find keys present in timesteps or parameters that could serve as hidden variables."""
        keys: set[str] = set()
        for obs in observations:
            keys.update(obs.parameters.keys())
            for ts in obs.timesteps:
                keys.update(ts.keys())
        return keys

    def _rule_based_proposals(
        self, analysis: ErrorShapeAnalysis, quantity_names: list[str],
        domain: str, observations: list[Observation] | None = None,
    ) -> list[HiddenVariableProposal]:
        proposals: list[HiddenVariableProposal] = []
        shape = analysis.shape
        has_E = any("E" in q or q == "E" for q in quantity_names)
        has_lambda = any("l" in q.lower() or "lambda" in q for q in quantity_names)
        has_energy_like = has_E or any(q in quantity_names for q in ["E", "energy", "W", "Q", "K_max"])
        has_velocity = any(q in quantity_names for q in ["v", "v1", "v2", "u", "vx", "vy", "beta"])
        has_c = "c" in quantity_names
        is_quantum = domain == "quantum" or "hbar" in quantity_names or "omega" in quantity_names
        is_relativistic = domain == "relativistic" or {"gamma", "beta", "c"} & set(quantity_names)
        cvs = analysis.mean_cv

        # Check for observable keys (v2 addition — find real hidden var names in data)
        observable_keys: set[str] = set()
        if observations:
            observable_keys = self._detect_observable_keys(observations)

        # v2: Check for potential hidden var names in observations that aren't listed as quantities
        potential_hidden_names: set[str] = set()
        for obs in (observations or []):
            for ts in obs.timesteps:
                for key in ts:
                    if (key not in quantity_names
                            and key not in ("t",)
                            and isinstance(ts[key], (int, float))):
                        potential_hidden_names.add(key)
            for key in obs.parameters:
                if (key not in quantity_names
                        and isinstance(obs.parameters[key], (int, float))):
                    potential_hidden_names.add(key)

        # =====================================================================
        # v2: Continuous ratio proposals (TYPE 1 — linear scaling with a quantity)
        # =====================================================================
        if shape in (SHAPE_LINEAR, SHAPE_LINEAR_RATIO) and (is_relativistic or has_velocity):
            # Relativistic gamma pattern: errors scale with velocity → propose ratio
            # Look for gamma, beta, or c in observable keys
            ratio_name = "gamma"  # default
            if "gamma" in potential_hidden_names:
                ratio_name = "gamma"
            elif "beta" in potential_hidden_names:
                ratio_name = "beta"
            elif "c" in potential_hidden_names:
                ratio_name = "c"
            proposals.append(HiddenVariableProposal(
                variable_type=VAR_CONTINUOUS_RATIO, variable_name=ratio_name,
                transform=TRANSFORM_RATIO,
                rationale=f"Linear residuals in relativistic domain — propose continuous ratio {ratio_name} (γ-like factor)",
                confidence=0.80, expression_patch=f"*{ratio_name}",
                expression_fragment="gamma = 1/sqrt(1 - beta^2)",
            ))
            # Also propose identity (plain continuous)
            proposals.append(HiddenVariableProposal(
                variable_type=VAR_CONTINUOUS_RATIO, variable_name=ratio_name,
                transform=TRANSFORM_IDENTITY,
                rationale=f"Relativistic linear residuals — propose continuous multiplier {ratio_name}",
                confidence=0.65, expression_patch=f"*{ratio_name}",
                expression_fragment="gamma = 1/sqrt(1 - beta^2)",
            ))

        # v2: Continuous ratio for non-relativistic (friction, drag, etc.)
        if shape in (SHAPE_LINEAR, SHAPE_LINEAR_RATIO) and not is_relativistic and not is_quantum:
            # General ratio pattern (μ, drag coefficient, refractive index)
            ratio_name = "mu"
            if "mu" in potential_hidden_names:
                ratio_name = "mu"
            elif "alpha" in potential_hidden_names:
                ratio_name = "alpha"
            proposals.append(HiddenVariableProposal(
                variable_type=VAR_CONTINUOUS_RATIO, variable_name=ratio_name,
                transform=TRANSFORM_RATIO,
                rationale=f"Linear scaling residuals — propose continuous ratio constant (like friction coefficient μ)",
                confidence=0.75, expression_patch=f"*{ratio_name}",
                expression_fragment="mu = F/N",
            ))

        # =====================================================================
        # v2: Continuous additive proposals (TYPE 1b — constant offset like φ)
        # =====================================================================
        # Photoelectric: K_max = hf - φ → φ is additive offset
        if shape in (SHAPE_LINEAR, SHAPE_LINEAR_RATIO) and is_quantum and has_energy_like:
            if "phi" in potential_hidden_names:
                proposals.append(HiddenVariableProposal(
                    variable_type=VAR_CONTINUOUS_ADDITIVE, variable_name="phi",
                    transform=TRANSFORM_OFFSET,
                    rationale="Linear residuals with energy offset — propose work function φ (additive constant)",
                    confidence=0.78, expression_patch="+phi",
                    expression_fragment="phi = h*f - K_max",
                ))
            elif "phi" in quantity_names:
                pass  # phi already a quantity
            else:
                # Check for any additive-like pattern
                has_freq = any(q in quantity_names for q in ["f", "nu", "omega"])
                if has_freq:
                    proposals.append(HiddenVariableProposal(
                        variable_type=VAR_CONTINUOUS_ADDITIVE, variable_name="phi",
                        transform=TRANSFORM_OFFSET,
                        rationale="Frequency-energy relationship with offset — propose additive constant φ",
                        confidence=0.72, expression_patch="+phi",
                        expression_fragment="phi = h*f - K_max",
                    ))

        # =====================================================================
        # v2: Multi-variable group proposals (TYPE 3 — PV/T = const)
        # =====================================================================
        if shape == SHAPE_MULTI_VAR or (shape == SHAPE_LINEAR_RATIO and len(quantity_names) >= 2):
            # Ideal gas: P, V, T must be discovered together
            if {"P", "V", "T"} & set(quantity_names) == {"P", "V", "T"}:
                proposals.append(HiddenVariableProposal(
                    variable_type=VAR_CONTINUOUS_RATIO, variable_name="R",
                    transform=TRANSFORM_RATIO,
                    rationale="Multi-variable correlation P,V,T — propose gas constant R (P*V = n*R*T)",
                    confidence=0.80, expression_patch="*R",
                    expression_fragment="R_gas = P*V/(n*T)",
                ))
            # Heat engine: Q_in - W_out - Q_out = 0
            elif any(q in quantity_names for q in ["Q_in", "W_out"]):
                proposals.append(HiddenVariableProposal(
                    variable_type=VAR_CONTINUOUS_ADDITIVE, variable_name="Q_loss",
                    transform=TRANSFORM_OFFSET,
                    rationale="Multi-variable energy flow — propose heat loss term",
                    confidence=0.70, expression_patch="+Q_loss",
                    expression_fragment="Q_loss = Q_in - W_out",
                ))
            else:
                # Generic multi-var
                proposals.append(HiddenVariableProposal(
                    variable_type=VAR_CONTINUOUS_RATIO, variable_name="alpha",
                    transform=TRANSFORM_RATIO,
                    rationale="Multi-variable residuals — propose coupling constant",
                    confidence=0.65, expression_patch="*alpha",
                    expression_fragment="alpha = 1/2",
                ))

        # =====================================================================
        # v2: Power law proposals (TYPE 2 — exponent hidden in data)
        # =====================================================================
        if shape == SHAPE_POWER_LAW:
            proposals.append(HiddenVariableProposal(
                variable_type=VAR_CONTINUOUS_RATIO, variable_name="alpha",
                transform=TRANSFORM_IDENTITY,
                rationale="Power-law residuals — propose hidden exponent/relationship",
                confidence=0.70, expression_patch="*alpha",
                expression_fragment="k_kepler = T^2/a^3",
            ))

        # =====================================================================
        # v1 patterns (kept for backward compatibility)
        # =====================================================================

        # inverse_square + quantum -> propose integer n with n^2
        if shape == SHAPE_INVERSE_SQUARE and is_quantum:
            if has_E and has_lambda:
                proposals.append(HiddenVariableProposal(
                    variable_type=VAR_INTEGER, variable_name="n",
                    transform=TRANSFORM_SQUARED,
                    rationale="Quantum spectrum with 1/n^2 residuals — propose integer n, use n^2",
                    confidence=0.85, expression_patch="*n^2",
                ))
                proposals.append(HiddenVariableProposal(
                    variable_type=VAR_INTEGER, variable_name="n",
                    transform=TRANSFORM_IDENTITY,
                    rationale="Quantum spectrum — propose integer n as-is",
                    confidence=0.60, expression_patch="*n",
                ))
            elif has_energy_like:
                proposals.append(HiddenVariableProposal(
                    variable_type=VAR_INTEGER, variable_name="n",
                    transform=TRANSFORM_SQUARED,
                    rationale="1/n^2 residuals with energy — propose integer n",
                    confidence=0.75, expression_patch="*n^2",
                ))

        # quadratic + energy -> propose integer n
        if shape == SHAPE_QUADRATIC and has_energy_like:
            proposals.append(HiddenVariableProposal(
                variable_type=VAR_INTEGER, variable_name="n",
                transform=TRANSFORM_SQUARED,
                rationale="Quadratic residuals suggest n^2 energy levels",
                confidence=0.70, expression_patch="*n^2",
            ))
            proposals.append(HiddenVariableProposal(
                variable_type=VAR_INTEGER, variable_name="n",
                transform=TRANSFORM_INV_SQUARED,
                rationale="Quadratic residuals — try 1/n^2 alternative",
                confidence=0.50, expression_patch="/n^2",
            ))

        # linear -> propose integer counting (harmonic oscillator pattern)
        if shape == SHAPE_LINEAR:
            proposals.append(HiddenVariableProposal(
                variable_type=VAR_INTEGER, variable_name="n",
                transform=TRANSFORM_IDENTITY,
                rationale="Linear residuals suggest counting/index pattern",
                confidence=0.65, expression_patch="*n",
            ))
            # Also propose half-integer for harmonic oscillator: E = (n+1/2)*hbar*omega
            if is_quantum or "omega" in quantity_names:
                proposals.append(HiddenVariableProposal(
                    variable_type=VAR_HALF_INTEGER, variable_name="j",
                    transform=TRANSFORM_IDENTITY,
                    rationale="Linear residuals in quantum system — could be (n+1/2) harmonic oscillator pattern",
                    confidence=0.55, expression_patch="*j",
                ))

        # periodic -> angular variable
        if shape == SHAPE_PERIODIC:
            proposals.append(HiddenVariableProposal(
                variable_type=VAR_ANGULAR_M, variable_name="m",
                transform=TRANSFORM_IDENTITY,
                rationale="Periodic residuals suggest angular momentum m",
                confidence=0.60, expression_patch="*m",
            ))

        # If values are nearly constant but not passing -> not a hidden var case
        if shape == SHAPE_CONSTANT and cvs < 0.05:
            pass

        # =====================================================================
        # v4: Grouped quantity proposals (spacetime metric discovery)
        # =====================================================================
        if shape == SHAPE_GROUPED or (shape in (SHAPE_LINEAR_RATIO, SHAPE_MULTI_VAR) and
                {"c", "t", "x", "tau"} & set(quantity_names)):
            # Detect which pairs form a group
            # Check for t-x pairing (spacetime signal)
            has_t = any(q in quantity_names for q in ["t", "t_lab"])
            has_x = any(q in quantity_names for q in ["x", "x_lab", "r"])
            has_c = "c" in quantity_names or "c" in potential_hidden_names
            has_tau = "tau" in quantity_names or "tau" in potential_hidden_names

            if has_t and has_x:
                # t and x are grouped -> propose metric relationships
                # The metric variable s2 represents the invariant interval
                metric_name = "s2"

                # Proposal 1: squared difference with c (Minkowski metric)
                if has_c:
                    proposals.append(HiddenVariableProposal(
                        variable_type=VAR_GROUPED, variable_name=metric_name,
                        transform=TRANSFORM_METRIC,
                        rationale="t and x co-vary across frames - propose Minkowski metric (ct)^2-x^2 invariant",
                        confidence=0.85, expression_patch=f"*{metric_name}",
                        expression_fragment="s2 = (c*t)^2 - x^2",
                    ))
                    # Proposal 2: Euclidean metric (alternative to falsify)
                    proposals.append(HiddenVariableProposal(
                        variable_type=VAR_GROUPED, variable_name=metric_name,
                        transform=TRANSFORM_METRIC,
                        rationale="t and x grouped - also try Euclidean (ct)^2+x^2 as alternative",
                        confidence=0.60, expression_patch=f"*{metric_name}",
                        expression_fragment="s2 = (c*t)^2 + x^2",
                    ))
                    if has_tau:
                        proposals.append(HiddenVariableProposal(
                            variable_type=VAR_GROUPED, variable_name=metric_name,
                            transform=TRANSFORM_METRIC,
                            rationale="t, x, tau grouped - try proper time interval (c tau)^2-x^2",
                            confidence=0.70, expression_patch=f"*{metric_name}",
                            expression_fragment="s2 = (c*tau)^2 - x^2",
                        ))
                else:
                    # No c known - propose scaled squared difference
                    proposals.append(HiddenVariableProposal(
                        variable_type=VAR_GROUPED, variable_name=metric_name,
                        transform=TRANSFORM_METRIC,
                        rationale="t and x co-vary - propose squared-difference metric between them",
                        confidence=0.70, expression_patch=f"*{metric_name}",
                        expression_fragment="s2 = t^2 - (x/c)^2",
                    ))

                # Proposal: try scaled forms
                proposals.append(HiddenVariableProposal(
                    variable_type=VAR_GROUPED, variable_name=metric_name,
                    transform=TRANSFORM_IDENTITY,
                    rationale="Grouped t, x - try product form as simpler candidate",
                    confidence=0.50, expression_patch=f"*{metric_name}",
                    expression_fragment="s2 = (c*t)^2 - x^2",
                ))

            # Sound wave grouping: f and lambda group as f*lambda = v_sound
            has_f = any(q in quantity_names for q in ["f", "nu", "omega"])
            has_lambda_q = "lambda" in quantity_names
            if has_f and has_lambda_q:
                proposals.append(HiddenVariableProposal(
                    variable_type=VAR_GROUPED, variable_name="v_sound",
                    transform=TRANSFORM_IDENTITY,
                    rationale="Frequency and wavelength co-vary - propose wave speed group f*lambda",
                    confidence=0.78, expression_patch="*v_sound",
                    expression_fragment="alpha = 1/2",
                ))

            # Fluid flow grouping: pressure and velocity group (Bernoulli)
            has_P = "P" in quantity_names
            has_rho = "rho" in quantity_names
            if has_P and has_velocity:
                proposals.append(HiddenVariableProposal(
                    variable_type=VAR_GROUPED, variable_name="B_const",
                    transform=TRANSFORM_IDENTITY,
                    rationale="Pressure and velocity co-vary - propose Bernoulli invariant P + 0.5*rho*v^2",
                    confidence=0.75, expression_patch="*B_const",
                    expression_fragment="alpha = 1/2",
                ))

        # Generic fallback
        proposals.append(HiddenVariableProposal(
            variable_type=VAR_INTEGER, variable_name="n",
            transform=TRANSFORM_SQUARED,
            rationale="Generic hidden integer variable (fallback)",
            confidence=0.40, expression_patch="*n^2",
            expression_fragment="n_scale = E/hbar",
        ))

        return proposals

    def _augment_observations(
        self, observations: list[Observation], proposal: HiddenVariableProposal,
        quantities: dict[str, Dimension],
    ) -> list[Observation]:
        aug_obs: list[Observation] = []
        for i, obs in enumerate(observations, start=1):
            new_params = dict(obs.parameters)
            new_timesteps = []
            for ts in obs.timesteps:
                new_ts = dict(ts)
                # If the variable already exists in the timestep, keep it
                if proposal.variable_name in ts or proposal.variable_name in obs.parameters:
                    new_timesteps.append(new_ts)
                    continue

                # Compute the variable value based on type
                if proposal.variable_type == VAR_INTEGER:
                    val = self._detect_integer_in_timestep(obs, i)
                    new_ts[proposal.variable_name] = float(val)
                elif proposal.variable_type == VAR_ANGULAR_M:
                    val = -(len(observations) // 2) + i - 1
                    new_ts[proposal.variable_name] = float(val)
                elif proposal.variable_type == VAR_HALF_INTEGER:
                    val = i / 2.0
                    new_ts[proposal.variable_name] = float(val)
                elif proposal.variable_type == VAR_SPIN:
                    val = 0.5
                    new_ts[proposal.variable_name] = float(val)
                elif proposal.variable_type == VAR_CONTINUOUS_RATIO:
                    # v2: Compute ratio from timestep data
                    val = self._compute_ratio_value(ts, obs, quantities, proposal)
                    new_ts[proposal.variable_name] = float(val)
                elif proposal.variable_type == VAR_CONTINUOUS_ADDITIVE:
                    # v2: Compute additive offset from timestep data
                    val = self._compute_additive_value(ts, obs, quantities, proposal)
                    new_ts[proposal.variable_name] = float(val)
                elif proposal.variable_type == VAR_GROUPED:
                    # v4: For grouped quantities, compute the metric invariant
                    val = self._compute_metric_value(ts, obs, quantities, proposal)
                    new_ts[proposal.variable_name] = float(val)
                else:
                    # Default: use observation index (continuous fallback)
                    new_ts[proposal.variable_name] = float(i)
                new_timesteps.append(new_ts)
            if not new_timesteps:
                new_timesteps = [dict(ts) for ts in obs.timesteps]
            aug_obs.append(Observation(
                id=obs.id, name=obs.name, description=obs.description,
                quantities={k: str(v) for k, v in quantities.items()},
                parameters=new_params, timesteps=new_timesteps,
                known_invariant=obs.known_invariant, lean_theorem=obs.lean_theorem,
                external_forces=obs.external_forces, phase_regions=obs.phase_regions,
                is_conservative=obs.is_conservative,
            ))
        return aug_obs

    def _compute_ratio_value(
        self, ts: dict, obs: Observation, quantities: dict[str, Dimension],
        proposal: HiddenVariableProposal,
    ) -> float:
        """Compute a continuous ratio variable from timestep data.

        For relativistic gamma: gamma = 1/sqrt(1-v^2/c^2)
        If both v and c are available, compute from them.
        For friction coefficient: mu = F/N
        For simple cases: use the value of the first available quantity.
        """
        # Try to find gamma/c explicitly in obs parameters
        if proposal.variable_name in obs.parameters:
            val = obs.parameters[proposal.variable_name]
            if isinstance(val, (int, float)):
                return float(val)

        # For gamma: check if v and c are available
        qnames = list(quantities.keys()) + list(obs.parameters.keys())
        ts_keys = set(ts.keys())

        # If gamma is requested, try computing from v and c
        if proposal.variable_name in ("gamma", "beta"):
            v_key = None
            c_key = None
            for k in ts_keys:
                if k in ("v", "v1", "beta"):
                    v_key = k
                if k == "c":
                    c_key = k
            if v_key and c_key:
                v_val = ts.get(v_key, 0.0)
                c_val = ts.get(c_key, 3e8)
                if isinstance(v_val, (int, float)) and isinstance(c_val, (int, float)) and abs(c_val) > 1e-10:
                    beta = float(v_val) / float(c_val)
                    if proposal.variable_name == "gamma" and abs(beta) < 1.0:
                        return 1.0 / math.sqrt(max(1e-10, 1.0 - beta ** 2))
                    return beta

        # For c (speed of light): check if it's in parameters
        if proposal.variable_name == "c":
            if "c" in obs.parameters:
                return float(obs.parameters["c"])
            return 3e8  # default

        # For mu (friction coefficient): compute from F and N
        if proposal.variable_name == "mu":
            F_val = ts.get("F", ts.get("friction", None))
            N_val = ts.get("N", ts.get("normal", None))
            if F_val is not None and N_val is not None and abs(N_val) > 1e-10:
                return float(F_val) / float(N_val)

        # Default: use parameter value or compute from first two quantities
        for qname in qnames:
            if qname in ts and isinstance(ts[qname], (int, float)) and qname != "t":
                return float(ts[qname])
        return 1.0

    def _compute_additive_value(
        self, ts: dict, obs: Observation, quantities: dict[str, Dimension],
        proposal: HiddenVariableProposal,
    ) -> float:
        """Compute a continuous additive variable (offset) from timestep data.

        For work function: phi is in parameters or can be computed.
        """
        # Check parameters first
        if proposal.variable_name in obs.parameters:
            val = obs.parameters[proposal.variable_name]
            if isinstance(val, (int, float)):
                return float(val)

        # For phi: check all timesteps
        if proposal.variable_name == "phi":
            if "phi" in ts:
                return float(ts["phi"])
            # Try to compute from K_max = h*f - phi → phi = h*f - K_max
            if "h" in obs.parameters and "f" in ts and "K_max" in ts:
                h_val = float(obs.parameters["h"])
                f_val = float(ts["f"])
                K_val = float(ts["K_max"])
                return h_val * f_val - K_val

        # Generic additive: find any constant in the data
        for key, val in ts.items():
            if key not in ("t",) and isinstance(val, (int, float)):
                if key not in quantities:
                    return float(val)

        return 1.0

    def _compute_metric_value(
        self, ts: dict, obs: Observation, quantities: dict[str, Dimension],
        proposal: HiddenVariableProposal,
    ) -> float:
        """v4: Compute a metric invariant from grouped quantities.

        For spacetime interval: s2 = (c*t)^2 - x^2
        For general metric: compute from the expression fragment.
        """
        expr_frag = proposal.expression_fragment
        ts_keys = set(ts.keys())
        param_keys = set(obs.parameters.keys())

        if "c*t" in expr_frag or "c*tau" in expr_frag:
            c_val = None
            t_val = None
            x_val = None
            tau_val = None

            if "c" in param_keys:
                c_val = float(obs.parameters["c"])
            elif "c" in ts_keys:
                c_val = float(ts["c"])

            for k in ts_keys:
                if k in ("t", "t_lab") and isinstance(ts[k], (int, float)):
                    t_val = float(ts[k])
                if k in ("x", "x_lab", "r") and isinstance(ts[k], (int, float)):
                    x_val = float(ts[k])
                if k == "tau" and isinstance(ts[k], (int, float)):
                    tau_val = float(ts[k])

            if c_val is None:
                c_val = 299792458.0

            if t_val is not None and x_val is not None:
                if "c*tau" in expr_frag and tau_val is not None:
                    return (c_val * tau_val) ** 2 - x_val ** 2
                elif "+" in expr_frag or "x^2 +" in expr_frag:
                    return (c_val * t_val) ** 2 + x_val ** 2
                else:
                    return (c_val * t_val) ** 2 - x_val ** 2

        vals: list[float] = []
        for k in ts_keys:
            if k in ("t",) or not isinstance(ts[k], (int, float)):
                continue
            vals.append(float(ts[k]))
        if len(vals) >= 2:
            return vals[0] ** 2 - vals[1] ** 2
        if vals:
            return vals[0]
        return 1.0

    @staticmethod
    def _detect_integer_in_timestep(obs: Observation, obs_index: int) -> int:
        for key in ["n", "n_i", "n_f", "level", "index"]:
            if key in obs.parameters:
                val = obs.parameters[key]
                if isinstance(val, (int, float)):
                    return int(val)
            for ts in obs.timesteps[:1]:
                if key in ts:
                    val = ts[key]
                    if isinstance(val, (int, float)):
                        return int(val)
                    if isinstance(val, dict) and "value" in val:
                        return int(val["value"])
        return obs_index

    def _collect_all_scored(
        self, quantities: dict[str, Dimension], observations: list[Observation],
        max_exprs: int = 500,
    ) -> dict[str, float]:
        scored: dict[str, float] = {}
        names = list(quantities.keys())
        ops = ["*", "/", "+", "-"]
        for i, a in enumerate(names):
            for b in names[i:]:
                for op in ops:
                    expr = f"{a}{op}{b}"
                    try:
                        s = sum(
                            self._evaluator.score(expr, obs) for obs in observations
                        ) / len(observations)
                        scored[expr] = s
                    except Exception:
                        pass
                    if len(scored) >= max_exprs:
                        return scored
        return scored


# =============================================================================
# 4. Training Data and Training Loop (v2 — expanded)
# =============================================================================

@dataclass
class HiddenVarTrainingExample:
    error_shape: str
    quantity_names: list[str]
    domain: str
    expected_var_type: str
    expected_transform: str
    expected_expression: str   # v3: the full expression template, e.g. "gamma = 1/sqrt(1 - beta^2)"
    description: str


def generate_synthetic_training_examples() -> list[HiddenVarTrainingExample]:
    """Generate ~150 synthetic training examples covering all 3 archetypes."""
    examples: list[HiddenVarTrainingExample] = []

    # ── Group A: Integer/quantum patterns (v1 — kept for compatibility, ~58) ──

    # Hydrogen spectrum variants (12)
    for i in range(12):
        qtys = [["E", "lambda"], ["E", "lambda", "hbar"],
                ["E", "lambda", "hbar", "omega"], ["E", "lambda", "c"]][i % 4]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_INVERSE_SQUARE, quantity_names=qtys,
            domain="quantum", expected_var_type=VAR_INTEGER,
            expected_transform=TRANSFORM_SQUARED,
            expected_expression="n = sqrt(hbar*c/(E*lambda))",
            description=f"H spectrum variant {i+1}: {', '.join(qtys)}",
        ))

    # Particle in box (8)
    for i in range(8):
        qtys_v = [["E", "L", "m"], ["E", "L", "hbar"], ["E", "m", "hbar"],
                  ["E", "L", "m", "hbar"], ["L", "m", "hbar"], ["E", "L"],
                  ["m", "L"], ["E", "m"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_QUADRATIC, quantity_names=qtys_v[i],
            domain="quantum", expected_var_type=VAR_INTEGER,
            expected_transform=TRANSFORM_SQUARED,
            expected_expression="n = L*sqrt(2*m*E)/(pi*hbar)",
            description=f"Particle in box variant {i+1}",
        ))

    # Harmonic oscillator (8) — half are integer, half are half-integer
    for i in range(4):
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_LINEAR, quantity_names=["E", "omega"],
            domain="quantum", expected_var_type=VAR_HALF_INTEGER,
            expected_transform=TRANSFORM_IDENTITY,
            expected_expression="j = (E/(hbar*omega)) - 0.5",
            description=f"QHO half-integer variant {i+1}",
        ))
    for i in range(4):
        qtys_v = [["E", "k", "m"], ["v", "omega", "x"], ["a", "omega", "t"], ["E", "t", "omega"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_LINEAR, quantity_names=qtys_v[i],
            domain="quantum", expected_var_type=VAR_INTEGER,
            expected_transform=TRANSFORM_IDENTITY,
            expected_expression="n_scale = E/hbar",
            description=f"Counting pattern variant {i+1}",
        ))

    # Zeeman splitting (6)
    for i in range(6):
        qtys_v = [["E", "B"], ["E", "B", "hbar"], ["E", "B", "m"],
                  ["delta_E", "B"], ["E", "B", "L"], ["omega", "B"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_PERIODIC, quantity_names=qtys_v[i],
            domain="quantum", expected_var_type=VAR_ANGULAR_M,
            expected_transform=TRANSFORM_IDENTITY,
            expected_expression="m_l = delta_E/(mu_B*B)",
            description=f"Zeeman variant {i+1}",
        ))

    # Spin measurements (6)
    for i in range(6):
        qtys_v = [["mu", "B"], ["E", "B", "mu"], ["mu", "S"],
                  ["omega", "B", "g"], ["E", "S"], ["mu", "B", "S"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_PERIODIC, quantity_names=qtys_v[i],
            domain="quantum", expected_var_type=VAR_SPIN,
            expected_transform=TRANSFORM_IDENTITY,
            expected_expression="s = (delta_E/(g*mu_B*B) - 1)/2",
            description=f"Spin variant {i+1}",
        ))

    # General counting patterns (8)
    domains = ["gravity", "spring", "em", "thermal"] * 2
    for i in range(8):
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_LINEAR,
            quantity_names=["F", "x"] if i % 2 == 0 else ["P", "V"],
            domain=domains[i], expected_var_type=VAR_INTEGER,
            expected_transform=TRANSFORM_IDENTITY,
            expected_expression="n_scale = E/hbar",
            description=f"Counting pattern variant {i+1}",
        ))

    # No hidden variable — random noise (6)
    for i in range(6):
        qtys_v = [["m", "g", "v"], ["P", "V", "T"], ["q", "E", "v"],
                  ["m", "k", "x"], ["F", "a", "m"], ["rho", "V", "m"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_RANDOM, quantity_names=qtys_v[i],
            domain=["gravity", "thermal", "em", "spring", "gravity", "thermal"][i],
            expected_var_type=VAR_CONTINUOUS, expected_transform=TRANSFORM_IDENTITY,
            expected_expression="alpha = 1/2",
            description=f"No hidden variable (random) variant {i+1}",
        ))

    # ── Group B: Continuous ratio (TYPE 1 — ~30 examples) ──

    # Friction: F = μN — 8 variants
    for i in range(8):
        qtys_v = [["F", "N"], ["F", "N", "m"], ["F", "N", "g"],
                  ["F", "N", "a"], ["F", "N", "mu"], ["F", "mu"],
                  ["N", "mu"], ["F", "N", "v"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_LINEAR_RATIO, quantity_names=qtys_v[i],
            domain="gravity", expected_var_type=VAR_CONTINUOUS_RATIO,
            expected_transform=TRANSFORM_RATIO,
            expected_expression="mu = F/N",
            description=f"Friction ratio variant {i+1}",
        ))

    # Drag: F = ½ρv²CA — 6 variants
    for i in range(6):
        qtys_v = [["F", "rho", "v", "A"], ["F", "rho", "v"], ["F", "v", "A"],
                  ["F", "rho", "A"], ["F", "v", "C_d"], ["F", "rho", "v", "C_d"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_LINEAR_RATIO, quantity_names=qtys_v[i],
            domain="gravity", expected_var_type=VAR_CONTINUOUS_RATIO,
            expected_transform=TRANSFORM_RATIO,
            expected_expression="C_d = 2*F/(rho*A*v^2)",
            description=f"Drag coefficient variant {i+1}",
        ))

    # Refraction: n = c/v — 6 variants
    for i in range(6):
        qtys_v = [["n_refr", "v"], ["n_refr", "c"], ["c", "v"],
                  ["theta", "n_refr"], ["n_refr", "v", "lambda"],
                  ["n_refr", "c", "v"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_LINEAR_RATIO, quantity_names=qtys_v[i],
            domain="em", expected_var_type=VAR_CONTINUOUS_RATIO,
            expected_transform=TRANSFORM_RATIO,
            expected_expression="n_refr = c/v",
            description=f"Refraction ratio variant {i+1}",
        ))

    # Pendulum: T ∝ √(L/g) — ratio with sqrt — 4 variants
    for i in range(4):
        qtys_v = [["T_period", "L"], ["T_period", "L", "g"],
                  ["T_period", "g"], ["T_period", "L", "m"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_LINEAR_RATIO, quantity_names=qtys_v[i],
            domain="gravity", expected_var_type=VAR_CONTINUOUS_RATIO,
            expected_transform=TRANSFORM_SQRT,
            expected_expression="T_factor = 2*pi*sqrt(L/g)",
            description=f"Pendulum ratio variant {i+1}",
        ))

    # Relativistic gamma ratio — 14 variants (v3: expanded for time dilation)
    for i in range(14):
        qtys_v = [["v", "c"], ["v", "c", "E"], ["v", "t"],
                  ["gamma", "v"], ["beta", "c"], ["E", "p"],
                  # v3 additions for time dilation / general relativistic
                  ["c", "t", "x"], ["c", "t", "v", "x"],
                  ["c", "tau", "x"], ["v", "c", "t", "x", "tau"],
                  ["c", "x", "t"], ["c", "v", "x"],
                  ["c", "t"], ["c", "v", "p"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_LINEAR_RATIO, quantity_names=qtys_v[i],
            domain="relativistic", expected_var_type=VAR_CONTINUOUS_RATIO,
            expected_transform=TRANSFORM_RATIO,
            expected_expression="gamma = 1/sqrt(1 - beta^2)",
            description=f"Relativistic gamma variant {i+1}",
        ))

    # ── Group C: Continuous additive (photoelectric-style, ~15 examples) ──

    # Photoelectric: K_max = hf - φ — 5 variants
    for i in range(5):
        qtys_v = [["K_max", "f"], ["K_max", "f", "h"],
                  ["K_max", "f", "nu"], ["K_max", "f", "E"],
                  ["K_max"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_LINEAR, quantity_names=qtys_v[i],
            domain="quantum", expected_var_type=VAR_CONTINUOUS_ADDITIVE,
            expected_transform=TRANSFORM_OFFSET,
            expected_expression="phi = h*f - K_max",
            description=f"Photoelectric offset variant {i+1}",
        ))

    # Work function / contact potential — 5 variants
    for i in range(5):
        qtys_v = [["V", "E"], ["W", "Q"], ["E", "phi"],
                  ["V", "f", "E"], ["W", "E", "f"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_LINEAR_RATIO, quantity_names=qtys_v[i],
            domain="em", expected_var_type=VAR_CONTINUOUS_ADDITIVE,
            expected_transform=TRANSFORM_OFFSET,
            expected_expression="phi = h*f - K_max",
            description=f"Work function variant {i+1}",
        ))

    # Thermal offset (heat loss) — 5 variants
    for i in range(5):
        qtys_v = [["Q", "T"], ["Q", "T", "W"], ["E", "T"],
                  ["Q", "m", "T"], ["Q", "T", "c"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_LINEAR, quantity_names=qtys_v[i],
            domain="thermal", expected_var_type=VAR_CONTINUOUS_ADDITIVE,
            expected_transform=TRANSFORM_OFFSET,
            expected_expression="eta = 1 - Q_out/Q_in",
            description=f"Thermal offset variant {i+1}",
        ))

    # ── Group D: Power law / exponent (TYPE 2 — ~30 examples) ──

    # Kepler: T² ∝ a³ — 8 variants
    for i in range(8):
        qtys_v = [["T_period", "a"], ["T_period", "a", "m"],
                  ["T_period", "a", "M"], ["T_period"],
                  ["a", "T_period"], ["T_period", "a", "r"],
                  ["T_period", "a", "v"], ["omega", "a"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_POWER_LAW, quantity_names=qtys_v[i],
            domain="gravity", expected_var_type=VAR_CONTINUOUS_RATIO,
            expected_transform=TRANSFORM_IDENTITY,
            expected_expression="k_kepler = T^2/a^3",
            description=f"Kepler power-law variant {i+1}",
        ))

    # Ideal gas: PV/T = const — 8 variants
    for i in range(8):
        qtys_v = [["P", "V", "T"], ["P", "V"], ["P", "T"],
                  ["V", "T"], ["P", "V", "T", "n"],
                  ["P", "V", "R"], ["P", "V", "T", "R"],
                  ["P", "V", "n", "R"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_MULTI_VAR, quantity_names=qtys_v[i],
            domain="thermal", expected_var_type=VAR_CONTINUOUS_RATIO,
            expected_transform=TRANSFORM_RATIO,
            expected_expression="R_gas = P*V/(n*T)",
            description=f"Ideal gas multi-var variant {i+1}",
        ))

    # Spring energy: E ∝ x² — 7 variants
    for i in range(7):
        qtys_v = [["E", "x"], ["E", "x", "k"], ["F", "x"],
                  ["E", "x", "m"], ["E", "x", "v"],
                  ["E", "k", "x"], ["E", "k"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_POWER_LAW, quantity_names=qtys_v[i],
            domain="spring", expected_var_type=VAR_CONTINUOUS_RATIO,
            expected_transform=TRANSFORM_SQUARED,
            expected_expression="k_spring = 2*E/x^2",
            description=f"Spring energy power-law variant {i+1}",
        ))

    # Inverse square law (gravity/EM) — 7 variants
    for i in range(7):
        qtys_v = [["F", "r"], ["F", "r", "m1", "m2"],
                  ["F", "r", "q1", "q2"], ["E", "r"],
                  ["F", "r", "G"], ["F", "r", "m"], ["E", "r", "lambda"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_POWER_LAW, quantity_names=qtys_v[i],
            domain="gravity", expected_var_type=VAR_CONTINUOUS_RATIO,
            expected_transform=TRANSFORM_IDENTITY,
            expected_expression="G_invsq = F*r^2/(m1*m2)",
            description=f"Inverse square power-law variant {i+1}",
        ))

    # ── Group E: Multi-variable (TYPE 3 — ~20 examples) ──

    # Ideal gas group — 6 more explicit multi-var
    for i in range(6):
        qtys_v = [["P", "V", "T", "n"], ["P", "V", "T"],
                  ["V", "T", "n"], ["P", "T", "n"],
                  ["P", "V", "n"], ["P", "V", "T", "n", "R"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_MULTI_VAR, quantity_names=qtys_v[i],
            domain="thermal", expected_var_type=VAR_CONTINUOUS_RATIO,
            expected_transform=TRANSFORM_RATIO,
            expected_expression="R_gas = P*V/(n*T)",
            description=f"Ideal gas group variant {i+7}",
        ))

    # Coupled oscillators — 4 variants
    for i in range(4):
        qtys_v = [["x1", "x2", "v1", "v2"], ["x1", "x2"],
                  ["x1", "v1", "x2", "v2", "k1", "k2"],
                  ["E1", "E2", "x1", "x2"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_MULTI_VAR, quantity_names=qtys_v[i],
            domain="spring", expected_var_type=VAR_CONTINUOUS_RATIO,
            expected_transform=TRANSFORM_IDENTITY,
            expected_expression="alpha = 1/2",
            description=f"Coupled oscillator variant {i+1}",
        ))

    # Heat engines — 5 variants
    for i in range(5):
        qtys_v = [["Q_in", "Q_out", "W_out"], ["Q_in", "W_out"],
                  ["Q_in", "Q_out"], ["W_out", "Q_out", "T_hot", "T_cold"],
                  ["eta", "Q_in", "W_out"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_MULTI_VAR, quantity_names=qtys_v[i],
            domain="thermal", expected_var_type=VAR_CONTINUOUS_ADDITIVE,
            expected_transform=TRANSFORM_OFFSET,
            expected_expression="Q_loss = Q_in - W_out",
            description=f"Heat engine variant {i+1}",
        ))

    # Multi-body momentum — 5 variants
    for i in range(5):
        qtys_v = [["m1", "v1", "m2", "v2"], ["m1", "v1", "m2"],
                  ["m1", "m2", "v1", "v2", "p"], ["p1", "p2"],
                  ["m1", "v1", "m2", "v2", "E"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_MULTI_VAR, quantity_names=qtys_v[i],
            domain="gravity", expected_var_type=VAR_CONTINUOUS_RATIO,
            expected_transform=TRANSFORM_IDENTITY,
            expected_expression="alpha = 1/2",
            description=f"Multi-body variant {i+1}",
        ))

    # === Group F: Grouped quantity / metric (TYPE 4 - ~50 examples) ===

    # Spacetime metric: t and x grouped with c - 15 variants
    for i in range(15):
        qtys_v = [
            ["c", "t", "x"], ["c", "t", "x", "tau"], ["c", "t", "x", "v"],
            ["c", "t", "x", "gamma"], ["c", "t", "x", "p"], ["c", "t", "r"],
            ["c", "t", "x", "tau", "gamma"], ["c", "x", "t_lab", "x_lab"],
            ["c", "t", "x", "E"], ["c", "t", "x", "s2"],
            ["c", "tau", "x"], ["c", "t", "x", "ds2"],
            ["c", "t", "x", "y", "z"], ["c", "t", "x_lab"],
            ["c", "t", "x", "tau", "v"],
        ]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_GROUPED, quantity_names=qtys_v[i],
            domain="relativistic", expected_var_type=VAR_GROUPED,
            expected_transform=TRANSFORM_METRIC,
            expected_expression="s2 = (c*t)^2 - x^2",
            description=f"Spacetime metric grouped variant {i+1}",
        ))

    # Sound wave grouping: f*lambda = v_sound - 8 variants
    for i in range(8):
        qtys_v = [
            ["f", "lambda"], ["f", "lambda", "v"], ["nu", "lambda"],
            ["f", "lambda", "c"], ["omega", "lambda"], ["f", "lambda", "T_period"],
            ["f", "lambda", "k"], ["nu", "lambda", "v"],
        ]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_GROUPED, quantity_names=qtys_v[i],
            domain="em", expected_var_type=VAR_GROUPED,
            expected_transform=TRANSFORM_IDENTITY,
            expected_expression="alpha = 1/2",
            description=f"Sound wave grouped variant {i+1}",
        ))

    # Fluid flow (Bernoulli) grouping: P and v co-vary - 8 variants
    for i in range(8):
        qtys_v = [
            ["P", "v", "rho"], ["P", "v", "rho", "h"],
            ["P", "v", "rho", "g"], ["P", "v", "rho", "A"],
            ["P", "v"], ["P", "v", "h", "rho", "g"],
            ["P", "v", "h"], ["P", "rho", "v"],
        ]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_GROUPED, quantity_names=qtys_v[i],
            domain="gravity", expected_var_type=VAR_GROUPED,
            expected_transform=TRANSFORM_IDENTITY,
            expected_expression="alpha = 1/2",
            description=f"Bernoulli grouped variant {i+1}",
        ))

    # Planetary orbit grouping: r and theta co-vary - 8 variants
    for i in range(8):
        qtys_v = [
            ["r", "theta"], ["r", "theta", "omega"],
            ["r", "theta", "v"], ["r", "theta", "L"],
            ["r", "theta", "m"], ["r", "theta", "a"],
            ["r", "theta", "v", "omega"], ["r", "drdt", "dphidt"],
        ]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_GROUPED, quantity_names=qtys_v[i],
            domain="gravity", expected_var_type=VAR_GROUPED,
            expected_transform=TRANSFORM_IDENTITY,
            expected_expression="alpha = 1/2",
            description=f"Orbital grouped variant {i+1}",
        ))

    # Coupled oscillator grouping: x1 and x2 co-vary as normal modes - 8 variants
    for i in range(8):
        qtys_v = [
            ["x1", "x2"], ["x1", "x2", "v1", "v2"],
            ["x1", "x2", "k"], ["x1", "x2", "m1", "m2"],
            ["x1", "x2", "omega1", "omega2"], ["x1", "x2", "E"],
            ["x1", "x2", "k1", "k2"], ["x1", "x2", "v1", "v2", "m1", "m2"],
        ]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_GROUPED, quantity_names=qtys_v[i],
            domain="spring", expected_var_type=VAR_GROUPED,
            expected_transform=TRANSFORM_IDENTITY,
            expected_expression="alpha = 1/2",
            description=f"Coupled oscillator grouped variant {i+1}",
        ))

    return examples


# v3: Map (error_shape, var_type, domain) → expression template for training data
_EXPR_TRAINING_MAP: dict[tuple[str, str, str], str] = {
    # Quantum integer (inverse square)
    (SHAPE_INVERSE_SQUARE, VAR_INTEGER, "quantum"): "n = sqrt(hbar*c/(E*lambda))",
    # Quantum integer (quadratic - particle in box)
    (SHAPE_QUADRATIC, VAR_INTEGER, "quantum"): "n = L*sqrt(2*m*E)/(pi*hbar)",
    # Quantum half-integer (harmonic oscillator)
    (SHAPE_LINEAR, VAR_HALF_INTEGER, "quantum"): "j = (E/(hbar*omega)) - 0.5",
    # Linear counting patterns
    (SHAPE_LINEAR, VAR_INTEGER, "quantum"): "n_scale = E/hbar",
    (SHAPE_LINEAR, VAR_INTEGER, "gravity"): "n_scale = E/hbar",
    (SHAPE_LINEAR, VAR_INTEGER, "spring"): "n_scale = E/hbar",
    (SHAPE_LINEAR, VAR_INTEGER, "em"): "n_scale = E/hbar",
    (SHAPE_LINEAR, VAR_INTEGER, "thermal"): "n_scale = E/hbar",
    # Angular momentum (Zeeman)
    (SHAPE_PERIODIC, VAR_ANGULAR_M, "quantum"): "m_l = delta_E/(mu_B*B)",
    # Spin
    (SHAPE_PERIODIC, VAR_SPIN, "quantum"): "s = (delta_E/(g*mu_B*B) - 1)/2",
    # Random noise = no hidden variable
    (SHAPE_RANDOM, VAR_CONTINUOUS, "*"): "none",
    # Continuous ratio — friction (gravity domain)
    (SHAPE_LINEAR_RATIO, VAR_CONTINUOUS_RATIO, "gravity"): "mu = F/N",
    # Continuous ratio — refraction (em domain)
    (SHAPE_LINEAR_RATIO, VAR_CONTINUOUS_RATIO, "em"): "n_refr = c/v",
    # Continuous ratio — relativistic gamma
    (SHAPE_LINEAR_RATIO, VAR_CONTINUOUS_RATIO, "relativistic"): "gamma = 1/sqrt(1 - beta^2)",
    # Continuous additive — photoelectric work function
    (SHAPE_LINEAR, VAR_CONTINUOUS_ADDITIVE, "quantum"): "phi = h*f - K_max",
    (SHAPE_LINEAR_RATIO, VAR_CONTINUOUS_ADDITIVE, "em"): "phi = h*f - K_max",
    # Continuous additive — thermal offset
    (SHAPE_LINEAR, VAR_CONTINUOUS_ADDITIVE, "thermal"): "Q_loss = Q_in - W_out",
    # Power law — Kepler
    (SHAPE_POWER_LAW, VAR_CONTINUOUS_RATIO, "gravity"): "k_kepler = T^2/a^3",
    # Power law — ideal gas
    (SHAPE_MULTI_VAR, VAR_CONTINUOUS_RATIO, "thermal"): "R_gas = P*V/(n*T)",
    # Power law — spring energy
    (SHAPE_POWER_LAW, VAR_CONTINUOUS_RATIO, "spring"): "k_spring = 2*E/x^2",
    # Multi-variable — coupled oscillators (no clear expression)
    (SHAPE_MULTI_VAR, VAR_CONTINUOUS_RATIO, "spring"): "none",
    # Multi-variable — heat engine
    (SHAPE_MULTI_VAR, VAR_CONTINUOUS_ADDITIVE, "thermal"): "eta = 1 - Q_out/Q_in",
    # Multi-variable — momentum conservation
    (SHAPE_MULTI_VAR, VAR_CONTINUOUS_RATIO, "gravity"): "alpha = 1/2",
    # v4: Grouped quantity — spacetime metric
    (SHAPE_GROUPED, VAR_GROUPED, "relativistic"): "s2 = (c*t)^2 - x^2",
    (SHAPE_GROUPED, VAR_GROUPED, "em"): "s2 = (c*t)^2 - x^2",
    (SHAPE_GROUPED, VAR_GROUPED, "gravity"): "s2 = (c*t)^2 - x^2",
    (SHAPE_GROUPED, VAR_GROUPED, "spring"): "s2 = (c*t)^2 - x^2",
    (SHAPE_GROUPED, VAR_GROUPED, "*"): "s2 = (c*t)^2 - x^2",
    (SHAPE_GROUPED, VAR_GROUPED, "thermal"): "s2 = (c*t)^2 - x^2",
    # Fallback
    (SHAPE_POWER_LAW, VAR_CONTINUOUS_RATIO, "*"): "alpha = log(E/E0)/log(x)",
}


def _assign_expression_templates(examples: list[HiddenVarTrainingExample]) -> None:
    """v3: Assign expected_expression to each training example based on archetype."""
    for ex in examples:
        key = (ex.error_shape, ex.expected_var_type, ex.domain)
        wc_key = (ex.error_shape, ex.expected_var_type, "*")
        if key in _EXPR_TRAINING_MAP:
            ex.expected_expression = _EXPR_TRAINING_MAP[key]
        elif wc_key in _EXPR_TRAINING_MAP:
            ex.expected_expression = _EXPR_TRAINING_MAP[wc_key]
        else:
            ex.expected_expression = "none"


def build_training_batch(
    examples: list[HiddenVarTrainingExample],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(examples)
    input_dim = NUM_SHAPES + NUM_HV_QUANTITIES + NUM_HV_DOMAINS
    output_dim = NUM_VAR_TYPES + NUM_TRANSFORMS + 1 + NUM_EXPR_TEMPLATES
    inputs = torch.zeros(batch_size, input_dim)
    targets = torch.zeros(batch_size, output_dim)

    for i, ex in enumerate(examples):
        inputs[i, :NUM_SHAPES] = shape_to_encoding(ex.error_shape, soft=True)
        offset = NUM_SHAPES
        for qname in ex.quantity_names:
            if qname in HV_QTY_TO_IDX:
                inputs[i, offset + HV_QTY_TO_IDX[qname]] = 1.0
        offset = NUM_SHAPES + NUM_HV_QUANTITIES
        if ex.domain in HV_DOMAIN_TO_IDX:
            inputs[i, offset + HV_DOMAIN_TO_IDX[ex.domain]] = 1.0
        if ex.expected_var_type in VAR_TYPE_TO_IDX:
            targets[i, VAR_TYPE_TO_IDX[ex.expected_var_type]] = 1.0
        if ex.expected_transform in TRANSFORM_TO_IDX:
            targets[i, NUM_VAR_TYPES + TRANSFORM_TO_IDX[ex.expected_transform]] = 1.0
        # v3: expression template target
        exp_tmpl = getattr(ex, 'expected_expression', 'none') or 'none'
        if exp_tmpl in EXPR_TEMPLATE_TO_IDX:
            targets[i, NUM_VAR_TYPES + NUM_TRANSFORMS + 1 + EXPR_TEMPLATE_TO_IDX[exp_tmpl]] = 1.0
        targets[i, NUM_VAR_TYPES + NUM_TRANSFORMS] = 1.0  # confidence target

    return inputs, targets


def train_hidden_var_proposer(
    *,
    proposer: HiddenVariableProposer | None = None,
    epochs: int = 300,
    lr: float = 0.003,
    device: str = "cpu",
    checkpoint_path: str | None = None,
) -> HiddenVariableProposer:
    if proposer is None:
        proposer = HiddenVariableProposer()
    proposer.to(device)
    proposer.train()

    examples = generate_synthetic_training_examples()
    _assign_expression_templates(examples)  # v3: add expression template targets
    print(f"  Generated {len(examples)} training examples")
    inputs, targets = build_training_batch(examples)
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer = torch.optim.Adam(proposer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    var_loss_fn = nn.CrossEntropyLoss()
    transform_loss_fn = nn.CrossEntropyLoss()
    conf_loss_fn = nn.BCEWithLogitsLoss()
    expr_loss_fn = nn.CrossEntropyLoss()  # v3: expression template loss

    best_loss = float('inf')
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = proposer(inputs)
        var_logits = output[:, :NUM_VAR_TYPES]
        transform_logits = output[:, NUM_VAR_TYPES:NUM_VAR_TYPES + NUM_TRANSFORMS]
        conf_logits = output[:, NUM_VAR_TYPES + NUM_TRANSFORMS]
        expr_logits = output[:, NUM_VAR_TYPES + NUM_TRANSFORMS + 1:]

        var_targets = targets[:, :NUM_VAR_TYPES].argmax(dim=-1)
        transform_targets = targets[:, NUM_VAR_TYPES:NUM_VAR_TYPES + NUM_TRANSFORMS].argmax(dim=-1)
        conf_targets = targets[:, NUM_VAR_TYPES + NUM_TRANSFORMS]
        expr_targets = targets[:, NUM_VAR_TYPES + NUM_TRANSFORMS + 1:].argmax(dim=-1)

        loss = (var_loss_fn(var_logits, var_targets)
                + transform_loss_fn(transform_logits, transform_targets)
                + 0.1 * conf_loss_fn(conf_logits, conf_targets)
                + 0.5 * expr_loss_fn(expr_logits, expr_targets))
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                var_acc = (var_logits.argmax(-1) == var_targets).float().mean()
                transform_acc = (transform_logits.argmax(-1) == transform_targets).float().mean()
                expr_acc = (expr_logits.argmax(-1) == expr_targets).float().mean()
            print(f"  epoch {epoch+1}/{epochs}  loss={loss.item():.4f}  "
                  f"var_acc={var_acc.item():.3f}  transform_acc={transform_acc.item():.3f}  "
                  f"expr_acc={expr_acc.item():.3f}")

    proposer.eval()
    print(f"  Training complete. Best loss={best_loss:.4f}")

    if checkpoint_path:
        save_path = Path(checkpoint_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": proposer.state_dict(),
            "num_shapes": NUM_SHAPES, "num_var_types": NUM_VAR_TYPES,
            "num_transforms": NUM_TRANSFORMS, "num_hv_domains": NUM_HV_DOMAINS,
            "num_hv_quantities": NUM_HV_QUANTITIES,
            "num_expr_templates": NUM_EXPR_TEMPLATES,  # v3
            "shape_to_idx": SHAPE_TO_IDX, "var_type_to_idx": VAR_TYPE_TO_IDX,
            "transform_to_idx": TRANSFORM_TO_IDX, "domain_to_idx": HV_DOMAIN_TO_IDX,
            "qty_to_idx": HV_QTY_TO_IDX, "expr_template_to_idx": EXPR_TEMPLATE_TO_IDX,
            "version": "v3",
        }, save_path)
        print(f"  Saved checkpoint to {save_path}")
    return proposer


def load_hidden_var_proposer(checkpoint_path: str, device: str = "cpu") -> HiddenVariableProposer:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    version = checkpoint.get("version", "v1")

    proposer = HiddenVariableProposer()
    if version == "v1":
        _load_v1_checkpoint(proposer, checkpoint)
    elif version == "v2":
        _load_v2_to_v3_checkpoint(proposer, checkpoint)
    else:
        proposer.load_state_dict(checkpoint["model_state_dict"])

    proposer.to(device)
    proposer.eval()
    return proposer


def _load_v2_to_v3_checkpoint(proposer: HiddenVariableProposer, checkpoint: dict) -> None:
    """Load a v2 checkpoint into a v3 model with expression template dimension remapping.

    v2: output = NUM_VAR_TYPES(7) + NUM_TRANSFORMS(6) + 1 = 14
    v3: output = 14 + NUM_EXPR_TEMPLATES(22) = 36
    """
    v2_state = checkpoint["model_state_dict"]
    v3_state = proposer.state_dict()

    # fc1 and fc2 unchanged — copy directly
    v3_state["fc1.weight"].copy_(v2_state["fc1.weight"])
    v3_state["fc1.bias"].copy_(v2_state["fc1.bias"])
    v3_state["fc2.weight"].copy_(v2_state["fc2.weight"])
    v3_state["fc2.bias"].copy_(v2_state["fc2.bias"])

    # fc3: v2 [14, 32] → v3 [36, 32]
    v3_state["fc3.weight"].zero_()
    v3_state["fc3.weight"][:14, :] = v2_state["fc3.weight"]
    v3_state["fc3.bias"].zero_()
    v3_state["fc3.bias"][:14] = v2_state["fc3.bias"]

    proposer.load_state_dict(v3_state)


def _load_v1_checkpoint(proposer: HiddenVariableProposer, checkpoint: dict) -> None:
    """Load a v1 checkpoint into a v2 model with dimension remapping.

    v1: 7 shapes, 5 var types, 4 transforms, 7 domains, 33 quantities
    v2: 10 shapes, 7 var types, 6 transforms, 7 domains, 52 quantities
    """
    v1_state = checkpoint["model_state_dict"]
    v2_state = proposer.state_dict()

    # Remap fc1.weight: v1 (32, 47) → v2 (64, 69)
    # Strategy: copy v1 weights to top-left corner, zero-init new dimensions
    v1_input_dim = 7 + 33 + 7  # shapes + quantities + domains = 47
    v2_input_dim = NUM_SHAPES + NUM_HV_QUANTITIES + NUM_HV_DOMAINS  # 69
    v1_hidden = 32
    v2_hidden = 64

    # fc1.weight: v1 [32, 47] → v2 [64, 69]
    v2_state["fc1.weight"].zero_()
    v2_state["fc1.weight"][:v1_hidden, :v1_input_dim] = v1_state["fc1.weight"]
    # fc1.bias: v1 [32] → v2 [64]
    v2_state["fc1.bias"].zero_()
    v2_state["fc1.bias"][:v1_hidden] = v1_state["fc1.bias"]

    # fc2.weight: v1 [32, 32] → v2 [32, 64] — just the bottom half
    v2_state["fc2.weight"].zero_()
    v2_state["fc2.weight"][:v1_hidden, :v1_hidden] = v1_state["fc2.weight"]
    # fc2.bias: keep same
    v2_state["fc2.bias"].zero_()
    v2_state["fc2.bias"][:v1_hidden] = v1_state["fc2.bias"]

    # fc3.weight: v1 [10, 32] → v2 [14, 32]
    v1_output_dim = 5 + 4 + 1  # var types + transforms + confidence = 10
    v2_output_dim = NUM_VAR_TYPES + NUM_TRANSFORMS + 1  # 14
    v2_state["fc3.weight"].zero_()
    v2_state["fc3.weight"][:v1_output_dim, :v1_hidden] = v1_state["fc3.weight"]
    # fc3.bias: v1 [10] → v2 [14]
    v2_state["fc3.bias"].zero_()
    v2_state["fc3.bias"][:v1_output_dim] = v1_state["fc3.bias"]

    proposer.load_state_dict(v2_state)


# =============================================================================
# 5. High-Level Pipeline (unchanged from v1)
# =============================================================================

def run_discovery_pipeline(
    quantity_dict: dict[str, Dimension],
    observations: list[Observation],
    *,
    domain: str = "unknown",
    proposer: HiddenVariableProposer | None = None,
    max_proposals: int = 5,
    discovery_threshold: float = 0.95,
) -> DiscoveryResult:
    from src.physics.search import ExpressionSearch

    def beam_search_wrapper(quantities, obs):
        search = ExpressionSearch(
            quantities=quantities, train_observations=obs,
            max_depth=10, max_expansions=20000,
            discovery_threshold=discovery_threshold,
        )
        search.run()
        return search

    discovery = HiddenVariableDiscovery(
        proposer=proposer, max_proposals=max_proposals,
        discovery_threshold=discovery_threshold,
    )
    return discovery.discover(
        quantities=quantity_dict, observations=observations,
        beam_search_fn=beam_search_wrapper, domain=domain,
        quantity_names=list(quantity_dict.keys()),
    )


def save_discovery_results(results: list[DiscoveryResult], path: str) -> None:
    output = []
    for r in results:
        output.append({
            "discovered": r.discovered,
            "hidden_variable": r.hidden_variable,
            "transform": r.transform,
            "best_expression": r.best_expression,
            "best_score": r.best_score,
            "baseline_score": r.baseline_score,
            "num_proposals_tried": r.num_proposals_tried,
            "error_shape": r.error_analysis.shape if r.error_analysis else None,
            "error_confidence": r.error_analysis.shape_confidence if r.error_analysis else None,
            "metadata": r.metadata,
        })
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(output)} discovery results to {save_path}")


# =============================================================================
# 6. Unsupervised Pattern Detector — Autoencoder for residual structure
# =============================================================================

# Residual curve length for padding/truncation
RESIDUAL_CURVE_LENGTH = 32


class ResidualAutoencoder(nn.Module):
    """1D convolutional autoencoder that compresses residual curves.

    Trained on mixed structured+noise data WITHOUT labels. The bottleneck
    forces the model to learn efficient representations; structured patterns
    are more compressible → lower reconstruction error → higher structure score.
    """

    def __init__(
        self,
        input_dim: int = RESIDUAL_CURVE_LENGTH,
        hidden_dim: int = 64,
        bottleneck_dim: int = 8,
        *,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        # Encoder: input_dim → hidden → hidden//2 → bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, bottleneck_dim),
        )

        # Decoder: bottleneck → hidden//2 → hidden → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (reconstruction, latent)."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error (batched)."""
        x_hat, _ = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=-1)

    def structuredness_score(self, x: torch.Tensor) -> torch.Tensor:
        """Higher = more structured. -reconstruction_error normalized."""
        mse = self.reconstruction_error(x)
        return -mse

    def has_structure(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        return self.structuredness_score(x) > threshold

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UnsupervisedPatternDetector:
    """Detect hidden structure in residuals WITHOUT labeled archetypes.

    Trains an autoencoder on mixed structured+noise residual curves.
    Structured patterns compress better through the bottleneck → lower
    reconstruction error → higher structuredness score.

    The detector never sees labels about which pattern type exists — it
    learns to separate structure from noise purely from compressibility.
    """

    def __init__(
        self,
        *,
        input_dim: int = RESIDUAL_CURVE_LENGTH,
        hidden_dim: int = 64,
        bottleneck_dim: int = 8,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.device = device
        self.model = ResidualAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        ).to(device)
        self._threshold: float | None = None
        self._trained: bool = False

    @property
    def threshold(self) -> float | None:
        return self._threshold

    def _residual_to_tensor(
        self, residual_values: list[float],
    ) -> torch.Tensor:
        """Pad/truncate residual values to fixed length and normalize."""
        n = len(residual_values)
        if n == 0:
            return torch.zeros(self.input_dim)

        # Normalize to zero mean, unit std (robust to scale differences)
        arr = torch.tensor(residual_values, dtype=torch.float32)
        mean = arr.mean()
        std = arr.std()
        if std < 1e-10:
            std = 1.0
        arr = (arr - mean) / std

        # Pad or truncate
        if n < self.input_dim:
            padded = torch.zeros(self.input_dim)
            padded[:n] = arr
            return padded
        elif n > self.input_dim:
            # Evenly sample
            indices = torch.linspace(0, n - 1, self.input_dim).long()
            return arr[indices]
        return arr

    def score_curve(self, residual_values: list[float]) -> float:
        """Score a single residual curve — higher = more structured."""
        tensor = self._residual_to_tensor(residual_values).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model.structuredness_score(tensor)
        return score.item()

    def score_curves(self, curves: list[list[float]]) -> list[float]:
        """Score multiple residual curves."""
        if not curves:
            return []
        batch = torch.stack([
            self._residual_to_tensor(c) for c in curves
        ]).to(self.device)
        with torch.no_grad():
            scores = self.model.structuredness_score(batch)
        return scores.cpu().tolist()

    def has_structure(self, residual_values: list[float]) -> bool:
        if self._threshold is None:
            return False
        return self.score_curve(residual_values) > self._threshold

    def set_threshold_from_scores(
        self,
        structured_scores: list[float],
        noise_scores: list[float],
    ) -> float:
        """Set binary threshold at midpoint of structured/noise medians."""
        s_med = float(torch.tensor(structured_scores).median())
        n_med = float(torch.tensor(noise_scores).median())
        self._threshold = (s_med + n_med) / 2.0
        return self._threshold

    def fit_threshold_percentile(
        self,
        noise_scores: list[float],
        percentile: float = 95.0,
    ) -> float:
        """Set threshold at noise percentile (e.g., 95th → 5% false positives)."""
        self._threshold = float(
            torch.tensor(noise_scores).quantile(percentile / 100.0)
        )
        return self._threshold

    def count_parameters(self) -> int:
        return self.model.count_parameters()


# =============================================================================
# 6b. Synthetic Residual Data Generation (Unsupervised)
# =============================================================================


def _generate_structured_residual(n_points: int, rng: random.Random) -> list[float]:
    """Generate one structured residual curve with a random hidden pattern.

    Patterns include: 1/n^2, n^2, linear, exponential, periodic, logarithmic,
    step function, polynomial, and combinations. The specific pattern type
    is NOT recorded — the detector must learn to recognize structure vs noise
    without labels.
    """
    x = torch.arange(1, n_points + 1, dtype=torch.float32)
    pattern_type = rng.randint(0, 9)

    if pattern_type == 0:  # 1/n^2 (inverse square)
        a = rng.uniform(1.0, 50.0)
        b = rng.uniform(-2.0, 2.0)
        y = a / (x ** 2) + b
    elif pattern_type == 1:  # n^2 (quadratic)
        a = rng.uniform(0.1, 2.0)
        b = rng.uniform(-1.0, 1.0)
        c = rng.uniform(-5.0, 5.0)
        y = a * x**2 + b * x + c
    elif pattern_type == 2:  # linear
        a = rng.uniform(-2.0, 2.0)
        b = rng.uniform(-5.0, 5.0)
        y = a * x + b
    elif pattern_type == 3:  # exponential
        a = rng.uniform(0.5, 3.0)
        rate = rng.uniform(-0.5, 0.5)
        y = a * torch.exp(rate * x)
    elif pattern_type == 4:  # periodic
        amp = rng.uniform(0.5, 4.0)
        freq = rng.uniform(0.1, 0.5)
        phase = rng.uniform(0.0, 2 * math.pi)
        offset = rng.uniform(-3.0, 3.0)
        y = amp * torch.sin(freq * x + phase) + offset
    elif pattern_type == 5:  # logarithmic
        a = rng.uniform(1.0, 5.0)
        b = rng.uniform(-3.0, 3.0)
        y = a * torch.log(x.float()) + b
    elif pattern_type == 6:  # step function
        y = torch.zeros(n_points)
        n_steps = rng.randint(2, 6)
        for _ in range(n_steps):
            pos = rng.randint(1, n_points - 1)
            val = rng.uniform(-5.0, 5.0)
            y[pos:] += val
    elif pattern_type == 7:  # polynomial (cubic or quartic)
        coeffs = [rng.uniform(-0.5, 0.5) for _ in range(4)]
        y = sum(c * x**i for i, c in enumerate(coeffs))
    else:  # 8 — ratio-scaled (a * n / (n + b))
        a = rng.uniform(1.0, 10.0)
        b = rng.uniform(1.0, 10.0)
        y = a * x / (x + b)

    # Add small structured noise
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    y_std = float(y_tensor.float().std().item()) if y_tensor.numel() > 0 else 1.0
    noise_std = rng.uniform(0.01, 0.15) * y_std
    y_final = y_tensor + torch.randn(int(n_points)) * max(noise_std, 0.01)

    return y_final.tolist()


def _generate_noise_residual(n_points: int, rng: random.Random) -> list[float]:
    """Generate pure noise residuals — various noise distributions."""
    noise_type = rng.randint(0, 3)
    if noise_type == 0:
        # Gaussian
        std = rng.uniform(0.5, 5.0)
        y = torch.randn(n_points) * std
    elif noise_type == 1:
        # Uniform
        scale = rng.uniform(1.0, 10.0)
        y = (torch.rand(n_points) - 0.5) * 2 * scale
    elif noise_type == 2:
        # Random walk (integrated noise)
        steps = torch.randn(n_points) * rng.uniform(0.3, 2.0)
        y = torch.cumsum(steps, dim=0)
    else:
        # Mixture of Gaussians
        std1 = rng.uniform(0.5, 3.0)
        std2 = rng.uniform(0.5, 3.0)
        mean2 = rng.uniform(-3.0, 3.0)
        mask = torch.rand(n_points) > rng.uniform(0.3, 0.7)
        y = torch.randn(n_points) * std1
        y[mask] = torch.randn(mask.sum().item()) * std2 + mean2
    return y.tolist()


def generate_unsupervised_residual_data(
    n_structured: int = 500,
    n_noise: int = 500,
    n_points: int = 50,
    *,
    seed: int = 42,
) -> tuple[list[list[float]], list[list[float]]]:
    """Generate mixed structured+noise residual curves WITHOUT labels.

    Returns (structured_curves, noise_curves). The caller is responsible
    for evaluation; during training both sets are mixed together.
    """
    rng = random.Random(seed)
    structured = [
        _generate_structured_residual(n_points, rng)
        for _ in range(n_structured)
    ]
    noise = [
        _generate_noise_residual(n_points, rng)
        for _ in range(n_noise)
    ]
    return structured, noise


# =============================================================================
# 6c. Unsupervised Training Loop
# =============================================================================


def train_unsupervised_detector(
    detector: UnsupervisedPatternDetector,
    structured_curves: list[list[float]],
    noise_curves: list[list[float]],
    *,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 0.001,
    device: str = "cpu",
    checkpoint_path: str | None = None,
) -> UnsupervisedPatternDetector:
    """Train the autoencoder on mixed structured+noise data (no labels).

    The model sees ALL data mixed together during training. It's never told
    which are structured and which are noise. The bottleneck forces it to
    learn efficient representations of patterns.
    """
    import random as _random

    detector.model.train()
    detector.model.to(device)

    # Combine all data
    all_curves = structured_curves + noise_curves
    all_tensors = [
        detector._residual_to_tensor(c) for c in all_curves
    ]
    dataset = torch.stack(all_tensors).to(device)
    n_samples = len(dataset)

    optimizer = torch.optim.Adam(detector.model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    indices = list(range(n_samples))
    rng = _random.Random(42)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] = {}

    for epoch in range(epochs):
        rng.shuffle(indices)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch = dataset[batch_idx]

            optimizer.zero_grad()
            x_hat, _ = detector.model(batch)
            loss = loss_fn(x_hat, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in detector.model.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            # Compute separation metric on training data
            with torch.no_grad():
                s_tensors = torch.stack([
                    detector._residual_to_tensor(c) for c in structured_curves[:200]
                ]).to(device)
                n_tensors = torch.stack([
                    detector._residual_to_tensor(c) for c in noise_curves[:200]
                ]).to(device)
                s_mse = detector.model.reconstruction_error(s_tensors).mean().item()
                n_mse = detector.model.reconstruction_error(n_tensors).mean().item()
                separation = n_mse - s_mse  # positive = structured compresses better
            print(
                f"  epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  "
                f"s_mse={s_mse:.4f}  n_mse={n_mse:.4f}  sep={separation:.4f}"
            )

    # Restore best state
    detector.model.load_state_dict(best_state)
    detector.model.eval()
    detector._trained = True

    if checkpoint_path:
        save_path = Path(checkpoint_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": detector.model.state_dict(),
            "input_dim": detector.input_dim,
            "threshold": detector._threshold,
            "config": {
                "hidden_dim": detector.model.encoder[0].out_features,
                "bottleneck_dim": detector.model.bottleneck_dim,
            },
        }, save_path)
        print(f"  Saved checkpoint to {save_path}")

    return detector


def load_unsupervised_detector(
    checkpoint_path: str,
    device: str = "cpu",
) -> UnsupervisedPatternDetector:
    """Load a trained unsupervised detector from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint.get("config", {})
    detector = UnsupervisedPatternDetector(
        input_dim=checkpoint.get("input_dim", RESIDUAL_CURVE_LENGTH),
        hidden_dim=config.get("hidden_dim", 64),
        bottleneck_dim=config.get("bottleneck_dim", 8),
        device=device,
    )
    detector.model.load_state_dict(checkpoint["model_state_dict"])
    detector.model.eval()
    detector._trained = True
    detector._threshold = checkpoint.get("threshold")
    return detector


# =============================================================================
# 6d. Evaluation Helpers
# =============================================================================


def evaluate_unsupervised_detector(
    detector: UnsupervisedPatternDetector,
    structured_curves: list[list[float]],
    noise_curves: list[list[float]],
) -> dict[str, float]:
    """Evaluate detector: ROC-AUC, separation, classification metrics.

    Computes ROC-AUC by treating structuredness_score as a continuous
    predictor of "has structure". Sets threshold at optimal Youden point.
    """
    s_scores = detector.score_curves(structured_curves)
    n_scores = detector.score_curves(noise_curves)

    # Set threshold automatically
    detector.set_threshold_from_scores(s_scores, n_scores)

    # Binary classification
    y_true = [1] * len(s_scores) + [0] * len(n_scores)
    y_scores = s_scores + n_scores

    # ROC-AUC via sorting (no sklearn dependency)
    auc = _compute_roc_auc(y_true, y_scores)

    # Classification at threshold
    threshold = detector._threshold
    tp = sum(1 for s in s_scores if s > threshold)
    fn = len(s_scores) - tp
    fp = sum(1 for s in n_scores if s > threshold)
    tn = len(n_scores) - fp

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-10, precision + recall)

    # Separation metrics
    s_mean = sum(s_scores) / max(1, len(s_scores))
    n_mean = sum(n_scores) / max(1, len(n_scores))
    separation = s_mean - n_mean

    return {
        "roc_auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "separation": separation,
        "structured_mean_score": s_mean,
        "noise_mean_score": n_mean,
        "threshold": threshold or 0.0,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def _compute_roc_auc(y_true: list[int], y_scores: list[float]) -> float:
    """Compute ROC-AUC via pairwise comparison (no sklearn)."""
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pairs = [(s, t) for s, t in zip(y_scores, y_true)]
    pos_scores = [s for s, t in pairs if t == 1]
    neg_scores = [s for s, t in pairs if t == 0]

    # Count concordant pairs
    concordant = 0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                concordant += 1
            elif abs(ps - ns) < 1e-12:
                concordant += 0.5

    return concordant / (n_pos * n_neg)


# =============================================================================
# Linear Algebra Helper
# =============================================================================

def _solve_linear_3x3(A: list[list[float]], B: list[float]) -> tuple[float, float, float]:
    det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
           - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
           + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
    if abs(det) < 1e-15:
        raise ValueError("Singular matrix")

    def cofactor(row: int, col: int) -> float:
        minor_rows = [r for r in range(3) if r != row]
        minor_cols = [c for c in range(3) if c != col]
        r0, r1 = minor_rows
        c0, c1 = minor_cols
        minor_det = A[r0][c0] * A[r1][c1] - A[r0][c1] * A[r1][c0]
        return minor_det if (row + col) % 2 == 0 else -minor_det

    x = (B[0] * cofactor(0, 0) + B[1] * cofactor(1, 0) + B[2] * cofactor(2, 0)) / det
    y = (B[0] * cofactor(0, 1) + B[1] * cofactor(1, 1) + B[2] * cofactor(2, 1)) / det
    z = (B[0] * cofactor(0, 2) + B[1] * cofactor(1, 2) + B[2] * cofactor(2, 2)) / det
    return x, y, z


# =============================================================================
# 7. Grouped Quantity Detection — co-varying quantity groups
# =============================================================================
# The system currently treats t and x as independent. Time dilation needs
# them seen as a GROUP. This module detects when quantities co-vary across
# observations and should be treated with a metric relationship between them.

# Metric types for grouped quantities
GROUPED_METRIC_SQUARED_DIFF = "squared_diff"   # q1² - q2² (spacetime-like)
GROUPED_METRIC_SUM_SQUARES = "sum_squares"      # q1² + q2² (Euclidean-like)
GROUPED_METRIC_PRODUCT = "product"              # q1 * q2 (conjugate pairs)
GROUPED_METRIC_RATIO = "ratio"                  # q1 / q2 (scaling relation)
GROUPED_METRIC_WEIGHTED_DIFF = "weighted_diff"  # (k*q1)² - q2² (spacetime with scale)

ALL_GROUPED_METRICS = [
    GROUPED_METRIC_SQUARED_DIFF, GROUPED_METRIC_SUM_SQUARES,
    GROUPED_METRIC_PRODUCT, GROUPED_METRIC_RATIO, GROUPED_METRIC_WEIGHTED_DIFF,
]
NUM_GROUPED_METRICS = len(ALL_GROUPED_METRICS)
GROUPED_METRIC_TO_IDX = {m: i for i, m in enumerate(ALL_GROUPED_METRICS)}
IDX_TO_GROUPED_METRIC = {i: m for m, i in GROUPED_METRIC_TO_IDX.items()}

# Pre-1905 quantity groups (what "group" training looks like pre-Einstein)
PRE1905_GROUP_PATTERNS = [
    # Sound waves: frequency and wavelength group as f*λ = v_sound
    ("f", "lambda", GROUPED_METRIC_PRODUCT, "Frequency-wavelength: f*λ = v_sound"),
    ("f", "lambda", GROUPED_METRIC_RATIO, "Frequency-wavelength inverse: f = v_sound/λ"),
    # Fluid flow: pressure and velocity group in Bernoulli
    ("P", "v", GROUPED_METRIC_SUM_SQUARES, "Bernoulli: P + ½ρv² = const"),
    ("P", "v", GROUPED_METRIC_SQUARED_DIFF, "Bernoulli alt: P - ½ρv² pattern"),
    # Planetary orbits: r and θ group for angular momentum
    ("r", "theta", GROUPED_METRIC_PRODUCT, "Angular momentum: r²*dθ/dt = const"),
    ("r", "theta", GROUPED_METRIC_SQUARED_DIFF, "Orbital: r² - (dθ/dt)² pattern"),
    # Coupled oscillators: x₁ and x₂ group as normal modes
    ("x1", "x2", GROUPED_METRIC_SQUARED_DIFF, "Normal mode: x₁² - x₂² = const"),
    ("x1", "x2", GROUPED_METRIC_SUM_SQUARES, "Normal mode: x₁² + x₂² = const"),
    # Galilean relativity: t and x are independent in classical physics
    ("t", "x", GROUPED_METRIC_PRODUCT, "Galilean: t*x (trivial group)"),
    ("t", "x", GROUPED_METRIC_RATIO, "Galilean: x/t = v (velocity)"),
    # Thermal: P, V group as PV = const (isothermal)
    ("P", "V", GROUPED_METRIC_PRODUCT, "Boyle: P*V = const"),
    # Ideal gas compression: P, T group
    ("P", "T", GROUPED_METRIC_RATIO, "Gay-Lussac: P/T = const"),
    # Mass and velocity group for momentum
    ("m", "v", GROUPED_METRIC_PRODUCT, "Momentum: m*v"),
    # Energy and velocity group for kinetic
    ("E", "v", GROUPED_METRIC_SQUARED_DIFF, "Kinetic: E - ½mv² = 0"),
    # Period and length for pendulum
    ("T_period", "L", GROUPED_METRIC_SQUARED_DIFF, "Pendulum: T²/L = 4π²/g"),
    # Spring: F and x group
    ("F", "x", GROUPED_METRIC_PRODUCT, "Hooke: F = k*x"),
    # Frequency and period
    ("f", "T_period", GROUPED_METRIC_PRODUCT, "Frequency-period: f*T = 1"),
    # Mass and acceleration for force
    ("m", "a", GROUPED_METRIC_PRODUCT, "Newton: F = m*a"),
]
NUM_PRE1905_GROUP_PATTERNS = len(PRE1905_GROUP_PATTERNS)

# Quantity vocabulary for the grouped detector — focused on classical physics
_GROUPED_QTY_VOCAB = [
    "t", "x", "y", "z", "v", "v1", "v2", "vx", "vy",
    "m", "m1", "m2", "F", "a", "g", "k",
    "E", "W", "Q", "P", "V", "T", "S", "n",
    "f", "lambda", "omega", "T_period", "L", "r", "theta",
    "x1", "x2", "x0", "v0", "c",
    "rho", "A", "p", "mu", "N",
]
GROUPED_QTY_TO_IDX = {q: i for i, q in enumerate(_GROUPED_QTY_VOCAB)}
NUM_GROUPED_QUANTITIES = len(_GROUPED_QTY_VOCAB)


# ── Grouped Quantity Feature encoding ────────────────────────────────────────

def _compute_pairwise_corr(
    q1_vals: list[float], q2_vals: list[float],
) -> float:
    """Pearson correlation between two quantity value sequences."""
    n = min(len(q1_vals), len(q2_vals))
    if n < 3:
        return 0.0
    v1 = q1_vals[:n]
    v2 = q2_vals[:n]
    m1 = sum(v1) / n
    m2 = sum(v2) / n
    cov = sum((v1[i] - m1) * (v2[i] - m2) for i in range(n))
    s1 = math.sqrt(sum((x - m1) ** 2 for x in v1))
    s2 = math.sqrt(sum((y - m2) ** 2 for y in v2))
    if s1 < 1e-12 or s2 < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, cov / (s1 * s2)))


def _compute_cv(values: list[float]) -> float:
    """Coefficient of variation (spread/mean) — measure of constancy."""
    n = len(values)
    if n < 2:
        return 1.0
    mean = sum(values) / n
    if abs(mean) < 1e-12:
        return 1.0
    var = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(max(var, 0.0)) / abs(mean)


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class GroupedQuantityDetection:
    """Result of detecting a co-varying quantity group."""
    group: list[str]                          # The quantities identified as grouped
    correlation_matrix: dict[tuple[str, str], float]  # pairwise correlations
    mean_abs_correlation: float               # how strongly they co-vary
    co_variation_pattern: str                 # "linear", "quadratic", "inverse", etc.
    pattern_confidence: float                 # confidence in the co-variation pattern
    across_observation_consistency: float     # does the relationship hold across observations?
    suggested_metrics: list[str]              # metric types likely relevant
    detection_source: str                     # "correlation", "residual_analysis", "mlp"


@dataclass
class MetricProposal:
    """A proposed metric relationship for a quantity group."""
    metric_type: str                          # squared_diff, sum_squares, product, ratio, weighted_diff
    quantities: list[str]                     # the grouped quantities
    expression_template: str                  # e.g., "(c*t)^2 - x^2"
    scale_factor: float | None = None         # for weighted_diff (the k in (k*q1)² - q2²)
    confidence: float = 0.0
    rationale: str = ""


@dataclass
class GroupedMetricDiscoveryResult:
    """Complete result of grouped metric discovery attempt."""
    discovered: bool
    detected_group: list[str]
    metric_type: str | None
    best_invariant: str | None
    best_constancy: float
    baseline_constancy: float
    proposals_tried: int
    search_candidates: list[str] = field(default_factory=list)
    candidate_scores: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# =============================================================================
# 8. GroupedQuantityDetector — rule-based + MLP hybrid
# =============================================================================

class GroupedQuantityDetector:
    """Detect when quantities co-vary across observations and should be grouped.

    Uses correlation analysis on observation timesteps to find pairs/groups
    of quantities that move together. The key insight: if q1 and q2 show strong
    correlation across multiple observations, they likely belong to a group
    that should be described by a metric relationship.

    This is ERA-GATED: trained only on pre-1905 patterns (sound waves,
    fluids, orbits, oscillators). Never taught Lorentz or spacetime metric.
    Yet when shown muon data, it should detect (t, x) as a group.
    """

    def __init__(self, *, correlation_threshold: float = 0.6) -> None:
        self.correlation_threshold = correlation_threshold

    def detect_groups(
        self,
        observations: list[Observation],
        quantity_names: list[str],
    ) -> list[GroupedQuantityDetection]:
        """Find all co-varying quantity groups across observations.

        Algorithm:
        1. For each pair of quantities, compute correlation across timesteps
        2. Cluster strongly-correlated pairs into groups
        3. For each group, determine co-variation pattern and propose metrics
        """
        if not observations or len(quantity_names) < 2:
            return []

        # Extract value sequences for each quantity
        qty_values: dict[str, list[float]] = {q: [] for q in quantity_names}
        for obs in observations:
            for qname in quantity_names:
                for ts in obs.timesteps:
                    val = ts.get(qname)
                    if isinstance(val, (int, float)):
                        qty_values[qname].append(float(val))

        # Filter to quantities with enough data
        valid_qtys = [q for q in quantity_names if len(qty_values[q]) >= 3]
        if len(valid_qtys) < 2:
            return []

        # Compute pairwise correlation matrix
        corr_matrix: dict[tuple[str, str], float] = {}
        for i, qa in enumerate(valid_qtys):
            for qb in valid_qtys[i + 1:]:
                corr = _compute_pairwise_corr(qty_values[qa], qty_values[qb])
                corr_matrix[(qa, qb)] = corr
                corr_matrix[(qb, qa)] = corr  # symmetric

        # Find strongly-correlated pairs → candidate groups
        strong_pairs: list[tuple[str, str, float]] = []
        for (qa, qb), corr in corr_matrix.items():
            if qa < qb and abs(corr) >= self.correlation_threshold:
                strong_pairs.append((qa, qb, corr))

        # Cluster pairs into groups (transitive closure over strongly-correlated pairs)
        groups: list[set[str]] = []
        for qa, qb, _ in strong_pairs:
            placed = False
            for grp in groups:
                if qa in grp or qb in grp:
                    grp.add(qa)
                    grp.add(qb)
                    placed = True
                    break
            if not placed:
                groups.append({qa, qb})

        # Merge overlapping groups
        merged = True
        while merged:
            merged = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if groups[i] & groups[j]:
                        groups[i] |= groups[j]
                        groups.pop(j)
                        merged = True
                        break
                if merged:
                    break

        # Build detections for each group (min size 2)
        detections: list[GroupedQuantityDetection] = []
        for grp in groups:
            if len(grp) < 2:
                continue
            grp_list = sorted(grp)

            # Mean absolute correlation for this group
            group_corrs: list[float] = []
            grp_sub_matrix: dict[tuple[str, str], float] = {}
            for qa in grp_list:
                for qb in grp_list:
                    if qa < qb and (qa, qb) in corr_matrix:
                        c = corr_matrix[(qa, qb)]
                        group_corrs.append(abs(c))
                        grp_sub_matrix[(qa, qb)] = c
            mean_abs_corr = sum(group_corrs) / len(group_corrs) if group_corrs else 0.0

            # Determine co-variation pattern
            pattern, pattern_conf = self._classify_pattern(
                grp_list, qty_values, corr_matrix,
            )

            # Across-observation consistency: does CV stay constant?
            across_consistency = self._compute_across_obs_consistency(
                grp_list, observations,
            )

            # Suggest metrics based on pattern
            suggested_metrics = self._suggest_metrics(pattern, grp_list)

            detections.append(GroupedQuantityDetection(
                group=grp_list,
                correlation_matrix=grp_sub_matrix,
                mean_abs_correlation=mean_abs_corr,
                co_variation_pattern=pattern,
                pattern_confidence=pattern_conf,
                across_observation_consistency=across_consistency,
                suggested_metrics=suggested_metrics,
                detection_source="correlation",
            ))

        # Sort by mean_abs_correlation (strongest first)
        detections.sort(key=lambda d: -d.mean_abs_correlation)
        return detections

    def _classify_pattern(
        self,
        grp: list[str],
        qty_values: dict[str, list[float]],
        corr_matrix: dict[tuple[str, str], float],
    ) -> tuple[str, float]:
        """Classify the co-variation pattern for a group of quantities."""
        # Check if any pair has near-1 correlation → linear
        linear_count = 0
        inverse_count = 0
        for qa in grp:
            for qb in grp:
                if qa < qb and (qa, qb) in corr_matrix:
                    c = corr_matrix[(qa, qb)]
                    if abs(c) > 0.9:
                        linear_count += 1
                    elif c < -0.7:
                        inverse_count += 1

        n_pairs = len(grp) * (len(grp) - 1) // 2
        if n_pairs == 0:
            return "unknown", 0.0

        linear_ratio = linear_count / n_pairs
        inverse_ratio = inverse_count / n_pairs

        if linear_ratio > 0.6:
            return "linear", linear_ratio
        elif inverse_ratio > 0.6:
            return "inverse", inverse_ratio
        elif linear_ratio + inverse_ratio > 0.5:
            return "mixed_linear", linear_ratio + inverse_ratio

        # Check for quadratic by looking at squared values
        for qa in grp:
            for qb in grp:
                if qa >= qb:
                    continue
                v1 = qty_values.get(qa, [])
                v2 = qty_values.get(qb, [])
                n = min(len(v1), len(v2))
                if n < 4:
                    continue
                v1_sq = [x * x for x in v1[:n]]
                sq_corr = _compute_pairwise_corr(v1_sq, v2[:n])
                if abs(sq_corr) > 0.8:
                    return "quadratic", abs(sq_corr)

        return "co_varying", max(linear_ratio, inverse_ratio)

    def _compute_across_obs_consistency(
        self, grp: list[str], observations: list[Observation],
    ) -> float:
        """How consistent is the relationship across different observations?"""
        if len(observations) < 2 or len(grp) < 2:
            return 1.0

        # For each observation, compute CV of pairwise product/ratio
        cv_list: list[float] = []
        for obs in observations:
            vals_a: list[float] = []
            vals_b: list[float] = []
            for ts in obs.timesteps:
                a = ts.get(grp[0])
                b = ts.get(grp[1]) if len(grp) > 1 else 1.0
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    vals_a.append(float(a))
                    vals_b.append(float(b))
            if len(vals_a) < 2:
                continue
            n = min(len(vals_a), len(vals_b))
            products = [vals_a[i] * vals_b[i] for i in range(n)]
            cv_list.append(_compute_cv(products))

        if not cv_list:
            return 0.0
        # Lower mean CV = more consistent
        mean_cv = sum(cv_list) / len(cv_list)
        return max(0.0, 1.0 - mean_cv)

    def _suggest_metrics(
        self, pattern: str, grp: list[str],
    ) -> list[str]:
        """Suggest metric types for a group based on co-variation pattern."""
        suggestions: list[str] = []

        # Product is the default for linear co-variation
        if pattern in ("linear", "mixed_linear"):
            suggestions.extend([GROUPED_METRIC_PRODUCT, GROUPED_METRIC_RATIO])
        if pattern == "inverse":
            suggestions.extend([GROUPED_METRIC_RATIO, GROUPED_METRIC_PRODUCT])
        if pattern == "quadratic":
            suggestions.extend([GROUPED_METRIC_SQUARED_DIFF, GROUPED_METRIC_SUM_SQUARES])

        # Always include squared_diff and sum_squares — the "metric" concept
        # (there's always a possible distance relation between grouped quantities)
        if GROUPED_METRIC_SQUARED_DIFF not in suggestions:
            suggestions.append(GROUPED_METRIC_SQUARED_DIFF)
        if GROUPED_METRIC_SUM_SQUARES not in suggestions:
            suggestions.append(GROUPED_METRIC_SUM_SQUARES)

        # For any pair that looks like a potential (t, x) — suggest weighted_diff
        grp_set = set(grp)
        if "t" in grp_set:
            spatials = grp_set & {"x", "y", "z"}
            if spatials:
                if GROUPED_METRIC_WEIGHTED_DIFF not in suggestions:
                    suggestions.insert(0, GROUPED_METRIC_WEIGHTED_DIFF)
                if GROUPED_METRIC_SQUARED_DIFF not in suggestions:
                    suggestions.append(GROUPED_METRIC_SQUARED_DIFF)

        return suggestions


# =============================================================================
# 9. GroupedMetricProposer — MLP that proposes metrics for grouped quantities
# =============================================================================

class GroupedMetricProposer(nn.Module):
    """MLP that proposes metric types for detected quantity groups.

    Input:  [group_encoding (NUM_GROUPED_QUANTITIES * 2) + correlation_features (6)]
    Hidden: 48 -> 32
    Output: [metric_logits (NUM_GROUPED_METRICS) + confidence + scale_factor]

    ~11K parameters.
    Trained ONLY on pre-1905 patterns (sound waves, fluids, orbits, oscillators).
    Era-gated: never shown Lorentz or spacetime data during training.
    """

    def __init__(self, *, hidden_dim: int = 48, dropout: float = 0.1) -> None:
        super().__init__()
        # Input: 2 quantity one-hots + 6 correlation features
        input_dim = NUM_GROUPED_QUANTITIES * 2 + 6
        output_dim = NUM_GROUPED_METRICS + 1 + 1  # metrics + confidence + scale_factor

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        return self.fc3(h)

    def propose(
        self,
        q1_onehot: torch.Tensor,
        q2_onehot: torch.Tensor,
        correlation_features: torch.Tensor,
        *,
        temperature: float = 0.15,
    ) -> list[MetricProposal]:
        """Propose metric types for a quantity pair.

        Args:
            q1_onehot: [batch, NUM_GROUPED_QUANTITIES] one-hot for first quantity
            q2_onehot: [batch, NUM_GROUPED_QUANTITIES] one-hot for second quantity
            correlation_features: [batch, 6] with [abs_corr, cv_product, cv_ratio,
                                  cv_diff, linear_score, consistency]
        """
        x = torch.cat([q1_onehot, q2_onehot, correlation_features], dim=-1)
        output = self.forward(x)

        metric_logits = output[:, :NUM_GROUPED_METRICS]
        confidence_logit = output[:, NUM_GROUPED_METRICS]
        scale_logit = output[:, NUM_GROUPED_METRICS + 1]

        metric_probs = F.softmax(metric_logits / max(temperature, 1e-8), dim=-1)
        confidence = torch.sigmoid(confidence_logit)
        scale_factor = F.softplus(scale_logit)  # positive scale

        batch_size = x.size(0)
        results: list[MetricProposal] = []
        for b in range(batch_size):
            _, top_idx = metric_probs[b].topk(min(3, NUM_GROUPED_METRICS))
            for rank, mi in enumerate(top_idx):
                metric_type = IDX_TO_GROUPED_METRIC[mi.item()]
                q1_idx = q1_onehot[b].argmax().item()
                q2_idx = q2_onehot[b].argmax().item()
                q1_name = _GROUPED_QTY_VOCAB[q1_idx] if q1_idx < len(_GROUPED_QTY_VOCAB) else "q1"
                q2_name = _GROUPED_QTY_VOCAB[q2_idx] if q2_idx < len(_GROUPED_QTY_VOCAB) else "q2"
                sc = scale_factor[b].item()

                # Build expression template
                if metric_type == GROUPED_METRIC_SQUARED_DIFF:
                    expr = f"{q1_name}^2 - {q2_name}^2"
                elif metric_type == GROUPED_METRIC_SUM_SQUARES:
                    expr = f"{q1_name}^2 + {q2_name}^2"
                elif metric_type == GROUPED_METRIC_PRODUCT:
                    expr = f"{q1_name} * {q2_name}"
                elif metric_type == GROUPED_METRIC_RATIO:
                    expr = f"{q1_name} / {q2_name}"
                elif metric_type == GROUPED_METRIC_WEIGHTED_DIFF:
                    expr = f"({sc:.3g}*{q1_name})^2 - {q2_name}^2"
                else:
                    expr = f"{q1_name} ~ {q2_name}"

                conf = confidence[b].item() * (1.0 - 0.15 * rank)
                results.append(MetricProposal(
                    metric_type=metric_type,
                    quantities=[q1_name, q2_name],
                    expression_template=expr,
                    scale_factor=sc if metric_type == GROUPED_METRIC_WEIGHTED_DIFF else None,
                    confidence=conf,
                    rationale=f"Top-{rank+1} metric for ({q1_name},{q2_name}) group: {metric_type} (conf={conf:.3f})",
                ))

        return results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 10. GroupedMetricSearch — search over metric candidates
# =============================================================================

class GroupedMetricSearch:
    """Search over metric candidates to discover invariants for grouped quantities.

    Given a detected group (e.g., t, x) and proposed metrics, systematically
    searches expression space to find constant combinations:

    For metric squared_diff on (t, x):
      Try: t² - x², (k*t)² - x² for various k
      Score constancy → pick best

    For weighted_diff on (t, x):
      Sweep scale factor k to find (k*t)² - x² with maximum constancy
      → if k ≈ c (speed of light), discover (ct)² - x²
    """

    def __init__(
        self,
        *,
        scale_sweep_range: tuple[float, float] = (0.5, 400.0),
        scale_sweep_steps: int = 100,
        discovery_threshold: float = 0.90,
    ) -> None:
        self.scale_sweep_range = scale_sweep_range
        self.scale_sweep_steps = scale_sweep_steps
        self.discovery_threshold = discovery_threshold
        self._evaluator = ExpressionEvaluator()

    def discover(
        self,
        detection: GroupedQuantityDetection,
        proposals: list[MetricProposal],
        observations: list[Observation],
        *,
        existing_quantities: dict[str, Dimension] | None = None,
    ) -> GroupedMetricDiscoveryResult:
        """Search for invariant under metric proposals for the detected group."""
        baseline_constancy = self._baseline_constancy(detection.group, observations)
        best_constancy = baseline_constancy
        best_invariant: str | None = None
        best_metric: str | None = None
        candidates: list[str] = []
        scores: list[float] = []
        tried = 0

        for proposal in proposals[:10]:  # limit proposals
            if proposal.metric_type == GROUPED_METRIC_WEIGHTED_DIFF:
                # Sweep scale factor
                for sc in self._scale_sweep():
                    expr = self._build_weighted_diff_expr(
                        proposal.quantities, sc,
                    )
                    const = self._evaluate_constancy(expr, observations)
                    candidates.append(expr)
                    scores.append(const)
                    tried += 1
                    if const > best_constancy:
                        best_constancy = const
                        best_invariant = expr
                        best_metric = proposal.metric_type
            else:
                expr = proposal.expression_template
                const = self._evaluate_constancy(expr, observations)
                candidates.append(expr)
                scores.append(const)
                tried += 1
                if const > best_constancy:
                    best_constancy = const
                    best_invariant = expr
                    best_metric = proposal.metric_type

        # Also try with existing quantities as modifiers for ALL pairs in group
        if existing_quantities:
            for i, qi in enumerate(detection.group):
                for qj in detection.group[i + 1:]:
                    for qname in existing_quantities:
                        if qname in detection.group:
                            continue
                        # Try (qname*qi)^2 ± qj^2 pattern
                        for sign, sign_name in [("-", "squared_diff"), ("+", "sum_squares")]:
                            expr = f"({qname}*{qi})^2 {sign} {qj}^2"
                            const = self._evaluate_constancy(expr, observations)
                            candidates.append(expr)
                            scores.append(const)
                            tried += 1
                            if const > best_constancy:
                                best_constancy = const
                                best_invariant = expr
                                best_metric = sign_name
                        # Also try reversed: (qname*qj)^2 ± qi^2
                        for sign, sign_name in [("-", "squared_diff"), ("+", "sum_squares")]:
                            expr = f"({qname}*{qj})^2 {sign} {qi}^2"
                            const = self._evaluate_constancy(expr, observations)
                            candidates.append(expr)
                            scores.append(const)
                            tried += 1
                            if const > best_constancy:
                                best_constancy = const
                                best_invariant = expr
                                best_metric = sign_name

        discovered = best_constancy >= self.discovery_threshold
        return GroupedMetricDiscoveryResult(
            discovered=discovered,
            detected_group=detection.group,
            metric_type=best_metric,
            best_invariant=best_invariant,
            best_constancy=best_constancy,
            baseline_constancy=baseline_constancy,
            proposals_tried=tried,
            search_candidates=candidates[:200],
            candidate_scores=scores[:200],
            metadata={
                "group": detection.group,
                "suggested_metrics": detection.suggested_metrics,
                "co_variation_pattern": detection.co_variation_pattern,
                "mean_abs_correlation": detection.mean_abs_correlation,
            },
        )

    def _scale_sweep(self) -> list[float]:
        """Logarithmic sweep of scale factors."""
        lo, hi = self.scale_sweep_range
        steps = self.scale_sweep_steps
        log_lo = math.log(lo)
        log_hi = math.log(hi)
        return [math.exp(log_lo + (log_hi - log_lo) * i / (steps - 1))
                for i in range(steps)]

    def _build_weighted_diff_expr(
        self, quantities: list[str], scale: float,
    ) -> str:
        q1, q2 = quantities[0], quantities[1] if len(quantities) > 1 else "q2"
        return f"({scale:.6g}*{q1})^2 - {q2}^2"

    def _evaluate_constancy(
        self, expr: str, observations: list[Observation],
    ) -> float:
        """Score constancy of expression across observations."""
        try:
            ast = self._evaluator.parse(expr)
        except Exception:
            return 0.0

        values: list[float] = []
        for obs in observations:
            for ts in obs.timesteps:
                context = {**obs.parameters, **ts}
                try:
                    val = evaluate_node(ast, context)
                    if isinstance(val, (int, float)) and not math.isnan(val):
                        values.append(float(val))
                except (EvalError, ZeroDivisionError, ValueError, OverflowError):
                    pass

        if len(values) < 3:
            return 0.0

        mean = sum(values) / len(values)
        if abs(mean) < 1e-12:
            return 0.0
        var = sum((v - mean) ** 2 for v in values) / len(values)
        cv = math.sqrt(max(var, 0.0)) / abs(mean)
        return max(0.0, min(1.0, 1.0 / (1.0 + cv)))

    def _baseline_constancy(
        self, group: list[str], observations: list[Observation],
    ) -> float:
        """Baseline constancy — how constant is the product of grouped quantities?"""
        if len(group) < 2:
            return 0.0
        expr = " * ".join(group)
        return self._evaluate_constancy(expr, observations)


# =============================================================================
# 11. Training Data — pre-1905 only, era-gated
# =============================================================================

@dataclass
class GroupedMetricTrainingExample:
    """Training example for the grouped metric proposer.

    ALL pre-1905 — no relativistic or quantum patterns.
    """
    q1: str
    q2: str
    abs_corr: float       # |pearson correlation| between the quantities
    cv_product: float     # CV of q1*q2
    cv_ratio: float       # CV of q1/q2
    cv_diff: float        # CV of |q1 - q2|
    linear_score: float   # how linear is relationship (0-1)
    consistency: float    # across-observation consistency
    expected_metric: str  # what metric type should be proposed
    description: str


def generate_grouped_metric_training_data(
    era_cutoff: int = 1905,
) -> list[GroupedMetricTrainingExample]:
    """Generate era-gated training data.

    Examples are tagged by the earliest era in which they appear.
    Only examples from eras <= cutoff are included:
      1905 (always): classical physics — sound, fluids, orbits, oscillators,
                     Galilean, Boyle, Hooke, momentum, pendulum, Newton
      1920: + special relativity metrics, early quantum (Bohr, Rydberg)
      1950: + QED symmetries, nuclear binding, Dirac spinor invariants
      1970: + Standard Model, QCD, electroweak unification
      Today: + Higgs mechanism, neutrino oscillations, dark matter candidates

    NOT taught at cutoff=1905: Lorentz transforms, spacetime metric,
    (c*t)² - x², c as limiting speed (these appear at 1920).
    """
    examples: list[GroupedMetricTrainingExample] = []

    # ═════════════════════════════════════════════════════════════════════
    # ALWAYS: Pre-1905 classical physics (131 examples)
    # ═════════════════════════════════════════════════════════════════════

    # ── Sound waves: f*λ = v_sound (constant product) ──
    for i in range(15):
        v_sound = 343.0 + random.uniform(-20, 20)
        n_pts = random.randint(5, 12)
        freqs = [random.uniform(100, 2000) for _ in range(n_pts)]
        lambdas = [v_sound / f for f in freqs]
        products = [freqs[j] * lambdas[j] for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "f", "lambda", freqs, lambdas, products,
            expected_metric=GROUPED_METRIC_PRODUCT,
            desc=f"Sound wave f*λ={v_sound:.0f} variant {i+1}",
        ))

    # ── Bernoulli: P + ½ρv² = const (sum-of-squares-like) ──
    for i in range(12):
        rho = random.uniform(1.0, 1.3)
        n_pts = random.randint(5, 10)
        P0 = random.uniform(100000, 101325)
        velocities = [random.uniform(1, 30) for _ in range(n_pts)]
        pressures = [P0 - 0.5 * rho * v ** 2 for v in velocities]
        sum_sqs = [pressures[j] + 0.5 * rho * velocities[j] ** 2 for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "P", "v", pressures, velocities, sum_sqs,
            expected_metric=GROUPED_METRIC_SUM_SQUARES,
            desc=f"Bernoulli P+½ρv²={P0:.0f} variant {i+1}",
        ))

    # ── Orbits: r² * dθ/dt ≈ constant (angular momentum) ──
    for i in range(12):
        L_const = random.uniform(0.5, 5.0)
        n_pts = random.randint(5, 10)
        radii = [random.uniform(1.0, 10.0) for _ in range(n_pts)]
        thetas = [L_const / (r ** 2) for r in radii]  # dθ/dt
        products_r_theta = [radii[j] * radii[j] * thetas[j] for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "r", "theta", radii, thetas, products_r_theta,
            expected_metric=GROUPED_METRIC_PRODUCT,
            desc=f"Orbital L=r²*θ̇={L_const:.2f} variant {i+1}",
        ))

    # ── Coupled oscillators: x₁² ± x₂² = const (normal modes) ──
    for i in range(10):
        n_pts = random.randint(5, 10)
        A = random.uniform(1.0, 3.0)
        t_vals = [random.uniform(0, 2 * math.pi) for _ in range(n_pts)]
        # In-phase: x1 = A*cos(t), x2 = A*cos(t) → x1 - x2 = 0
        x1_vals = [A * math.cos(t) for t in t_vals]
        x2_vals = [A * math.cos(t) for t in t_vals]
        sq_diffs = [x1_vals[j] ** 2 - x2_vals[j] ** 2 for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "x1", "x2", x1_vals, x2_vals, sq_diffs,
            expected_metric=GROUPED_METRIC_SQUARED_DIFF,
            desc=f"Coupled oscillator same-phase variant {i+1}",
        ))

    # ── Coupled oscillators: out-of-phase → x₁² + x₂² = const ──
    for i in range(8):
        n_pts = random.randint(5, 10)
        A = random.uniform(1.0, 3.0)
        t_vals = [random.uniform(0, 2 * math.pi) for _ in range(n_pts)]
        x1_vals = [A * math.cos(t) for t in t_vals]
        x2_vals = [A * math.sin(t) for t in t_vals]
        sum_sqs = [x1_vals[j] ** 2 + x2_vals[j] ** 2 for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "x1", "x2", x1_vals, x2_vals, sum_sqs,
            expected_metric=GROUPED_METRIC_SUM_SQUARES,
            desc=f"Coupled oscillator quadrature variant {i+1}",
        ))

    # ── Galilean: x/t = v (ratio is constant) ──
    for i in range(12):
        v_const = random.uniform(5.0, 50.0)
        n_pts = random.randint(5, 12)
        times = [random.uniform(1.0, 10.0) for _ in range(n_pts)]
        positions = [v_const * t for t in times]
        ratios = [positions[j] / times[j] for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "t", "x", times, positions, ratios,
            expected_metric=GROUPED_METRIC_RATIO,
            desc=f"Galilean x/t=v={v_const:.1f} variant {i+1}",
        ))

    # ── Galilean: t and x product (trivial — no invariant) ──
    for i in range(8):
        n_pts = random.randint(5, 10)
        v = random.uniform(5, 30)
        times = [random.uniform(1, 10) for _ in range(n_pts)]
        positions = [v * t + random.uniform(-2, 2) for t in times]
        products = [times[j] * positions[j] for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "t", "x", times, positions, products,
            expected_metric=GROUPED_METRIC_PRODUCT,
            desc=f"Galilean t*x variant {i+1}",
        ))

    # ── Boyle's Law: P*V = const ──
    for i in range(10):
        n_pts = random.randint(5, 10)
        k_PV = random.uniform(1.0, 10.0)
        volumes = [random.uniform(0.5, 5.0) for _ in range(n_pts)]
        pressures = [k_PV / v for v in volumes]
        products = [pressures[j] * volumes[j] for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "P", "V", pressures, volumes, products,
            expected_metric=GROUPED_METRIC_PRODUCT,
            desc=f"Boyle PV={k_PV:.2f} variant {i+1}",
        ))

    # ── Hooke's Law: F/x = k (ratio constant) ──
    for i in range(10):
        k = random.uniform(10.0, 500.0)
        n_pts = random.randint(5, 10)
        displacements = [random.uniform(0.01, 0.5) for _ in range(n_pts)]
        forces = [k * d for d in displacements]
        ratios = [forces[j] / displacements[j] for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "F", "x", forces, displacements, ratios,
            expected_metric=GROUPED_METRIC_RATIO,
            desc=f"Hooke F/x=k={k:.0f} variant {i+1}",
        ))

    # ── Momentum: m*v = p (product constant for each object) ──
    for i in range(8):
        m = random.uniform(0.5, 10.0)
        n_pts = random.randint(5, 10)
        velocities = [random.uniform(1, 20) for _ in range(n_pts)]
        momenta = [m * v for v in velocities]
        products = [m * velocities[j] for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "m", "v", [m] * n_pts, velocities, products,
            expected_metric=GROUPED_METRIC_PRODUCT,
            desc=f"Momentum m*v=p variant {i+1}",
        ))

    # ── Pendulum: T²/L = 4π²/g (ratio constant) ──
    for i in range(8):
        g = 9.81
        n_pts = random.randint(5, 10)
        lengths = [random.uniform(0.5, 3.0) for _ in range(n_pts)]
        periods = [2 * math.pi * math.sqrt(L / g) for L in lengths]
        ratios = [periods[j] ** 2 / lengths[j] for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "T_period", "L", periods, lengths, ratios,
            expected_metric=GROUPED_METRIC_RATIO,
            desc=f"Pendulum T²/L=4π²/g variant {i+1}",
        ))

    # ── Newton's Second Law: F/a = m (ratio) ──
    for i in range(8):
        m = random.uniform(1.0, 20.0)
        n_pts = random.randint(5, 10)
        accelerations = [random.uniform(0.5, 15.0) for _ in range(n_pts)]
        forces = [m * a for a in accelerations]
        ratios = [forces[j] / accelerations[j] for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "F", "a", forces, accelerations, ratios,
            expected_metric=GROUPED_METRIC_RATIO,
            desc=f"Newton F/a=m={m:.1f} variant {i+1}",
        ))

    # ═════════════════════════════════════════════════════════════════════
    # ERA >= 1920: Special relativity + Early quantum (~40 examples)
    # ═════════════════════════════════════════════════════════════════════
    if era_cutoff < 1920:
        return examples

    # ── SR: Lorentz invariant (c*t)² - x² = const ──
    for i in range(10):
        c = 3e8
        s2_const = random.uniform(1e-12, 100e-12)
        n_pts = random.randint(5, 10)
        offsets = [random.uniform(-0.5, 0.5) for _ in range(n_pts)]
        x_vals = [random.uniform(10, 500) for _ in range(n_pts)]
        t_vals = [math.sqrt((x ** 2 + s2_const) / c ** 2) + off
                  for x, off in zip(x_vals, offsets)]
        sq_diffs = [(c * t_vals[j]) ** 2 - x_vals[j] ** 2 for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "t", "x", t_vals, x_vals, sq_diffs,
            expected_metric=GROUPED_METRIC_SQUARED_DIFF,
            desc=f"SR: (c*t)²-x²={s2_const:.2e} variant {i+1}",
        ))

    # ── SR: Energy-momentum invariant E² - (p*c)² = (m*c²)² ──
    for i in range(8):
        c = 3e8
        m = random.uniform(0.5, 10.0)
        E0 = m * c ** 2
        n_pts = random.randint(5, 10)
        E_vals: list[float] = []
        p_vals: list[float] = []
        for _ in range(n_pts):
            v = random.uniform(0.1, 0.99) * c
            gamma = 1.0 / math.sqrt(1.0 - v ** 2 / c ** 2)
            E_vals.append(gamma * E0)
            p_vals.append(gamma * m * v)
        sq_diffs = [E_vals[j] ** 2 - (p_vals[j] * c) ** 2 for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "E", "p", E_vals, p_vals, sq_diffs,
            expected_metric=GROUPED_METRIC_SQUARED_DIFF,
            desc=f"SR: E²-(p*c)²=(mc²)²={E0:.1e} variant {i+1}",
        ))

    # ── Bohr model: E_n ∝ 1/n² (ratio constant) ──
    for i in range(8):
        R = 13.6  # Rydberg energy
        n_pts = random.randint(5, 10)
        n_vals = [random.uniform(1, 6) for _ in range(n_pts)]
        E_vals = [-R / n ** 2 for n in n_vals]
        ratios = [E_vals[j] * n_vals[j] ** 2 for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "E", "n", E_vals, n_vals, ratios,
            expected_metric=GROUPED_METRIC_PRODUCT,
            desc=f"Bohr E*n²={-R} variant {i+1}",
        ))

    # ── Rydberg: 1/λ = R*(1/n1² - 1/n2²) → combinational invariance ──
    for i in range(8):
        Ryd = 1.097e7
        n_pts = random.randint(5, 10)
        n1_vals = [float(random.randint(1, 4)) for _ in range(n_pts)]
        n2_vals = [float(random.randint(5, 10)) for _ in range(n_pts)]
        inv_lambdas = [Ryd * (1 / n1 ** 2 - 1 / n2 ** 2)
                       for n1, n2 in zip(n1_vals, n2_vals)]
        ratios = [inv_lambdas[j] * n1_vals[j] * n2_vals[j]
                  for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "lambda", "n1", inv_lambdas, n1_vals, ratios,
            expected_metric=GROUPED_METRIC_RATIO,
            desc=f"Rydberg 1/λ variant {i+1}",
        ))

    # ── Time dilation: gamma = 1/sqrt(1-β²) → squared-diff pattern ──
    for i in range(6):
        c = 3e8
        n_pts = random.randint(5, 10)
        tau_vals: list[float] = []
        t_vals: list[float] = []
        for _ in range(n_pts):
            v = random.uniform(0.1, 0.99) * c
            gamma = 1.0 / math.sqrt(1.0 - v ** 2 / c ** 2)
            tau = 1.0
            t_vals.append(gamma * tau)
            tau_vals.append(tau)
        # (c*τ)² - (c*t)² + x² → not quite. Use ratio for t/tau = gamma
        ratios = [t_vals[j] / tau_vals[j] for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "t", "tau", t_vals, tau_vals, ratios,
            expected_metric=GROUPED_METRIC_RATIO,
            desc=f"SR time dilation t/τ=γ variant {i+1}",
        ))

    # ═════════════════════════════════════════════════════════════════════
    # ERA >= 1950: QED + Nuclear + Dirac (~30 examples)
    # ═════════════════════════════════════════════════════════════════════
    if era_cutoff < 1950:
        return examples

    # ── QED: Fine structure α = e²/(4πε₀ℏc) — ratio invariant ──
    for i in range(8):
        alpha = 1.0 / 137.036
        n_pts = random.randint(5, 10)
        e_vals = [random.uniform(1e-19, 5e-19) for _ in range(n_pts)]
        hc_vals = [e ** 2 / alpha for e in e_vals]
        ratios = [e_vals[j] / hc_vals[j] for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "e", "hc", e_vals, hc_vals, ratios,
            expected_metric=GROUPED_METRIC_RATIO,
            desc=f"QED: e²/hc={alpha:.4f} variant {i+1}",
        ))

    # ── Nuclear binding: B/A ≈ const (ratio for mid-mass nuclei) ──
    for i in range(6):
        B_A = random.uniform(7.5, 8.5)  # MeV per nucleon
        n_pts = random.randint(5, 10)
        A_vals = [random.uniform(20, 120) for _ in range(n_pts)]
        B_vals = [B_A * A for A in A_vals]
        ratios = [B_vals[j] / A_vals[j] for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "B", "A", B_vals, A_vals, ratios,
            expected_metric=GROUPED_METRIC_RATIO,
            desc=f"Nuclear B/A={B_A:.1f} MeV variant {i+1}",
        ))

    # ── Dirac: spinor norm ψ†ψ = ρ (density) — quadratic invariant ──
    for i in range(8):
        n_pts = random.randint(5, 10)
        rho_const = random.uniform(0.5, 5.0)
        psi_re = [random.uniform(-1, 1) for _ in range(n_pts)]
        psi_im = [math.sqrt(max(0, rho_const - pr ** 2))
                  for pr in psi_re]
        sq_sums = [psi_re[j] ** 2 + psi_im[j] ** 2 for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "psi_re", "psi_im", psi_re, psi_im, sq_sums,
            expected_metric=GROUPED_METRIC_SUM_SQUARES,
            desc=f"Dirac spinor norm ψ†ψ={rho_const:.2f} variant {i+1}",
        ))

    # ── Compton scattering: Δλ = h/(m*c)*(1-cosθ) — ratio in λ,θ space ──
    for i in range(8):
        lambda_c = 2.426e-12
        n_pts = random.randint(5, 10)
        theta_vals = [random.uniform(0.1, math.pi) for _ in range(n_pts)]
        dlambda_vals = [lambda_c * (1 - math.cos(t)) for t in theta_vals]
        ratios = [dlambda_vals[j] / (1 - math.cos(theta_vals[j]))
                  for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "dlambda", "theta", dlambda_vals, theta_vals, ratios,
            expected_metric=GROUPED_METRIC_RATIO,
            desc=f"Compton dλ/(1-cosθ)=λc variant {i+1}",
        ))

    # ═════════════════════════════════════════════════════════════════════
    # ERA >= 1970: Standard Model + QCD + Electroweak (~30 examples)
    # ═════════════════════════════════════════════════════════════════════
    if era_cutoff < 1970:
        return examples

    # ── SM: Gauge coupling unification at high energy — product invariant ──
    for i in range(8):
        n_pts = random.randint(5, 10)
        coupling = random.uniform(0.01, 0.1)
        E_vals = [random.uniform(1e9, 1e16) for _ in range(n_pts)]
        g_vals = [coupling * math.sqrt(math.log(E / 1e9) + 1)
                  for E in E_vals]
        products = [g_vals[j] * E_vals[j] for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "g", "E", g_vals, E_vals, products,
            expected_metric=GROUPED_METRIC_PRODUCT,
            desc=f"SM running coupling g*E variant {i+1}",
        ))

    # ── QCD: Color singlet triality — sum-of-squares pattern ──
    for i in range(8):
        n_pts = random.randint(5, 10)
        C_const = random.uniform(0.5, 3.0)
        c1_vals = [random.uniform(-1, 1) for _ in range(n_pts)]
        c2_vals = [random.uniform(-1, 1) for _ in range(n_pts)]
        c3_vals = [math.sqrt(max(0, C_const - c1 ** 2 - c2 ** 2))
                   for c1, c2 in zip(c1_vals, c2_vals)]
        sum_sqs = [c1_vals[j] ** 2 + c2_vals[j] ** 2 + c3_vals[j] ** 2
                   for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "c1", "c2", c1_vals, c2_vals, sum_sqs,
            expected_metric=GROUPED_METRIC_SUM_SQUARES,
            desc=f"QCD color singlet variant {i+1}",
        ))

    # ── Electroweak: θ_W mixing angle → ratio of couplings ──
    for i in range(8):
        theta_w = random.uniform(0.4, 0.5)
        n_pts = random.randint(5, 10)
        g_vals = [random.uniform(0.3, 0.7) for _ in range(n_pts)]
        gp_vals = [g * math.tan(theta_w) for g in g_vals]
        ratios = [gp_vals[j] / g_vals[j] for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "gp", "g", gp_vals, g_vals, ratios,
            expected_metric=GROUPED_METRIC_RATIO,
            desc=f"EW mixing tanθW={math.tan(theta_w):.3f} variant {i+1}",
        ))

    # ── Asymptotic freedom: α_s(E) decreases with E — ratio pattern ──
    for i in range(6):
        Lambda = 0.217  # GeV
        n_pts = random.randint(5, 10)
        E_vals = [random.uniform(2, 100) for _ in range(n_pts)]
        alpha_s_vals = [1.0 / math.log(E / Lambda) for E in E_vals]
        ratios = [alpha_s_vals[j] * math.log(E_vals[j] / Lambda)
                  for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "alpha_s", "E", alpha_s_vals, E_vals, ratios,
            expected_metric=GROUPED_METRIC_PRODUCT,
            desc=f"QCD α_s*log(E/Λ)≈1 variant {i+1}",
        ))

    # ═════════════════════════════════════════════════════════════════════
    # ERA = Today: Higgs + Neutrino + Dark Matter (~25 examples)
    # ═════════════════════════════════════════════════════════════════════
    if era_cutoff < 9999:  # "Today" gate
        return examples

    # Threshold: today cutoff means year >= 2024
    if era_cutoff < 2024:
        return examples

    # ── Higgs: V(φ) = -μ²|φ|² + λ|φ|⁴ → squared-diff minimum ──
    for i in range(6):
        v = 246.0  # GeV
        n_pts = random.randint(5, 10)
        phi_vals = [random.uniform(0, 2 * v) for _ in range(n_pts)]
        V_vals = [0.25 * (p ** 2 - v ** 2) ** 2 for p in phi_vals]
        # V is proportional to (φ² - v²)² — squared-diff in φ², v² space
        sq_diffs = [phi_vals[j] ** 2 - v ** 2 for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "phi", "v", phi_vals, [v] * n_pts, sq_diffs,
            expected_metric=GROUPED_METRIC_SQUARED_DIFF,
            desc=f"Higgs V∝(φ²-v²)² variant {i+1}",
        ))

    # ── Neutrino oscillations: P(ν_e→ν_μ) = sin²(2θ)*sin²(Δm²L/4E) ──
    for i in range(8):
        n_pts = random.randint(5, 10)
        theta13 = random.uniform(0.1, 0.2)
        E_vals = [random.uniform(1, 10) for _ in range(n_pts)]  # GeV
        L_vals = [random.uniform(10, 1000) for _ in range(n_pts)]  # km
        # Oscillation phase φ = Δm²*L / (4*E)
        dm2 = 2.5e-3  # eV²
        ratios = [(dm2 * L_vals[j]) / (4 * E_vals[j]) for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "L", "E", L_vals, E_vals, ratios,
            expected_metric=GROUPED_METRIC_RATIO,
            desc=f"Neutrino L/E phase variant {i+1}",
        ))

    # ── Dark matter: v²(r) ∝ M(r)/r — ratio in velocity, radius ──
    for i in range(8):
        n_pts = random.randint(5, 10)
        M_const = random.uniform(1e10, 1e12)
        G = 6.67e-11
        r_vals = [random.uniform(1, 100) for _ in range(n_pts)]  # kpc
        v2_vals = [G * M_const / r for r in r_vals]
        ratios = [v2_vals[j] * r_vals[j] for j in range(n_pts)]
        examples.append(_build_grouped_example(
            "v2", "r", v2_vals, r_vals, ratios,
            expected_metric=GROUPED_METRIC_PRODUCT,
            desc=f"DM v²*r=GM variant {i+1}",
        ))

    return examples


def _build_grouped_example(
    q1: str, q2: str,
    vals1: list[float], vals2: list[float],
    combined_vals: list[float],
    expected_metric: str,
    desc: str,
) -> GroupedMetricTrainingExample:
    """Build a training example from synthetic value sequences."""
    n = min(len(vals1), len(vals2), len(combined_vals))
    v1 = vals1[:n]
    v2 = vals2[:n]
    cv = combined_vals[:n]

    abs_corr = abs(_compute_pairwise_corr(v1, v2))
    cv_product = _compute_cv([a * b for a, b in zip(v1, v2)])
    cv_ratio = _compute_cv([a / max(b, 1e-10) for b, a in zip(v1, v2)] if all(abs(b) > 1e-10 for b in v2) else [1.0])
    cv_diff = _compute_cv([abs(a - b) for a, b in zip(v1, v2)])
    linear_score = abs_corr if abs_corr > 0.7 else 0.5
    consistency = max(0.0, 1.0 - _compute_cv(cv))

    return GroupedMetricTrainingExample(
        q1=q1, q2=q2,
        abs_corr=abs_corr,
        cv_product=cv_product,
        cv_ratio=cv_ratio,
        cv_diff=cv_diff,
        linear_score=linear_score,
        consistency=consistency,
        expected_metric=expected_metric,
        description=desc,
    )


def build_grouped_metric_batch(
    examples: list[GroupedMetricTrainingExample],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build training batch for GroupedMetricProposer."""
    batch_size = len(examples)
    input_dim = NUM_GROUPED_QUANTITIES * 2 + 6
    output_dim = NUM_GROUPED_METRICS + 1 + 1

    inputs = torch.zeros(batch_size, input_dim)
    targets = torch.zeros(batch_size, output_dim)

    for i, ex in enumerate(examples):
        # One-hot for q1
        if ex.q1 in GROUPED_QTY_TO_IDX:
            inputs[i, GROUPED_QTY_TO_IDX[ex.q1]] = 1.0
        # One-hot for q2
        if ex.q2 in GROUPED_QTY_TO_IDX:
            inputs[i, NUM_GROUPED_QUANTITIES + GROUPED_QTY_TO_IDX[ex.q2]] = 1.0

        # Correlation features
        offset = NUM_GROUPED_QUANTITIES * 2
        inputs[i, offset] = ex.abs_corr
        inputs[i, offset + 1] = ex.cv_product
        inputs[i, offset + 2] = ex.cv_ratio
        inputs[i, offset + 3] = ex.cv_diff
        inputs[i, offset + 4] = ex.linear_score
        inputs[i, offset + 5] = ex.consistency

        # Target: metric type
        if ex.expected_metric in GROUPED_METRIC_TO_IDX:
            targets[i, GROUPED_METRIC_TO_IDX[ex.expected_metric]] = 1.0
        # Confidence target = 1.0
        targets[i, NUM_GROUPED_METRICS] = 1.0
        # Scale factor target = 1.0 (default, learned during training)
        targets[i, NUM_GROUPED_METRICS + 1] = 1.0

    return inputs, targets


# =============================================================================
# 12. Training Loop for GroupedMetricProposer
# =============================================================================

def train_grouped_metric_proposer(
    *,
    proposer: GroupedMetricProposer | None = None,
    epochs: int = 200,
    lr: float = 0.003,
    device: str = "cpu",
    checkpoint_path: str | None = None,
    era_cutoff: int = 1905,
) -> GroupedMetricProposer:
    """Train the GroupedMetricProposer on era-gated data.

    The model learns which metric types are appropriate for co-varying
    quantity groups. Training data is filtered by era_cutoff:
    only pre-cutoff physics is included.

    At cutoff=1905: classical physics only.
    What is NOT shown at 1905: Lorentz transforms, spacetime metrics,
    (c*t)² - x², c as limiting speed.
    """
    if proposer is None:
        proposer = GroupedMetricProposer()
    proposer.to(device)
    proposer.train()

    examples = generate_grouped_metric_training_data(era_cutoff=era_cutoff)
    pre1905_count = sum(1 for e in examples if "SR:" not in e.description
                        and "Bohr" not in e.description
                        and "QED" not in e.description
                        and "Nuclear" not in e.description
                        and "Dirac" not in e.description
                        and "Compton" not in e.description
                        and "SM " not in e.description
                        and "QCD" not in e.description
                        and "EW " not in e.description
                        and "Higgs" not in e.description
                        and "Neutrino" not in e.description
                        and "DM " not in e.description
                        and "Rydberg" not in e.description)
    post1905_count = len(examples) - pre1905_count
    print(f"  Generated {len(examples)} era-gated training examples "
          f"(cutoff={era_cutoff}, pre-1905: {pre1905_count}, "
          f"post-1905: {post1905_count})")
    inputs, targets = build_grouped_metric_batch(examples)
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer = torch.optim.Adam(proposer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    metric_loss_fn = nn.CrossEntropyLoss()
    conf_loss_fn = nn.BCEWithLogitsLoss()
    scale_loss_fn = nn.MSELoss()

    best_loss = float("inf")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = proposer(inputs)

        metric_logits = output[:, :NUM_GROUPED_METRICS]
        conf_logits = output[:, NUM_GROUPED_METRICS]
        scale_preds = output[:, NUM_GROUPED_METRICS + 1]

        metric_targets = targets[:, :NUM_GROUPED_METRICS].argmax(dim=-1)
        conf_targets = targets[:, NUM_GROUPED_METRICS]
        scale_targets = targets[:, NUM_GROUPED_METRICS + 1]

        loss = (metric_loss_fn(metric_logits, metric_targets)
                + 0.1 * conf_loss_fn(conf_logits, conf_targets)
                + 0.05 * scale_loss_fn(scale_preds, scale_targets))
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (epoch + 1) % 25 == 0:
            with torch.no_grad():
                metric_acc = (metric_logits.argmax(-1) == metric_targets).float().mean()
            print(f"  epoch {epoch+1}/{epochs}  loss={loss.item():.4f}  "
                  f"metric_acc={metric_acc.item():.3f}")

    proposer.eval()
    print(f"  Training complete. Best loss={best_loss:.4f}")

    if checkpoint_path:
        save_path = Path(checkpoint_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": proposer.state_dict(),
            "num_grouped_metrics": NUM_GROUPED_METRICS,
            "num_grouped_quantities": NUM_GROUPED_QUANTITIES,
            "grouped_qty_vocab": _GROUPED_QTY_VOCAB,
            "grouped_metric_to_idx": GROUPED_METRIC_TO_IDX,
            "version": "v1",
        }, save_path)
        print(f"  Saved checkpoint to {save_path}")

    return proposer


def load_grouped_metric_proposer(
    checkpoint_path: str, device: str = "cpu",
) -> GroupedMetricProposer:
    """Load a trained GroupedMetricProposer from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    proposer = GroupedMetricProposer()
    proposer.load_state_dict(checkpoint["model_state_dict"])
    proposer.to(device)
    proposer.eval()
    return proposer


# =============================================================================
# 13. Full Grouped Metric Discovery Pipeline
# =============================================================================

def run_grouped_metric_discovery(
    quantity_dict: dict[str, Dimension],
    observations: list[Observation],
    *,
    domain: str = "unknown",
    proposer: GroupedMetricProposer | None = None,
    discovery_threshold: float = 0.90,
    use_mlp: bool = True,
) -> list[GroupedMetricDiscoveryResult]:
    """Run the full grouped metric discovery pipeline.

    Pipeline:
    1. Detect co-varying quantity groups from observations
    2. For each group, propose metric candidates
    3. Search metric space for invariant combinations
    4. Return discoveries

    Era-gated: detection and proposal trained on pre-1905 data only.
    When shown post-1905 data (muon time dilation), should detect (t,x)
    as a group and discover (c*t)² - x² as invariant.
    """
    detector = GroupedQuantityDetector(correlation_threshold=0.5)
    searcher = GroupedMetricSearch(discovery_threshold=discovery_threshold)

    qty_names = list(quantity_dict.keys())
    detections = detector.detect_groups(observations, qty_names)

    results: list[GroupedMetricDiscoveryResult] = []

    for detection in detections:
        # Generate proposals for ALL pairs within the group
        if use_mlp and proposer is not None and len(detection.group) >= 2:
            mlp_proposals = _mlp_propose_grouped(
                proposer, detection,
            )
        else:
            mlp_proposals = []

        # Also include rule-based proposals from detection
        rule_proposals: list[MetricProposal] = []
        for mt in detection.suggested_metrics:
            if len(detection.group) >= 2:
                # Generate proposals for ALL pairs in the group
                for i, qa in enumerate(detection.group):
                    for qb in detection.group[i + 1:]:
                        if mt == GROUPED_METRIC_SQUARED_DIFF:
                            expr = f"{qa}^2 - {qb}^2"
                        elif mt == GROUPED_METRIC_SUM_SQUARES:
                            expr = f"{qa}^2 + {qb}^2"
                        elif mt == GROUPED_METRIC_PRODUCT:
                            expr = f"{qa} * {qb}"
                        elif mt == GROUPED_METRIC_RATIO:
                            expr = f"{qa} / {qb}"
                        elif mt == GROUPED_METRIC_WEIGHTED_DIFF:
                            expr = f"(c*{qa})^2 - {qb}^2"
                        else:
                            expr = f"{qa} ~ {qb}"
                        rule_proposals.append(MetricProposal(
                            metric_type=mt, quantities=[qa, qb],
                            expression_template=expr, confidence=0.7,
                            rationale=f"Rule-based: {detection.co_variation_pattern} → {mt} for ({qa},{qb})",
                        ))

        all_proposals = mlp_proposals + rule_proposals
        # Deduplicate
        seen: set[str] = set()
        unique_proposals: list[MetricProposal] = []
        for p in all_proposals:
            key = f"{p.metric_type}:{p.expression_template}"
            if key not in seen:
                unique_proposals.append(p)
                seen.add(key)

        result = searcher.discover(
            detection=detection,
            proposals=unique_proposals,
            observations=observations,
            existing_quantities=quantity_dict,
        )
        results.append(result)

    # Sort by best_constancy
    results.sort(key=lambda r: -r.best_constancy)
    return results


def _mlp_propose_grouped(
    proposer: GroupedMetricProposer,
    detection: GroupedQuantityDetection,
) -> list[MetricProposal]:
    """Generate MLP-based metric proposals for a detected group."""
    if len(detection.group) < 2:
        return []

    grp = detection.group
    proposals: list[MetricProposal] = []
    for i, qa in enumerate(grp):
        for qb in grp[i + 1:]:
            q1_oh = torch.zeros(NUM_GROUPED_QUANTITIES)
            q2_oh = torch.zeros(NUM_GROUPED_QUANTITIES)
            if qa in GROUPED_QTY_TO_IDX:
                q1_oh[GROUPED_QTY_TO_IDX[qa]] = 1.0
            if qb in GROUPED_QTY_TO_IDX:
                q2_oh[GROUPED_QTY_TO_IDX[qb]] = 1.0

            # Build correlation features from detection
            corr = abs(detection.correlation_matrix.get((qa, qb), 0.0))
            feat = torch.tensor([
                corr,
                max(0.0, 1.0 - detection.across_observation_consistency),  # cv_product proxy
                0.5,  # cv_ratio proxy
                0.5,  # cv_diff proxy
                corr,  # linear_score
                detection.across_observation_consistency,
            ])

            try:
                with torch.no_grad():
                    props = proposer.propose(
                        q1_oh.unsqueeze(0), q2_oh.unsqueeze(0),
                        feat.unsqueeze(0), temperature=0.3,
                    )
                proposals.extend(props)
            except Exception:
                pass

    # Sort by confidence
    proposals.sort(key=lambda p: -p.confidence)
    return proposals[:5]


def _feature_vector_for_pair(
    q1: str, q2: str,
    detection: GroupedQuantityDetection,
) -> torch.Tensor:
    """Build 6-d feature vector for a quantity pair."""
    corr = abs(detection.correlation_matrix.get((q1, q2), 0.0))
    return torch.tensor([
        corr,                                    # abs_corr
        max(0.0, 1.0 - detection.across_observation_consistency),  # cv_product proxy
        0.5,                                     # cv_ratio (unknown w/o raw data)
        0.5,                                     # cv_diff
        corr,                                    # linear_score
        detection.across_observation_consistency,  # consistency
    ])
