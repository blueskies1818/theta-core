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

ALL_SHAPES = [
    SHAPE_LINEAR, SHAPE_QUADRATIC, SHAPE_INVERSE_SQUARE,
    SHAPE_EXPONENTIAL, SHAPE_PERIODIC, SHAPE_RANDOM, SHAPE_CONSTANT,
    SHAPE_LINEAR_RATIO, SHAPE_POWER_LAW, SHAPE_MULTI_VAR,
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

VAR_TYPES = [
    VAR_INTEGER, VAR_HALF_INTEGER, VAR_ANGULAR_M, VAR_SPIN, VAR_CONTINUOUS,
    VAR_CONTINUOUS_RATIO, VAR_CONTINUOUS_ADDITIVE,
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

TRANSFORMS = [
    TRANSFORM_IDENTITY, TRANSFORM_SQUARED, TRANSFORM_INV_SQUARED,
    TRANSFORM_SQRT, TRANSFORM_RATIO, TRANSFORM_OFFSET,
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
        }
        patch_map = {
            TRANSFORM_IDENTITY: "*{name}", TRANSFORM_SQUARED: "*{name}^2",
            TRANSFORM_INV_SQUARED: "/{name}^2", TRANSFORM_SQRT: "*sqrt({name})",
            TRANSFORM_RATIO: "*{name}", TRANSFORM_OFFSET: "+{name}",
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
