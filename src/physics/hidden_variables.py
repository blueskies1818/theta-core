"""Hidden variable discovery — learn to propose new ingredients from failure patterns.

When beam search returns empty (no constant expression found), this module
analyzes the RESIDUALS of the best-scoring failed expressions to detect
hidden structure (e.g., hydrogen spectrum residuals follow 1/n^2).

Architecture:
  1. ErrorShapeDetector: classify how best-scoring failed expressions vary
  2. HiddenVariableProposer: small MLP (~10K params) that maps error shape
     + quantity names + domain -> proposed hidden variable
  3. HiddenVariableDiscovery: verification loop — propose -> test -> validate

Training data: ~60 synthetic scenarios where hidden variables exist
  - Hydrogen spectrum: residuals follow 1/n^2 -> propose integer n, use n^2
  - Particle in box: residuals follow n^2 -> propose integer n, use n^2
  - Harmonic oscillator: residuals evenly spaced -> propose half-integer j
  - Zeeman splitting: residuals symmetric -> propose angular m
"""

from __future__ import annotations

import json
import math
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
# Constants
# =============================================================================

SHAPE_LINEAR = "linear"
SHAPE_QUADRATIC = "quadratic"
SHAPE_INVERSE_SQUARE = "inverse_square"
SHAPE_EXPONENTIAL = "exponential"
SHAPE_PERIODIC = "periodic"
SHAPE_RANDOM = "random"
SHAPE_CONSTANT = "constant"

ALL_SHAPES = [
    SHAPE_LINEAR, SHAPE_QUADRATIC, SHAPE_INVERSE_SQUARE,
    SHAPE_EXPONENTIAL, SHAPE_PERIODIC, SHAPE_RANDOM, SHAPE_CONSTANT,
]
NUM_SHAPES = len(ALL_SHAPES)
SHAPE_TO_IDX = {s: i for i, s in enumerate(ALL_SHAPES)}
IDX_TO_SHAPE = {i: s for s, i in SHAPE_TO_IDX.items()}

VAR_INTEGER = "integer_n"
VAR_HALF_INTEGER = "half_integer"
VAR_ANGULAR_M = "angular_m"
VAR_SPIN = "spin_s"
VAR_CONTINUOUS = "continuous"

VAR_TYPES = [VAR_INTEGER, VAR_HALF_INTEGER, VAR_ANGULAR_M, VAR_SPIN, VAR_CONTINUOUS]
NUM_VAR_TYPES = len(VAR_TYPES)
VAR_TYPE_TO_IDX = {t: i for i, t in enumerate(VAR_TYPES)}
IDX_TO_VAR_TYPE = {i: t for t, i in VAR_TYPE_TO_IDX.items()}

TRANSFORM_IDENTITY = "identity"
TRANSFORM_SQUARED = "squared"
TRANSFORM_INV_SQUARED = "inv_squared"
TRANSFORM_SQRT = "sqrt"

TRANSFORMS = [TRANSFORM_IDENTITY, TRANSFORM_SQUARED, TRANSFORM_INV_SQUARED, TRANSFORM_SQRT]
NUM_TRANSFORMS = len(TRANSFORMS)
TRANSFORM_TO_IDX = {t: i for i, t in enumerate(TRANSFORMS)}
IDX_TO_TRANSFORM = {i: t for t, i in TRANSFORM_TO_IDX.items()}

HIDDEN_VAR_DOMAINS = ["gravity", "spring", "em", "thermal", "quantum", "relativistic", "unknown"]
NUM_HV_DOMAINS = len(HIDDEN_VAR_DOMAINS)
HV_DOMAIN_TO_IDX = {d: i for i, d in enumerate(HIDDEN_VAR_DOMAINS)}

_HV_QUANTITY_VOCAB = [
    "m", "g", "h", "v", "t", "k", "L", "q", "E", "x", "y", "r",
    "P", "V", "T", "S", "n", "R", "B", "W", "Q", "c", "p",
    "m1", "v1", "m2", "v2", "x1", "x2", "epsilon",
    "hbar", "omega", "gamma", "lambda", "tau",
    "vx", "vy", "theta", "delta_x", "delta_p",
    "n_i", "n_f", "f", "phi", "a", "e", "delta_phi_obs",
]
HV_QTY_TO_IDX = {q: i for i, q in enumerate(_HV_QUANTITY_VOCAB)}
NUM_HV_QUANTITIES = len(_HV_QUANTITY_VOCAB)


# =============================================================================
# Data Classes
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
    dimension_hint: str = "Scalar"


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
# 1. Error Shape Detector
# =============================================================================

class ErrorShapeDetector:
    """Analyze residuals from failed beam search to detect hidden structure."""

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
# 2. Hidden Variable Proposer (MLP)
# =============================================================================

class HiddenVariableProposer(nn.Module):
    """Small MLP that proposes hidden variables from error shape + context.
    Input: [shape_encoding(NUM_SHAPES) + quantity_vector(NUM_HV_QUANTITIES) + domain_onehot(NUM_HV_DOMAINS)]
    Hidden: 32 -> 32
    Output: [var_type_logits(NUM_VAR_TYPES) + transform_logits(NUM_TRANSFORMS) + confidence]
    ~3K parameters.
    """

    def __init__(self, *, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        input_dim = NUM_SHAPES + NUM_HV_QUANTITIES + NUM_HV_DOMAINS
        output_dim = NUM_VAR_TYPES + NUM_TRANSFORMS + 1

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
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
        temperature: float = 0.1,
    ) -> list[HiddenVariableProposal]:
        x = torch.cat([shape_encoding, quantity_vector, domain_onehot], dim=-1)
        output = self.forward(x)

        var_logits = output[:, :NUM_VAR_TYPES]
        transform_logits = output[:, NUM_VAR_TYPES:NUM_VAR_TYPES + NUM_TRANSFORMS]
        confidence_logit = output[:, -1]

        var_probs = F.softmax(var_logits / max(temperature, 1e-8), dim=-1)
        transform_probs = F.softmax(transform_logits / max(temperature, 1e-8), dim=-1)
        confidence = torch.sigmoid(confidence_logit)

        batch_size = x.size(0)
        qty_indices = quantity_vector.nonzero(as_tuple=False)
        name_map = {
            VAR_INTEGER: "n", VAR_HALF_INTEGER: "j",
            VAR_ANGULAR_M: "m_l", VAR_SPIN: "s", VAR_CONTINUOUS: "alpha",
        }
        patch_map = {
            TRANSFORM_IDENTITY: "*{name}", TRANSFORM_SQUARED: "*{name}^2",
            TRANSFORM_INV_SQUARED: "/{name}^2", TRANSFORM_SQRT: "*sqrt({name})",
        }

        results: list[HiddenVariableProposal] = []
        for b in range(batch_size):
            var_idx = var_probs[b].argmax().item()
            transform_idx = transform_probs[b].argmax().item()
            conf_val = confidence[b].item()
            var_type = IDX_TO_VAR_TYPE.get(var_idx, VAR_INTEGER)
            transform = IDX_TO_TRANSFORM.get(transform_idx, TRANSFORM_SQUARED)
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
            ))
        return results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Feature Construction
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
    return {
        SHAPE_QUADRATIC: [SHAPE_INVERSE_SQUARE, SHAPE_LINEAR],
        SHAPE_INVERSE_SQUARE: [SHAPE_QUADRATIC, SHAPE_CONSTANT],
        SHAPE_LINEAR: [SHAPE_QUADRATIC, SHAPE_CONSTANT],
        SHAPE_EXPONENTIAL: [SHAPE_QUADRATIC, SHAPE_LINEAR],
        SHAPE_PERIODIC: [SHAPE_RANDOM],
        SHAPE_CONSTANT: [SHAPE_LINEAR, SHAPE_INVERSE_SQUARE],
        SHAPE_RANDOM: [SHAPE_PERIODIC],
    }.get(shape, [])


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
# 3. Hidden Variable Discovery (Verification Loop)
# =============================================================================

class HiddenVariableDiscovery:
    """Orchestrate the hidden variable discovery pipeline.

    1. Run beam search with known quantities
    2. If no discovery, analyze error shapes
    3. Propose hidden variables via MLP + rule-based fallback
    4. Add each proposal to quantities, re-run search
    5. Return successful discovery or best effort
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
        proposals = self._propose(analysis, qty_names, domain)

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
        self, analysis: ErrorShapeAnalysis, quantity_names: list[str], domain: str,
    ) -> list[HiddenVariableProposal]:
        rule_proposals = self._rule_based_proposals(analysis, quantity_names, domain)
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

    def _rule_based_proposals(
        self, analysis: ErrorShapeAnalysis, quantity_names: list[str], domain: str,
    ) -> list[HiddenVariableProposal]:
        proposals: list[HiddenVariableProposal] = []
        shape = analysis.shape
        has_E = any("E" in q or q == "E" for q in quantity_names)
        has_lambda = any("l" in q.lower() or "lambda" in q for q in quantity_names)
        has_energy_like = has_E or any(q in quantity_names for q in ["E", "energy", "W", "Q"])
        is_quantum = domain == "quantum" or "hbar" in quantity_names or "omega" in quantity_names
        cvs = analysis.mean_cv

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
                if proposal.variable_name in ts or proposal.variable_name in obs.parameters:
                    new_timesteps.append(new_ts)  # Keep as-is, var already present
                    continue
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
                else:
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
# 4. Training Data and Training Loop
# =============================================================================

@dataclass
class HiddenVarTrainingExample:
    error_shape: str
    quantity_names: list[str]
    domain: str
    expected_var_type: str
    expected_transform: str
    description: str


def generate_synthetic_training_examples() -> list[HiddenVarTrainingExample]:
    """Generate ~60 synthetic training examples for the HiddenVariableProposer."""
    examples: list[HiddenVarTrainingExample] = []

    # Hydrogen spectrum variants (12)
    for i in range(12):
        qtys = [["E", "lambda"], ["E", "lambda", "hbar"],
                ["E", "lambda", "hbar", "omega"], ["E", "lambda", "c"]][i % 4]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_INVERSE_SQUARE, quantity_names=qtys,
            domain="quantum", expected_var_type=VAR_INTEGER,
            expected_transform=TRANSFORM_SQUARED,
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
            description=f"Particle in box variant {i+1}",
        ))

    # Harmonic oscillator (8) — half are integer, half are half-integer
    for i in range(4):
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_LINEAR, quantity_names=["E", "omega"],
            domain="quantum", expected_var_type=VAR_HALF_INTEGER,
            expected_transform=TRANSFORM_IDENTITY,
            description=f"QHO half-integer variant {i+1}",
        ))
    for i in range(4):
        qtys_v = [["E", "k", "m"], ["v", "omega", "x"], ["a", "omega", "t"], ["E", "t", "omega"]]
        examples.append(HiddenVarTrainingExample(
            error_shape=SHAPE_LINEAR, quantity_names=qtys_v[i],
            domain="quantum", expected_var_type=VAR_INTEGER,
            expected_transform=TRANSFORM_IDENTITY,
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
            description=f"No hidden variable (random) variant {i+1}",
        ))

    return examples


def build_training_batch(
    examples: list[HiddenVarTrainingExample],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(examples)
    input_dim = NUM_SHAPES + NUM_HV_QUANTITIES + NUM_HV_DOMAINS
    output_dim = NUM_VAR_TYPES + NUM_TRANSFORMS + 1
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
        targets[i, -1] = 1.0

    return inputs, targets


def train_hidden_var_proposer(
    *,
    proposer: HiddenVariableProposer | None = None,
    epochs: int = 200,
    lr: float = 0.003,
    device: str = "cpu",
    checkpoint_path: str | None = None,
) -> HiddenVariableProposer:
    if proposer is None:
        proposer = HiddenVariableProposer()
    proposer.to(device)
    proposer.train()

    examples = generate_synthetic_training_examples()
    inputs, targets = build_training_batch(examples)
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer = torch.optim.Adam(proposer.parameters(), lr=lr)
    var_loss_fn = nn.CrossEntropyLoss()
    transform_loss_fn = nn.CrossEntropyLoss()
    conf_loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = proposer(inputs)
        var_logits = output[:, :NUM_VAR_TYPES]
        transform_logits = output[:, NUM_VAR_TYPES:NUM_VAR_TYPES + NUM_TRANSFORMS]
        conf_logits = output[:, -1]

        var_targets = targets[:, :NUM_VAR_TYPES].argmax(dim=-1)
        transform_targets = targets[:, NUM_VAR_TYPES:NUM_VAR_TYPES + NUM_TRANSFORMS].argmax(dim=-1)
        conf_targets = targets[:, -1]

        loss = (var_loss_fn(var_logits, var_targets)
                + transform_loss_fn(transform_logits, transform_targets)
                + 0.1 * conf_loss_fn(conf_logits, conf_targets))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                var_acc = (var_logits.argmax(-1) == var_targets).float().mean()
                transform_acc = (transform_logits.argmax(-1) == transform_targets).float().mean()
            print(f"  epoch {epoch+1}/{epochs}  loss={loss.item():.4f}  "
                  f"var_acc={var_acc.item():.3f}  transform_acc={transform_acc.item():.3f}")

    proposer.eval()
    if checkpoint_path:
        save_path = Path(checkpoint_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": proposer.state_dict(),
            "num_shapes": NUM_SHAPES, "num_var_types": NUM_VAR_TYPES,
            "num_transforms": NUM_TRANSFORMS, "num_hv_domains": NUM_HV_DOMAINS,
            "num_hv_quantities": NUM_HV_QUANTITIES,
            "shape_to_idx": SHAPE_TO_IDX, "var_type_to_idx": VAR_TYPE_TO_IDX,
            "transform_to_idx": TRANSFORM_TO_IDX, "domain_to_idx": HV_DOMAIN_TO_IDX,
            "qty_to_idx": HV_QTY_TO_IDX,
        }, save_path)
        print(f"  Saved checkpoint to {save_path}")
    return proposer


def load_hidden_var_proposer(checkpoint_path: str, device: str = "cpu") -> HiddenVariableProposer:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    proposer = HiddenVariableProposer()
    proposer.load_state_dict(checkpoint["model_state_dict"])
    proposer.to(device)
    proposer.eval()
    return proposer


# =============================================================================
# 5. High-Level Pipeline
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
            max_depth=6, max_expansions=5000,
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
