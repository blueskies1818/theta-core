#!/usr/bin/env python3
"""Generate self-play training data for hidden variable proposer.

Phase E: Train Hidden Variable Proposer from Self-Play Data.

Approach: Generate expressions, simulate observations, then corrupt
observation values with known hidden-variable patterns. Compute residual
signatures from the corruption, producing (features → var_type) pairs.

The key insight: when a hidden variable is missing from an expression,
the expression's per-observation constancy breaks in a characteristic
way. We simulate this by running expressions through per-observation
modulation and measuring the residual pattern.

Output: data/self_play_hidden_var_data.pt with training tensors.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

import torch

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.physics.expression_generator import (
    SelfPlayExpressionGenerator,
    PRE_1905_QUANTITY_DIMS,
    DOMAIN_LABELS,
    HIDDEN_VAR_TYPES as GEN_VAR_TYPES,
)
from src.physics.observation_simulator import simulate_observations
from src.physics.hidden_variables import (
    SHAPE_LINEAR, SHAPE_QUADRATIC, SHAPE_INVERSE_SQUARE,
    SHAPE_EXPONENTIAL, SHAPE_PERIODIC, SHAPE_RANDOM, SHAPE_CONSTANT,
    SHAPE_LINEAR_RATIO, SHAPE_POWER_LAW, SHAPE_MULTI_VAR,
    VAR_INTEGER, VAR_HALF_INTEGER, VAR_ANGULAR_M, VAR_SPIN,
    VAR_CONTINUOUS, VAR_CONTINUOUS_RATIO, VAR_CONTINUOUS_ADDITIVE,
    VAR_GROUPED,
    VAR_TYPES, VAR_TYPE_TO_IDX, NUM_VAR_TYPES,
    NUM_SHAPES, SHAPE_TO_IDX,
    NUM_HV_QUANTITIES, HV_QTY_TO_IDX,
    NUM_HV_DOMAINS, HV_DOMAIN_TO_IDX,
    ErrorShapeDetector,
)

# Map generator hidden var types to proposer var types
GEN_TO_PROPOSER_VAR = {
    "integer_n": VAR_INTEGER,
    "half_integer": VAR_HALF_INTEGER,
    "squared_n": VAR_INTEGER,       # integer with squared transform
    "angular_m": VAR_ANGULAR_M,
    "spin": VAR_SPIN,
    "ratio": VAR_CONTINUOUS_RATIO,
    "metric": VAR_GROUPED,
}

# Per-variable-type observation corruption functions.
# Each returns a list of per-observation modulation factors.


def _corrupt_integer_n(
    rng: random.Random, num_obs: int,
) -> list[float]:
    """Integer quantization: 1/n² pattern (quantum energy levels).

    The expression value is modulated ~1/n² where n is the missing integer.
    This produces a strong inverse_square residual pattern.
    """
    start = rng.randint(1, 3)  # starting n
    noise = rng.uniform(0.01, 0.05)
    return [1.0 / ((start + i) ** 2) + rng.uniform(-noise, noise)
            for i in range(num_obs)]


def _corrupt_half_integer(
    rng: random.Random, num_obs: int,
) -> list[float]:
    """Half-integer offsets: (n + 0.5) pattern (harmonic oscillator).

    Energy levels follow E ~ (n + 1/2), producing a strong linear pattern.
    """
    offset = rng.uniform(-0.3, 0.3)
    noise = rng.uniform(0.05, 0.15)
    return [0.5 + float(i) + offset + rng.uniform(-noise, noise)
            for i in range(num_obs)]


def _corrupt_angular_m(
    rng: random.Random, num_obs: int,
) -> list[float]:
    """Angular momentum-like: symmetric integer spacing with periodic modulation.

    Values alternate in a way detectable as periodic (-l ... +l spacing).
    """
    amp = rng.uniform(1.5, 3.0)
    freq = rng.uniform(0.3, 0.8)
    phase = rng.uniform(0.0, 3.14)
    noise = rng.uniform(0.05, 0.15)
    return [amp * math.sin(freq * i + phase) + rng.uniform(-noise, noise)
            for i in range(num_obs)]


def _corrupt_spin(
    rng: random.Random, num_obs: int,
) -> list[float]:
    """Spin-like: clustered near ±1/2 or ±1 (discrete but near-constant clusters).

    Produces a nearly constant pattern with small jumps between clusters.
    """
    clusters = [rng.choice([-1.0, -0.5, 0.5, 1.0]) for _ in range(num_obs)]
    noise = rng.uniform(0.02, 0.08)
    return [c + rng.uniform(-noise, noise) for c in clusters]


def _corrupt_continuous(
    rng: random.Random, num_obs: int,
) -> list[float]:
    """Pure random scaling — no discernible pattern (random shape)."""
    return [rng.uniform(0.1, 3.0) for _ in range(num_obs)]


def _corrupt_continuous_ratio(
    rng: random.Random, num_obs: int,
) -> list[float]:
    """Continuous ratio: exponential growth/decay pattern.

    Ratio variables (drag coefficients, refractive indices) produce
    smooth exponential curves in residuals.
    """
    rate = rng.uniform(-0.4, 0.4)  # positive=growth, negative=decay
    noise = rng.uniform(0.03, 0.1)
    return [math.exp(rate * i) + rng.uniform(-noise, noise)
            for i in range(num_obs)]


def _corrupt_continuous_additive(
    rng: random.Random, num_obs: int,
) -> list[float]:
    """Additive offset: quadratic curvature pattern.

    Work functions and offsets produce residuals with quadratic curvature
    (the offset appears as a quadratic deviation from linear expectation).
    """
    a = rng.uniform(-0.15, 0.15)
    b = rng.uniform(0.3, 1.0)
    c = rng.uniform(-1.0, 1.0)
    noise = rng.uniform(0.05, 0.12)
    return [a * i * i + b * i + c + rng.uniform(-noise, noise)
            for i in range(num_obs)]


def _corrupt_grouped(
    rng: random.Random, num_obs: int,
) -> list[float]:
    """Grouped quantity: strong quadratic pattern with correlated noise.

    Spacetime-like grouped quantities produce quadratic residuals
    (s² = (ct)² - x² type patterns).
    """
    a = rng.uniform(0.1, 0.4)
    noise = rng.uniform(0.02, 0.08)
    return [a * i * i + rng.uniform(-noise, noise) for i in range(num_obs)]


# Map proposer var types to corruption functions
VAR_CORRUPT_FUNCS = {
    VAR_INTEGER: _corrupt_integer_n,
    VAR_HALF_INTEGER: _corrupt_half_integer,
    VAR_ANGULAR_M: _corrupt_angular_m,
    VAR_SPIN: _corrupt_spin,
    VAR_CONTINUOUS: _corrupt_continuous,
    VAR_CONTINUOUS_RATIO: _corrupt_continuous_ratio,
    VAR_CONTINUOUS_ADDITIVE: _corrupt_continuous_additive,
    VAR_GROUPED: _corrupt_grouped,
}


def compute_residual_features(
    per_obs_values: list[float],
    observations: list,
    scored_expressions: dict[str, float] | None = None,
    *,
    detector: ErrorShapeDetector | None = None,
) -> dict:
    """Compute residual signature from per-observation constancy values.

    Features:
      - mean_constancy: mean of per-obs values
      - var_constancy: variance of per-obs values
      - cv_constancy: coefficient of variation
      - best_shape: best-fit curve type
      - shape_confidence: confidence in best fit
      - shape_probs: probability distribution over shapes
    """
    n = len(per_obs_values)
    if n < 2:
        return {
            "mean_constancy": 0.0,
            "var_constancy": 0.0,
            "cv_constancy": 0.0,
            "best_shape": SHAPE_RANDOM,
            "shape_confidence": 0.0,
            "shape_probs": [0.0] * NUM_SHAPES,
        }

    mean_val = sum(per_obs_values) / n
    var_val = sum((v - mean_val) ** 2 for v in per_obs_values) / n
    cv_val = math.sqrt(max(var_val, 0.0)) / max(abs(mean_val), 1e-12)

    # Determine best-fit curve shape
    if detector is None:
        detector = ErrorShapeDetector()

    x = [float(i + 1) for i in range(n)]
    fits = detector._fit_all_shapes(per_obs_values)

    best_shape = SHAPE_RANDOM
    best_confidence = 0.0
    shape_scores = {}
    for shape, fit in fits.items():
        shape_scores[shape] = fit.r_squared
        if fit.r_squared > best_confidence:
            best_confidence = fit.r_squared
            best_shape = shape

    # Build shape probability vector (softmax over R²)
    shape_probs = [0.0] * NUM_SHAPES
    shape_items = sorted(shape_scores.items(), key=lambda x: -x[1])
    for shape, r2 in shape_items:
        idx = SHAPE_TO_IDX.get(shape)
        if idx is not None and idx < NUM_SHAPES:
            shape_probs[idx] = r2
    # Normalize
    total = sum(shape_probs)
    if total > 1e-12:
        shape_probs = [p / total for p in shape_probs]

    return {
        "mean_constancy": mean_val,
        "var_constancy": var_val,
        "cv_constancy": cv_val,
        "best_shape": best_shape,
        "shape_confidence": best_confidence,
        "shape_probs": shape_probs,
        "all_shape_scores": shape_scores,
    }


def generate_selfplay_example(
    generator: SelfPlayExpressionGenerator,
    rng: random.Random,
    *,
    level: int | None = None,
    num_observations: int = 10,
    noise_frac: float = 0.03,
    detector: ErrorShapeDetector | None = None,
) -> dict | None:
    """Generate one self-play training example.

    Returns dict with: features, domain, quantities, var_type_label
    or None if generation fails.
    """
    if detector is None:
        detector = ErrorShapeDetector()

    # Step 1: Generate base expression (with hidden vars enabled)
    gen_level = level if level is not None else rng.randint(2, 4)
    try:
        gen_expr = generator.generate(gen_level)
    except Exception:
        return None

    expr_str = gen_expr.expression_str
    quantities_dim = gen_expr.quantities_dict

    # Convert quantities to {name: dim_name} format for simulator.
    # Must use valid named dimension strings (not compound like "kg/s²").
    _QTY_TO_SIM_DIM: dict[str, str] = {
        "m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity",
        "t": "Time", "k": "Force",  # spring constant approximated as Force
        "x": "Length", "E": "Energy", "P": "Pressure", "V": "Volume",
        "T": "Scalar", "n": "Scalar", "R": "Energy",
    }
    quantities_sim: dict[str, str] = {}
    for qname in quantities_dim:
        if qname in _QTY_TO_SIM_DIM:
            quantities_sim[qname] = _QTY_TO_SIM_DIM[qname]
        else:
            # Unknown quantity — try to resolve dimension name
            try:
                dim_name = str(quantities_dim[qname])
                # Test if Dimension.named accepts this string
                from src.physics.dimensions import Dimension
                Dimension.named(dim_name)
                quantities_sim[qname] = dim_name
            except Exception:
                quantities_sim[qname] = "Scalar"

    if len(quantities_sim) < 2:
        return None  # Too simple

    # Step 2: Simulate observations with the visible expression
    try:
        observations = simulate_observations(
            expr_str, quantities_sim,
            num_configs=num_observations,
            noise_frac=noise_frac,
            seed=rng.randint(0, 2**31 - 1),
        )
    except Exception:
        return None

    if not observations:
        return None

    # Step 3: Pick a hidden var type
    # Use the generator's hidden var type if one was injected
    if gen_expr.hidden_variables:
        gen_var_type = list(gen_expr.hidden_variables.keys())[0]
        target_var_type = GEN_TO_PROPOSER_VAR.get(gen_var_type, VAR_CONTINUOUS)
    else:
        # No hidden var injected → randomly pick one for training
        target_var_type = rng.choice(VAR_TYPES)

    # Step 4: Compute per-timestep constancy scores with corruption pattern
    # The simulator returns one Observation with many timesteps.
    # We apply per-timestep corruption to simulate a missing hidden variable,
    # then measure the residual pattern across timesteps.
    from src.physics.evaluator import ExpressionEvaluator
    evaluator = ExpressionEvaluator()

    obs = observations[0]
    num_ts = len(obs.timesteps)
    if num_ts < 4:
        return None

    # Generate per-timestep corruption factors based on variable type
    corrupt_fn = VAR_CORRUPT_FUNCS.get(target_var_type, _corrupt_continuous)
    mod_factors = corrupt_fn(rng, num_ts)

    # Parse expression once
    try:
        ast = evaluator.parse(expr_str)
    except Exception:
        return None

    # Evaluate expression at each timestep, apply per-timestep modulation
    modulated_values: list[float] = []
    for ts, factor in zip(obs.timesteps, mod_factors):
        context = {**obs.parameters, **ts}
        try:
            from src.physics.evaluator import evaluate_node
            val = evaluate_node(ast, context)
            if isinstance(val, (int, float)) and not math.isnan(val):
                modulated_values.append(float(val) * factor)
            else:
                modulated_values.append(0.0)
        except Exception:
            modulated_values.append(0.0)

    if len(modulated_values) < 4:
        return None

    # Pass raw modulated values to shape detector — the corruption pattern
    # (1/n², linear, periodic, etc.) IS the residual signature.
    # compute_residual_features will fit shapes and extract mean/var/cv metrics.
    per_obs_constancies = modulated_values

    # Step 5: Compute residual features
    # Shape detection uses raw modulated values (which carry the pattern)
    # But scalar constancy features should be in [0,1] range
    features = compute_residual_features(
        per_obs_constancies, observations, detector=detector,
    )

    # Override scalar features with proper constancy metrics
    # from the unmodulated expression values
    unmodulated_values: list[float] = []
    for ts in obs.timesteps:
        context = {**obs.parameters, **ts}
        try:
            val = evaluate_node(ast, context)
            if isinstance(val, (int, float)) and not math.isnan(val):
                unmodulated_values.append(float(val))
        except Exception:
            pass

    if unmodulated_values and len(unmodulated_values) >= 2:
        um = sum(unmodulated_values) / len(unmodulated_values)
        uv = sum((v - um) ** 2 for v in unmodulated_values) / len(unmodulated_values)
        ucv = math.sqrt(max(uv, 0.0)) / max(abs(um), 1e-12)
        actual_constancy = 1.0 / (1.0 + ucv)
        # Modulation strength: how much did corruption change the values
        mean_mod = sum(per_obs_constancies) / len(per_obs_constancies)
        var_mod = sum((v - mean_mod) ** 2 for v in per_obs_constancies) / len(per_obs_constancies)
        cv_mod = math.sqrt(max(var_mod, 0.0)) / max(abs(mean_mod), 1e-12)
        mod_strength = 1.0 / (1.0 + cv_mod)
    else:
        actual_constancy = 0.5
        uv = 0.1
        mod_strength = 0.5

    features["mean_constancy"] = actual_constancy  # in [0,1]
    features["var_constancy"] = min(uv, 1.0) / max(abs(um) if 'um' in dir() else 1.0, 1.0)
    features["cv_constancy"] = ucv if 'ucv' in dir() else 0.1
    features["shape_confidence"] = features.get("shape_confidence", 0.5)
    # Add modulation strength as additional signal
    features["modulation_strength"] = mod_strength

    # Step 6: Extract quantity and domain features
    qty_names = list(quantities_sim.keys())
    domain = gen_expr.domain_label
    if domain not in HV_DOMAIN_TO_IDX:
        domain = "unknown"

    return {
        "mean_constancy": features["mean_constancy"],
        "var_constancy": features["var_constancy"],
        "cv_constancy": features["cv_constancy"],
        "best_shape": features["best_shape"],
        "shape_confidence": features["shape_confidence"],
        "shape_probs": features["shape_probs"],
        "quantity_names": qty_names,
        "domain": domain,
        "target_var_type": target_var_type,
        "expression_str": expr_str,
        "complexity_level": gen_level,
        "has_hidden_variable": True,  # This example IS corrupted with a pattern
    }


def build_training_tensors(
    examples: list[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert examples to (inputs, targets) tensors for training.

    Input: [shape_encoding(NUM_SHAPES) + quantity_vector(NUM_HV_QUANTITIES) +
            domain_onehot(NUM_HV_DOMAINS) + mean_constancy + var_constancy +
            cv_constancy + shape_confidence]
    Target: [var_type_onehot(NUM_VAR_TYPES) + transform_onehot(NUM_TRANSFORMS) +
             confidence(1)]
    """
    n = len(examples)
    input_dim = NUM_SHAPES + NUM_HV_QUANTITIES + NUM_HV_DOMAINS + 4
    output_dim = NUM_VAR_TYPES + 1  # var_type + confidence

    inputs = torch.zeros(n, input_dim)
    targets = torch.zeros(n, output_dim)

    for i, ex in enumerate(examples):
        # Shape encoding (from shape probs or one-hot)
        shape_probs = ex.get("shape_probs", [0.0] * NUM_SHAPES)
        for j in range(min(len(shape_probs), NUM_SHAPES)):
            inputs[i, j] = shape_probs[j]

        # Scalar features (must be BEFORE quantity and domain vectors)
        # Matches SelfPlayProposer input order: [SHAPE|4_SCALARS|QUANTITIES|DOMAIN]
        feat_offset = NUM_SHAPES
        inputs[i, feat_offset + 0] = ex.get("mean_constancy", 0.0)
        inputs[i, feat_offset + 1] = ex.get("var_constancy", 0.0)
        inputs[i, feat_offset + 2] = ex.get("cv_constancy", 0.0)
        inputs[i, feat_offset + 3] = ex.get("shape_confidence", 0.0)

        # Quantity vector
        qty_offset = NUM_SHAPES + 4
        for qname in ex.get("quantity_names", []):
            if qname in HV_QTY_TO_IDX:
                inputs[i, qty_offset + HV_QTY_TO_IDX[qname]] = 1.0

        # Domain
        dom_offset = NUM_SHAPES + 4 + NUM_HV_QUANTITIES
        domain = ex.get("domain", "unknown")
        if domain in HV_DOMAIN_TO_IDX:
            inputs[i, dom_offset + HV_DOMAIN_TO_IDX[domain]] = 1.0

        # Target: var_type one-hot
        target_var = ex.get("target_var_type", VAR_CONTINUOUS)
        if target_var in VAR_TYPE_TO_IDX:
            targets[i, VAR_TYPE_TO_IDX[target_var]] = 1.0
        else:
            # No hidden variable case — uniform distribution
            targets[i, :NUM_VAR_TYPES] = 1.0 / NUM_VAR_TYPES

        # Confidence: 0.0 when no hidden variable (high constancy, constant shape)
        # 1.0 when there IS a hidden variable pattern to detect
        mean_c = ex.get("mean_constancy", 0.0)
        best_shape = ex.get("best_shape", SHAPE_RANDOM)
        has_hidden_var = ex.get("has_hidden_variable", True)
        if not has_hidden_var or (mean_c > 0.95 and best_shape == SHAPE_CONSTANT):
            targets[i, NUM_VAR_TYPES] = 0.0  # No hidden var → low confidence
        else:
            targets[i, NUM_VAR_TYPES] = 1.0

    return inputs, targets


def generate_dataset(
    target_examples: int = 50_000,
    *,
    seed: int = 42,
    noise_frac: float = 0.03,
    min_examples_per_type: int = 2000,
) -> list[dict]:
    """Generate self-play hidden variable training data."""
    rng = random.Random(seed)
    generator = SelfPlayExpressionGenerator(
        seed=rng.randint(0, 2**31 - 1),
        include_hidden_vars=True,
        hidden_var_probability=0.7,
    )
    detector = ErrorShapeDetector()

    examples: list[dict] = []
    counts_by_type: dict[str, int] = {vt: 0 for vt in VAR_TYPES}
    attempts = 0
    start_time = time.time()

    levels = [2, 2, 3, 3, 4]  # Bias toward levels 2-3

    print(f"Target: {target_examples} examples")
    print(f"Min per type: {min_examples_per_type}")
    print()

    while len(examples) < target_examples:
        attempts += 1
        level = rng.choice(levels)

        ex = generate_selfplay_example(
            generator, rng,
            level=level,
            num_observations=rng.randint(6, 15),
            noise_frac=noise_frac,
            detector=detector,
        )

        if ex is None:
            continue

        # Track per-type counts for balance
        vt = ex["target_var_type"]
        if counts_by_type[vt] >= target_examples // len(VAR_TYPES) * 2:
            # Skip over-sampled types
            if rng.random() < 0.7:
                continue

        examples.append(ex)
        counts_by_type[vt] += 1

        if len(examples) % 5000 == 0:
            elapsed = time.time() - start_time
            rate = len(examples) / max(elapsed, 1)
            eta = (target_examples - len(examples)) / max(rate, 1e-6)
            print(f"  {len(examples):,d} / {target_examples:,d}  "
                  f"({len(examples)/target_examples*100:.1f}%)  "
                  f"attempts={attempts:,d}  "
                  f"rate={rate:.1f}/s  ETA={eta:.0f}s")
            print(f"    Counts by type: {counts_by_type}")

    elapsed = time.time() - start_time
    print(f"\\nDone! Generated {len(examples):,d} examples in {elapsed:.1f}s")
    print(f"Attempts: {attempts:,d}, success rate: {len(examples)/attempts*100:.1f}%")
    print(f"Counts by type: {counts_by_type}")

    # Add negative examples (no hidden variable) — ~10% of dataset
    neg_count = target_examples // 10
    print(f"\\nAdding {neg_count} negative examples (no hidden variable)...")
    domains = list(HV_DOMAIN_TO_IDX.keys())
    const_shapes = [SHAPE_CONSTANT, SHAPE_RANDOM]

    for _ in range(neg_count):
        qty_count = rng.randint(2, 5)
        qty_names = rng.sample(list(HV_QTY_TO_IDX.keys()), min(qty_count, len(HV_QTY_TO_IDX)))
        domain = rng.choice(domains)
        shape = rng.choice(const_shapes)

        examples.append({
            "mean_constancy": rng.uniform(0.90, 0.999),
            "var_constancy": rng.uniform(0.0, 0.01),
            "cv_constancy": rng.uniform(0.0, 0.05),
            "best_shape": shape,
            "shape_confidence": rng.uniform(0.7, 0.99),
            "shape_probs": [0.0] * NUM_SHAPES,
            "quantity_names": qty_names,
            "domain": domain,
            "target_var_type": "",  # No hidden variable
            "expression_str": "none",
            "complexity_level": 0,
            "has_hidden_variable": False,
        })
        # Set shape prob
        if shape in SHAPE_TO_IDX:
            examples[-1]["shape_probs"][SHAPE_TO_IDX[shape]] = 1.0

    print(f"  Added {neg_count} negative examples. Total: {len(examples):,d}")

    return examples


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate self-play hidden variable training data"
    )
    parser.add_argument(
        "--target", type=int, default=50_000,
        help="Target number of examples (default: 50000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(_project_root / "data" / "self_play_hidden_var_data.pt"),
        help="Output path for training tensors"
    )
    parser.add_argument(
        "--noise", type=float, default=0.03,
        help="Observation noise fraction (default: 0.03)"
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Number of CPU threads (default: 4)"
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    print(f"Generating {args.target:,d} self-play hidden var examples...")
    print(f"Seed: {args.seed}, noise: {args.noise}, threads: {args.threads}")
    print()

    examples = generate_dataset(
        target_examples=args.target,
        seed=args.seed,
        noise_frac=args.noise,
    )

    # Build and save tensors
    print("\nBuilding training tensors...")
    inputs, targets = build_training_tensors(examples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save tensors + metadata
    torch.save({
        "inputs": inputs,
        "targets": targets,
        "num_examples": len(examples),
        "num_shapes": NUM_SHAPES,
        "num_var_types": NUM_VAR_TYPES,
        "num_hv_quantities": NUM_HV_QUANTITIES,
        "num_hv_domains": NUM_HV_DOMAINS,
        "shape_to_idx": SHAPE_TO_IDX,
        "var_type_to_idx": VAR_TYPE_TO_IDX,
        "domain_to_idx": HV_DOMAIN_TO_IDX,
        "qty_to_idx": HV_QTY_TO_IDX,
        "examples_summary": examples[:10],  # Save first 10 for inspection
        "version": "self_play_v1",
    }, output_path)

    print(f"\nSaved {len(examples):,d} examples to {output_path}")
    print(f"Input tensor:  {inputs.shape}")
    print(f"Target tensor: {targets.shape}")

    # Quick stats
    var_counts = {}
    for ex in examples:
        vt = ex["target_var_type"]
        var_counts[vt] = var_counts.get(vt, 0) + 1
    print(f"\nFinal var type distribution:")
    for vt in VAR_TYPES:
        print(f"  {vt}: {var_counts.get(vt, 0)}")


if __name__ == "__main__":
    main()
