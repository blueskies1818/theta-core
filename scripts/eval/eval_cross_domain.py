#!/usr/bin/env python3
"""Cross-domain generalization evaluation for Phase F.

Tests whether the trained expression model can compose invariants
across domains it never saw combined during training.

THE TEST:
  Train on: gravity-only data + spring-only data (NEVER combined)
  Test on: gravity+spring scenario (mass on spring under gravity)
  Model must propose: m*g*h + 0.5*m*v^2 + 0.5*k*x^2
  (or equivalently: 0.5*m*v^2 + 0.5*k*h^2 - m*g*h)
  WITHOUT ever seeing that combination in training.

Also tests held-out scenario types for expression validity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from src.physics.model import (
    ExpressionTokenizer,
    ExpressionSequenceModel,
    extract_training_examples,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
)
from src.physics.evaluator import ExpressionEvaluator
from src.physics.grammar import Expression


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple:
    """Load model and tokenizer from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    tokenizer_path = checkpoint_path.parent / "tokenizer.json"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if tokenizer_path.exists():
        tokenizer = ExpressionTokenizer.load(tokenizer_path)
    else:
        tokenizer = ExpressionTokenizer()

    model = ExpressionSequenceModel(
        vocab_size=checkpoint.get("vocab_size", tokenizer.vocab_size),
        d_model=checkpoint.get("d_model", 128),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, tokenizer


def generate_candidates(
    model: ExpressionSequenceModel,
    tokenizer: ExpressionTokenizer,
    quantities: dict[str, str],
    scenario_type: str,
    num_samples: int = 10,
    temperature: float = 0.8,
    device: torch.device = torch.device("cpu"),
) -> list[str]:
    """Generate multiple candidate expressions for a given scenario.

    Returns list of unique expression strings.
    """
    qty_symbols = sorted(quantities.keys())
    src_with_scenario = qty_symbols + [scenario_type]

    # Build source tensor
    src_ids = [tokenizer.encode(s) for s in src_with_scenario]
    src_ids = src_ids[:16]
    src = torch.tensor(
        src_ids + [PAD_IDX] * (16 - len(src_ids)),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    src_mask = (src == PAD_IDX)

    candidates: set[str] = set()
    with torch.no_grad():
        for _ in range(num_samples * 2):  # Try more since sampling is stochastic
            if len(candidates) >= num_samples:
                break
            gen = model.generate(
                src,
                src_padding_mask=src_mask,
                max_len=64,
                temperature=temperature,
            )
            expr = tokenizer.detokenize_expression(gen[0])
            # Clean up: remove spaces around operators
            expr = _clean_expression(expr)
            if expr and len(expr) > 0:
                candidates.add(expr)

    return list(candidates)[:num_samples]


def _clean_expression(expr: str) -> str:
    """Clean up generated expression string.

    Removes spaces around operators for consistency, drops spurious tokens.
    """
    # Remove spaces that our detokenizer adds
    expr = expr.replace(" + ", "+").replace(" - ", "-")
    expr = expr.replace(" * ", "*").replace(" / ", "/")
    expr = expr.replace(" ^ ", "^")
    # Remove leading/trailing operators
    expr = expr.strip("+-*/^ ")
    # Remove SOS/EOS artifacts
    expr = expr.replace("<sos>", "").replace("<eos>", "")
    return expr.strip()


def score_candidates(
    candidates: list[str],
    observation_db_path: str,
    target_scenario_id: str,
) -> list[tuple[str, float]]:
    """Score candidate expressions using the constancy evaluator.

    Returns list of (expression, constancy_score) sorted by score descending.
    """
    from src.physics.observations import ObservationDatabase

    try:
        db = ObservationDatabase(observation_db_path)
        obs = db.get(target_scenario_id)
        evaluator = ExpressionEvaluator()
    except Exception as e:
        print(f"  Warning: Could not create evaluator: {e}")
        return [(c, 0.0) for c in candidates]

    scored: list[tuple[str, float]] = []
    for expr_str in candidates:
        try:
            score_val = evaluator.score(expr_str, obs)
            score_val = float(score_val)
        except Exception:
            score_val = 0.0
        scored.append((expr_str, score_val))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def run_cross_domain_eval(
    model_path: str | Path,
    observations_path: str | Path,
    output_path: str | Path | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run the cross-domain generalization evaluation.

    Returns evaluation results dict.
    """
    model, tokenizer = load_model(model_path, device)
    print(f"Model loaded: {model.count_parameters():,} params")

    # Load observation data to get scenario info
    with open(observations_path) as f:
        observations = json.load(f)

    # Build lookup
    obs_by_id = {o["id"]: o for o in observations}

    results = {
        "model_path": str(model_path),
        "observations_path": str(observations_path),
        "model_params": model.count_parameters(),
        "tests": {},
    }

    # ── Test 1: Cross-domain composition ─────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 1: Cross-domain composition (gravity + spring)")
    print("=" * 60)

    target_id = "mass_spring_gravity"
    if target_id not in obs_by_id:
        print(f"  WARNING: {target_id} not found in observations")
    else:
        obs = obs_by_id[target_id]
        print(f"  Scenario: {obs['name']}")
        print(f"  Quantities: {list(obs['quantities'].keys())}")
        print(f"  Known invariant: {obs['known_invariant']}")

        candidates = generate_candidates(
            model, tokenizer,
            quantities=obs["quantities"],
            scenario_type="gravity_spring",
            num_samples=10,
            temperature=0.8,
            device=device,
        )

        print(f"\n  Generated {len(candidates)} candidates:")
        for i, c in enumerate(candidates):
            print(f"    {i+1}. {c}")

        # Score candidates
        try:
            scored = score_candidates(
                candidates,
                str(observations_path),
                target_id,
            )
            print(f"\n  Scored candidates:")
            for expr, score in scored[:5]:
                marker = " ★" if score > 0.9 else ""
                print(f"    {score:.4f} — {expr}{marker}")

            best_expr, best_score = scored[0] if scored else ("", 0.0)
            results["tests"]["cross_domain"] = {
                "scenario_id": target_id,
                "known_invariant": obs["known_invariant"],
                "num_candidates": len(candidates),
                "best_expression": best_expr,
                "best_score": best_score,
                "all_scored": [(e, s) for e, s in scored[:5]],
                "passed": best_score > 0.7,
            }
            print(f"\n  Best: {best_expr} (score={best_score:.4f})")
            print(f"  Known: {obs['known_invariant']}")
            if best_score > 0.7:
                print(f"  RESULT: PASS (score > 0.7)")
            else:
                print(f"  RESULT: FAIL (score <= 0.7)")
        except Exception as e:
            print(f"  Error scoring: {e}")
            results["tests"]["cross_domain"] = {
                "scenario_id": target_id,
                "error": str(e),
                "passed": False,
            }

    # ── Test 2: Held-out scenario types ──────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: Held-out scenario types")
    print("=" * 60)

    # Pick one scenario from each held-out domain
    held_out_scenarios = {
        "collision_elastic_1d_equal_mass": "collision",
        "charged_particle_gravity": "em_gravity",
        "pendulum_small_angle": "pendulum",
    }

    for scenario_id, scenario_type in held_out_scenarios.items():
        if scenario_id not in obs_by_id:
            print(f"  {scenario_id}: not found")
            continue

        obs = obs_by_id[scenario_id]
        print(f"\n  Scenario: {obs['name']} ({scenario_type})")
        print(f"  Quantities: {list(obs['quantities'].keys())}")
        print(f"  Known invariant: {obs.get('known_invariant', 'N/A')}")

        candidates = generate_candidates(
            model, tokenizer,
            quantities=obs["quantities"],
            scenario_type=scenario_type,
            num_samples=5,
            temperature=0.8,
            device=device,
        )

        print(f"  Candidates: {candidates[:3]}")

        try:
            scored = score_candidates(
                candidates, str(observations_path), scenario_id,
            )
            if scored:
                best_expr, best_score = scored[0]
                passed = best_score > 0.7
                print(f"  Best: {best_expr} (score={best_score:.4f}) {'PASS' if passed else 'FAIL'}")
                results["tests"][scenario_id] = {
                    "best_expression": best_expr,
                    "best_score": best_score,
                    "known_invariant": obs.get("known_invariant"),
                    "passed": passed,
                }
        except Exception as e:
            print(f"  Error: {e}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    test_results = results["tests"]
    n_tests = len(test_results)
    n_passed = sum(1 for t in test_results.values() if t.get("passed", False))
    print(f"  Tests: {n_passed}/{n_tests} passed")

    if "cross_domain" in test_results:
        cd = test_results["cross_domain"]
        print(f"  Cross-domain composition: {'PASS' if cd.get('passed') else 'FAIL'}")
        if cd.get("best_score"):
            print(f"    Score: {cd['best_score']:.4f}")
            print(f"    Expression: {cd.get('best_expression', 'N/A')}")

    results["n_tests"] = n_tests
    results["n_passed"] = n_passed

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {output_path}")

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-domain generalization evaluation for Phase F"
    )
    parser.add_argument(
        "--model",
        default="checkpoints/expression_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--observations",
        default="data/observations/phase2_extended.json",
        help="Path to observation database",
    )
    parser.add_argument(
        "--output",
        default="data/phase_f_cross_domain.json",
        help="Path for evaluation results output",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature for generation",
    )
    args = parser.parse_args()

    project_root = _project_root
    model_path = project_root / args.model
    obs_path = project_root / args.observations
    output_path = project_root / args.output

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run training first: python scripts/training/train_expression_model.py --cross-domain")
        sys.exit(1)

    results = run_cross_domain_eval(
        model_path=str(model_path),
        observations_path=str(obs_path),
        output_path=str(output_path),
        device=torch.device("cpu"),
    )

    # Exit code based on cross-domain test
    cd_test = results["tests"].get("cross_domain", {})
    if cd_test.get("passed", False):
        print("\n✓ Cross-domain generalization test PASSED")
        sys.exit(0)
    else:
        print("\n✗ Cross-domain generalization test FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
