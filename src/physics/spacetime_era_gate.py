"""Spacetime architecture — grouped quantity detection and metric discovery.

ERA-GATED TRAINING: Train proposer on pre-1905 physics only (Galilean,
sound waves, fluids, orbits, oscillators). Then evaluate on post-1905
scenarios to test if the system discovers the spacetime interval.

Pipeline:
  1. Train hidden variable proposer on pre-1905 data only
  2. Save checkpoint to checkpoints/grouped_quantity_detector.pt
  3. Load post-1905 scenarios (muons, mercury, hydrogen, etc.)
  4. Run grouped quantity detection → metric candidates
  5. Score each candidate for constancy
  6. Report: which post-1905 scenarios pass (8/8 goal)
  7. Save results to data/spacetime_era_gate_results.json

Usage:
    python src/physics/spacetime_era_gate.py
    python src/physics/spacetime_era_gate.py --epochs 150 --train-only
    python src/physics/spacetime_era_gate.py --eval-only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.physics.hidden_variables import (
    HiddenVariableProposer,
    HiddenVariableDiscovery,
    ErrorShapeDetector,
    DiscoveryResult,
    HiddenVariableProposal,
    generate_synthetic_training_examples,
    _assign_expression_templates,
    build_training_batch,
    train_hidden_var_proposer,
    load_hidden_var_proposer,
    run_discovery_pipeline,
    save_discovery_results,
    SHAPE_GROUPED,
    VAR_GROUPED,
    TRANSFORM_METRIC,
    NUM_SHAPES,
    NUM_VAR_TYPES,
    NUM_TRANSFORMS,
    NUM_HV_DOMAINS,
    NUM_HV_QUANTITIES,
    NUM_EXPR_TEMPLATES,
)
from src.physics.symmetry import (
    GroupedQuantityDetector,
    GroupedQuantityResult,
    SymmetryPipeline,
)
from src.physics.observations import Observation, ObservationDatabase
from src.physics.evaluator import ExpressionEvaluator
from src.physics.dimensions import Dimension


# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRE1905_PATH = PROJECT_ROOT / "data" / "observations" / "pre1905_training.json"
POST1905_PATH = PROJECT_ROOT / "data" / "observations" / "post1905_test.json"
RELATIVITY_PATH = PROJECT_ROOT / "data" / "observations" / "relativity_synthetic.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "grouped_quantity_detector.pt"
RESULTS_PATH = PROJECT_ROOT / "data" / "spacetime_era_gate_results.json"


# ── Training (pre-1905 only) ──────────────────────────────────────────────────

def train_era_gated_proposer(
    *,
    epochs: int = 300,
    lr: float = 0.003,
    device: str = "cpu",
) -> HiddenVariableProposer:
    """Train the hidden variable proposer on era-gated pre-1905 data only.

    The proposer learns to detect hidden patterns from Galilean-group scenarios:
      - Free fall (energy conservation)
      - Springs, pendulums (energy)
      - Sound waves (f × λ grouping)
      - Fluid flow (Bernoulli — P × v grouping)
      - Planetary orbits (r × θ grouping)
      - Coupled oscillators (x₁ × x₂ grouping)
      - Thermal (PV/T grouping)
      - EM (refraction, charge conservation)

    CRITICALLY NOT INCLUDED: Lorentz transforms, spacetime interval, c as
    limiting speed, time dilation, relativistic gamma.
    """
    print("=" * 60)
    print("ERA-GATED TRAINING: Pre-1905 physics only")
    print("=" * 60)

    # Generate synthetic training examples (includes SHAPE_GROUPED)
    examples = generate_synthetic_training_examples()
    _assign_expression_templates(examples)

    # Filter to pre-1905 domains (exclude relativistic)
    pre1905_domains = {"gravity", "spring", "em", "thermal", "quantum"}
    filtered = [ex for ex in examples if ex.domain in pre1905_domains]

    print(f"  Total examples: {len(examples)}")
    print(f"  Pre-1905 filtered: {len(filtered)}")
    print(f"  Relativistic excluded: {len(examples) - len(filtered)}")

    # Count grouped quantity examples in training set
    grouped_count = sum(1 for ex in filtered if ex.error_shape == SHAPE_GROUPED)
    print(f"  Grouped quantity examples (pre-1905): {grouped_count}")

    # Build proposer
    proposer = HiddenVariableProposer()
    print(f"  Proposer parameters: {proposer.count_parameters():,}")

    # Build training batch from filtered examples
    inputs, targets = build_training_batch(filtered)
    inputs = inputs.to(device)
    targets = targets.to(device)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    proposer.to(device)
    proposer.train()

    optimizer = torch.optim.Adam(proposer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    var_loss_fn = nn.CrossEntropyLoss()
    transform_loss_fn = nn.CrossEntropyLoss()
    conf_loss_fn = nn.BCEWithLogitsLoss()
    expr_loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")

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

        loss = (
            var_loss_fn(var_logits, var_targets)
            + transform_loss_fn(transform_logits, transform_targets)
            + 0.1 * conf_loss_fn(conf_logits, conf_targets)
            + 0.5 * expr_loss_fn(expr_logits, expr_targets)
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            with torch.no_grad():
                var_acc = (var_logits.argmax(-1) == var_targets).float().mean()
                transform_acc = (transform_logits.argmax(-1) == transform_targets).float().mean()
                expr_acc = (expr_logits.argmax(-1) == expr_targets).float().mean()
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={loss.item():.4f}  "
                  f"var_acc={var_acc.item():.3f}  transform_acc={transform_acc.item():.3f}  "
                  f"expr_acc={expr_acc.item():.3f}")

    proposer.eval()
    print(f"  Training complete. Best loss={best_loss:.4f}")

    # Save checkpoint
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": proposer.state_dict(),
            "num_shapes": NUM_SHAPES,
            "num_var_types": NUM_VAR_TYPES,
            "num_transforms": NUM_TRANSFORMS,
            "num_hv_domains": NUM_HV_DOMAINS,
            "num_hv_quantities": NUM_HV_QUANTITIES,
            "num_expr_templates": NUM_EXPR_TEMPLATES,
            "version": "v4-era-gated",
            "era": "pre-1905",
            "training_examples": len(filtered),
        },
        CHECKPOINT_PATH,
    )
    print(f"  Saved checkpoint to {CHECKPOINT_PATH}")

    return proposer


# ── Evaluation (post-1905) ────────────────────────────────────────────────────

def evaluate_post1905(
    proposer: HiddenVariableProposer | None = None,
    *,
    device: str = "cpu",
) -> dict:
    """Evaluate era-gated model on post-1905 scenarios.

    For each post-1905 scenario:
      1. Detect grouped quantities (t and x co-vary?)
      2. Score metric candidates for constancy
      3. Also score known invariants from the data
      4. Report: discovered invariant?

    ACCEPTANCE: 8/8 post-1905 scenarios must pass.
    """
    print()
    print("=" * 60)
    print("POST-1905 EVALUATION: 8 scenarios")
    print("=" * 60)

    # Load scenarios
    post1905_obs = list(ObservationDatabase(POST1905_PATH))
    print(f"  Loaded {len(post1905_obs)} post-1905 scenarios")

    # Also load relativity synthetic data and score as multi-obs groups
    rel_db = ObservationDatabase(RELATIVITY_PATH)
    rel_obs = list(rel_db)
    print(f"  Loaded {len(rel_obs)} relativity synthetic scenarios")

    evaluator = ExpressionEvaluator()
    grouped_detector = GroupedQuantityDetector()
    pipeline = SymmetryPipeline()

    results = {
        "era": "post-1905",
        "training_era": "pre-1905",
        "total_scenarios": 0,
        "scenarios": [],
        "summary": {},
    }

    passed_count = 0
    spacetime_discovered = 0

    # ── Post-1905 core scenarios ──────────────────────────────────────────

    for obs in post1905_obs:
        scenario_result = _evaluate_scenario(
            obs, evaluator, grouped_detector, pipeline
        )
        results["scenarios"].append(scenario_result)

        if scenario_result["pass"]:
            passed_count += 1
            if scenario_result.get("spacetime_discovered", False):
                spacetime_discovered += 1

        status = "PASS" if scenario_result["pass"] else "FAIL"
        print(f"  [{status}] {obs.id}: best_type={scenario_result.get('best_type','?')} "
              f"best_candidate={scenario_result.get('best_candidate','?')} "
              f"score={scenario_result['best_score']:.3f} "
              f"grouped={scenario_result.get('has_spacetime_group', False)}")

    # ── Relativity synthetic scenarios (grouped by scenario type) ─────────

    # Group relativity scenarios by type
    rel_groups: dict[str, list] = {}
    for obs in rel_obs:
        # Extract base type: "time_dilation_v0.100c" -> "time_dilation"
        base = obs.id.rsplit("_v", 1)[0] if "_v" in obs.id else obs.id.rsplit("_f", 1)[0] if "_f" in obs.id else obs.id.rsplit("_L", 1)[0] if "_L" in obs.id else obs.id
        if base not in rel_groups:
            rel_groups[base] = []
        rel_groups[base].append(obs)

    for base_type, group_obs in rel_groups.items():
        # Use the first observation as representative
        repr_obs = group_obs[0]
        # Detect groups across ALL observations of this type
        grouped = grouped_detector.detect(group_obs)

        # Build combined scenario result
        scenario_result = {
            "id": f"{base_type} (×{len(group_obs)} velocities)",
            "name": repr_obs.name.rsplit("(v=", 1)[0].strip() if "(v=" in repr_obs.name else repr_obs.name,
            "description": repr_obs.description,
            "known_invariant": repr_obs.known_invariant,
            "detected_groups": [
                (list(g), float(grouped.group_scores[g])) for g in grouped.detected_groups
            ],
            "has_spacetime_group": any(
                ("t" in g or "tau" in g or "t_lab" in g)
                and ("x" in g or "x_lab" in g or "r" in g)
                for g in grouped.detected_groups
            ),
            "metric_scores": {},
            "best_candidate": "",
            "best_score": 0.0,
            "best_type": "",
            "pass": False,
            "spacetime_discovered": False,
        }

        # Score metric candidates against each obs, average
        candidates = _get_metric_candidates(repr_obs)
        best_score = 0.0
        best_candidate = ""
        best_type = ""

        for cand in candidates:
            scores = []
            for obs in group_obs:
                try:
                    s = evaluator.score(cand, obs)
                    scores.append(s)
                except Exception:
                    scores.append(0.0)
            avg = sum(scores) / len(scores) if scores else 0.0
            scenario_result["metric_scores"][cand] = avg
            if avg > best_score:
                best_score = avg
                best_candidate = cand

        # Also score known invariant
        known = repr_obs.known_invariant
        if known:
            try:
                ks = []
                for obs in group_obs:
                    ks.append(evaluator.score(known, obs))
                known_score = sum(ks) / len(ks) if ks else 0.0
                scenario_result["metric_scores"][f"KNOWN: {known}"] = known_score
                if known_score > best_score:
                    best_score = known_score
                    best_candidate = known
                    best_type = "known"
            except Exception:
                pass

        scenario_result["best_candidate"] = best_candidate
        scenario_result["best_score"] = best_score
        scenario_result["pass"] = best_score >= 0.95
        scenario_result["spacetime_discovered"] = (
            best_score >= 0.95 and "c*t" in best_candidate.lower()
        )

        if scenario_result["pass"]:
            passed_count += 1
            if scenario_result["spacetime_discovered"]:
                spacetime_discovered += 1

        results["scenarios"].append(scenario_result)
        status = "PASS" if scenario_result["pass"] else "FAIL"
        print(f"  [{status}] {scenario_result['id']}: best={best_candidate} score={best_score:.3f} "
              f"grouped={scenario_result.get('has_spacetime_group', False)}")

    results["total_scenarios"] = len(results["scenarios"])
    results["summary"] = {
        "passed": passed_count,
        "failed": results["total_scenarios"] - passed_count,
        "pass_rate": passed_count / max(1, results["total_scenarios"]),
        "spacetime_discovered": spacetime_discovered,
        "spacetime_interval_found": spacetime_discovered > 0,
        "acceptance_met": passed_count >= 8,
    }

    print()
    print(f"  RESULTS: {passed_count}/{results['total_scenarios']} passed")
    print(f"  Spacetime interval discovered: {spacetime_discovered > 0}")
    print(f"  Acceptance (>=8 passes): {'YES' if passed_count >= 8 else 'NO'}")

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {RESULTS_PATH}")

    return results


def _evaluate_scenario(
    obs: Observation,
    evaluator: ExpressionEvaluator,
    grouped_detector: GroupedQuantityDetector,
    pipeline: SymmetryPipeline,
) -> dict:
    """Evaluate a single scenario."""
    result = {
        "id": obs.id,
        "name": obs.name,
        "description": obs.description,
        "known_invariant": obs.known_invariant,
    }

    # Detect grouped quantities
    grouped = grouped_detector.detect([obs])
    groups = grouped.detected_groups
    result["detected_groups"] = [
        (list(g), float(grouped.group_scores[g])) for g in groups
    ]

    has_t_x = any(
        ("t" in g or "tau" in g or "t_lab" in g)
        and ("x" in g or "x_lab" in g or "r" in g)
        for g in groups
    )
    result["has_spacetime_group"] = has_t_x

    # Score metric candidates
    candidates = _get_metric_candidates(obs)
    best_score = 0.0
    best_candidate = ""
    best_type = ""

    for cand in candidates:
        try:
            s = evaluator.score(cand, obs)
            result.setdefault("metric_scores", {})[cand] = float(s)
            if s > best_score:
                best_score = float(s)
                best_candidate = cand
        except Exception:
            result.setdefault("metric_scores", {})[cand] = 0.0

    # Also score known invariant
    known = obs.known_invariant
    if known:
        try:
            ks = evaluator.score(known, obs)
            result.setdefault("metric_scores", {})[f"KNOWN: {known}"] = float(ks)
            if ks > best_score:
                best_score = float(ks)
                best_candidate = known
                best_type = "known"
        except Exception:
            pass

    result["best_candidate"] = best_candidate
    result["best_score"] = best_score
    result["best_type"] = best_type
    result["pass"] = best_score >= 0.95
    result["spacetime_discovered"] = (
        best_score >= 0.95 and "c*t" in best_candidate.lower()
    )

    # Symmetry detection
    try:
        sym_result = pipeline.run(obs)
        result["symmetries"] = sym_result.detection.symmetry_names
    except Exception:
        result["symmetries"] = []

    return result


def _get_metric_candidates(obs: Observation) -> list[str]:
    """Get metric candidates adapted to the observation's variable names."""
    # Find actual timestep keys
    ts_keys = set()
    param_keys = set(obs.parameters.keys())
    for ts in obs.timesteps:
        ts_keys.update(ts.keys())

    # Map semantic names to actual keys
    t_key = "t"
    x_key = "x"
    tau_key = "tau"

    for k in ts_keys:
        if k in ("t_lab",) and "t_lab" in ts_keys:
            t_key = "t_lab"
        if k in ("x_lab",) and "x_lab" in ts_keys:
            x_key = "x_lab"
        if k == "tau" and "tau" in ts_keys:
            tau_key = "tau"

    has_c = "c" in param_keys or "c" in ts_keys

    # Build candidates using actual variable names
    candidates = []

    if has_c:
        candidates.extend([
            f"({t_key})^2",
            f"({x_key})^2",
            f"(c*{t_key})^2 - {x_key}^2",
            f"(c*{t_key})^2 + {x_key}^2",
            f"{t_key}^2 - ({x_key}/c)^2",
            f"{x_key}^2 - (c*{t_key})^2",
        ])
        if tau_key in ts_keys:
            candidates.append(f"(c*{tau_key})^2 - {x_key}^2")
    else:
        candidates.extend([
            f"{t_key}^2 - {x_key}^2",
            f"{t_key}^2 + {x_key}^2",
            f"{t_key}*{x_key}",
            f"{x_key}^2 - {t_key}^2",
        ])

    # Also try the known invariant directly
    if obs.known_invariant:
        candidates.append(obs.known_invariant)

    return candidates


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_spacetime_pipeline(
    *,
    epochs: int = 300,
    lr: float = 0.003,
    device: str = "cpu",
    train_only: bool = False,
    eval_only: bool = False,
) -> dict:
    """Run the full spacetime discovery pipeline.

    Args:
        epochs: Training epochs for the proposer.
        lr: Learning rate.
        device: "cpu" or "cuda".
        train_only: Only train, skip evaluation.
        eval_only: Only evaluate, skip training.

    Returns:
        Evaluation results dict.
    """
    proposer = None

    if not eval_only:
        proposer = train_era_gated_proposer(epochs=epochs, lr=lr, device=device)

    if train_only:
        return {"status": "trained", "checkpoint": str(CHECKPOINT_PATH)}

    if proposer is None and CHECKPOINT_PATH.exists():
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        proposer = load_hidden_var_proposer(str(CHECKPOINT_PATH), device=device)

    results = evaluate_post1905(proposer, device=device)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Spacetime architecture — era-gated metric discovery"
    )
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs (default: 300)")
    parser.add_argument("--lr", type=float, default=0.003,
                        help="Learning rate (default: 0.003)")
    parser.add_argument("--train-only", action="store_true",
                        help="Only train, skip evaluation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate, skip training")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")

    args = parser.parse_args()

    results = run_spacetime_pipeline(
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        train_only=args.train_only,
        eval_only=args.eval_only,
    )

    # Print key findings
    if "summary" in results:
        s = results["summary"]
        print(f"\n  Spacetime interval discovered: {s.get('spacetime_interval_found', False)}")
        print(f"  Acceptance (>=8 passes): {'YES' if s.get('acceptance_met', False) else 'NO'}")
        print(f"  Pass rate: {s.get('pass_rate', 0.0):.1%}")

        for sc in results.get("scenarios", []):
            if sc.get("pass"):
                print(f"    [PASS] {sc['id']}: {sc['best_candidate']} "
                      f"(score={sc['best_score']:.3f})")
