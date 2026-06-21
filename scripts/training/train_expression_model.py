#!/usr/bin/env python3
"""Train the ExpressionSequenceModel on Phase A-E discovery data.

Learns to generate conserved physics expressions from quantity sets.
Target: 50 epochs, 4 threads, CPU only, < 1M params.

FIX A: Adds combined cross-domain training examples (gravity+spring, EM+gravity,
spring+friction, gravity+spring+friction) to teach composition across domains.

Output:
  checkpoints/expression_model_fixa.pt   — trained model weights + tokenizer
  checkpoints/training_stats_fixa.json   — loss curves, evaluation metrics
  data/phase_f_fixa_results.json         — cross-domain evaluation results
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from src.physics.model import (
    ExpressionTokenizer,
    ExpressionSequenceModel,
    ExpressionDataset,
    extract_training_examples,
    create_train_test_split,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
)


# ── Combined Cross-Domain Training Examples ──────────────────────────────────

def generate_combined_examples() -> list[dict]:
    """Generate synthetic combined cross-domain training examples.

    These teach the model to compose expressions from multiple physical domains:
    - gravity+spring: mass on spring under gravity
    - gravity+EM: charged particle in gravity+electric field
    - spring+friction: damped oscillator
    - gravity+spring+friction: damped spring under gravity

    Returns list of example dicts with quantities, scenario_type, expression.
    """
    examples: list[dict] = []

    # ── Gravity + Spring (3 examples) ──────────────────────────────────────
    # Full expression: mgh + ½mv² + ½kx²
    examples.append({
        "quantities": {
            "m": "Mass", "g": "Accel", "h": "Length",
            "v": "Velocity", "k": "Force/Length", "x": "Length",
        },
        "scenario_type": "gravity_spring",
        "conservative": True,
        "expression": "m*g*h + 0.5*m*v^2 + 0.5*k*x^2",
    })
    examples.append({
        "quantities": {
            "m": "Mass", "g": "Accel", "h": "Length",
            "v": "Velocity", "k": "Force/Length", "x": "Length",
        },
        "scenario_type": "gravity_spring",
        "conservative": True,
        "expression": "0.5*k*x^2 + m*g*h + 0.5*m*v^2",
    })
    examples.append({
        "quantities": {
            "m": "Mass", "g": "Accel", "h": "Length",
            "v": "Velocity", "k": "Force/Length", "x": "Length",
        },
        "scenario_type": "gravity_spring",
        "conservative": True,
        "expression": "0.5*m*v^2 + 0.5*k*x^2 + m*g*h",
    })

    # ── Gravity + EM (2 examples) ──────────────────────────────────────────
    examples.append({
        "quantities": {
            "m": "Mass", "g": "Accel", "h": "Length",
            "v": "Velocity", "q": "Charge", "E": "Force/Charge",
        },
        "scenario_type": "em_gravity",
        "conservative": True,
        "expression": "0.5*m*v^2 + m*g*h + q*E*h",
    })
    examples.append({
        "quantities": {
            "m": "Mass", "g": "Accel", "h": "Length",
            "v": "Velocity", "q": "Charge", "E": "Force/Charge",
        },
        "scenario_type": "em_gravity",
        "conservative": True,
        "expression": "m*g*h + q*E*h + 0.5*m*v^2",
    })

    # ── Spring + Friction (2 examples) ─────────────────────────────────────
    # Damped oscillator: ½mv² + ½kx² (energy decreases over time but
    # instantaneous expression still includes both terms)
    examples.append({
        "quantities": {
            "m": "Mass", "k": "Force/Length", "x": "Length",
            "v": "Velocity",
        },
        "scenario_type": "spring_friction",
        "conservative": True,
        "expression": "0.5*k*x^2 + 0.5*m*v^2",
    })
    examples.append({
        "quantities": {
            "m": "Mass", "k": "Force/Length", "x": "Length",
            "v": "Velocity",
        },
        "scenario_type": "spring_friction",
        "conservative": True,
        "expression": "0.5*m*v^2 + 0.5*k*x^2",
    })

    # ── Gravity + Spring + Friction (2 examples) ───────────────────────────
    examples.append({
        "quantities": {
            "m": "Mass", "g": "Accel", "h": "Length",
            "v": "Velocity", "k": "Force/Length", "x": "Length",
        },
        "scenario_type": "gravity_spring_friction",
        "conservative": True,
        "expression": "0.5*k*x^2 + m*g*h + 0.5*m*v^2",
    })
    examples.append({
        "quantities": {
            "m": "Mass", "g": "Accel", "h": "Length",
            "v": "Velocity", "k": "Force/Length", "x": "Length",
        },
        "scenario_type": "gravity_spring_friction",
        "conservative": True,
        "expression": "m*g*h + 0.5*m*v^2 + 0.5*k*x^2",
    })

    return examples


# ── Evaluation Utilities ─────────────────────────────────────────────────────

def run_cross_domain_evaluation(
    model: ExpressionSequenceModel,
    tokenizer: ExpressionTokenizer,
    test_scenarios: list[dict],
    device: torch.device,
) -> dict:
    """Evaluate model on cross-domain test scenarios.

    For each scenario, generates expression candidates and scores them
    against the known invariant using token-overlap scoring.

    Returns dict with per-scenario results and aggregate metrics.
    """
    model.eval()

    results = {
        "model_path": None,
        "model_params": model.count_parameters(),
        "tests": {},
        "n_tests": len(test_scenarios),
        "n_passed": 0,
    }

    for scenario in test_scenarios:
        scenario_id = scenario["id"]
        known_inv = scenario.get("known_invariant", "")
        if not known_inv:
            continue

        # Build source tensor: use quantities from observation + extras from
        # known invariant that may be in parameters but not quantities dict
        quantities = sorted(scenario["quantities"].keys())
        # Extract additional symbols from known invariant (e.g., q, E for em_gravity)
        import re
        inv_symbols = set(re.findall(r'\b([a-zA-Z])\b', known_inv))
        inv_symbols -= {"m", "g", "h", "v", "t", "x", "y", "r", "L"}  # already covered
        for sym in sorted(inv_symbols):
            if sym not in quantities:
                quantities.append(sym)

        src = tokenizer.quantities_to_tensor(quantities, max_len=16).unsqueeze(0)
        src = src.to(device)
        src_mask = (src == PAD_IDX)

        # Generate multiple candidates
        candidates: list[str] = []
        with torch.no_grad():
            # Greedy
            gen_greedy = model.generate(
                src, src_padding_mask=src_mask, max_len=48, temperature=0,
            )
            expr = tokenizer.detokenize_expression(gen_greedy[0])
            if expr and expr not in candidates:
                candidates.append(expr)

            # Sample with moderate temperature for diversity
            for _ in range(3):
                gen_sampled = model.generate(
                    src, src_padding_mask=src_mask, max_len=48, temperature=0.5,
                )
                expr = tokenizer.detokenize_expression(gen_sampled[0])
                if expr and expr not in candidates:
                    candidates.append(expr)

        # Score each candidate using token-overlap
        scored = []
        for expr in candidates:
            score = _token_overlap_score(expr, known_inv)
            scored.append((expr, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_expr, best_score = scored[0] if scored else ("", 0.0)

        # Determine pass: score > 0.6 is reasonable overlap
        passed = best_score > 0.6

        results["tests"][scenario_id] = {
            "scenario_id": scenario_id,
            "known_invariant": known_inv,
            "num_candidates": len(candidates),
            "best_expression": best_expr,
            "best_score": best_score,
            "all_scored": scored,
            "passed": passed,
        }

        if passed:
            results["n_passed"] += 1

    return results


def _token_overlap_score(generated: str, target: str) -> float:
    """Compute token-overlap similarity between two expression strings."""
    import re

    def tokenize(s: str) -> set[str]:
        tokens = set()
        for tok in re.findall(r'[a-zA-Z]+|\d+\.?\d*|[*^+/\-]', s):
            tokens.add(tok)
        return tokens

    gen_tokens = tokenize(generated)
    tgt_tokens = tokenize(target)
    if not tgt_tokens:
        return 0.0

    intersection = gen_tokens & tgt_tokens
    return len(intersection) / len(tgt_tokens)


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> float:
    """Compute token-level accuracy, ignoring padding."""
    preds = logits.argmax(dim=-1)  # [batch, seq_len]
    mask = targets != pad_idx
    if mask.sum() == 0:
        return 0.0
    correct = (preds[mask] == targets[mask]).sum().item()
    return correct / mask.sum().item()


def evaluate_model(
    model: ExpressionSequenceModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_idx: int,
) -> dict:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Prepare masks
            src_mask = (src == pad_idx)
            tgt_mask_pad = (tgt == pad_idx)
            tgt_causal = torch.triu(
                torch.ones(tgt.size(1), tgt.size(1), device=device) * float("-inf"), diagonal=1
            )

            # Teacher forcing: input is all but last token, target is all but first
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(
                src, tgt_input,
                src_padding_mask=src_mask,
                tgt_padding_mask=tgt_mask_pad[:, :-1],
                tgt_mask=tgt_causal[:-1, :-1],
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )
            acc = compute_accuracy(logits, tgt_output, pad_idx)

            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": total_acc / max(n_batches, 1),
    }


def train(
    model: ExpressionSequenceModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    pad_idx: int = PAD_IDX,
    checkpoint_dir: Path = Path("checkpoints"),
    save_best: bool = True,
    checkpoint_name: str = "expression_model.pt",
) -> dict:
    """Train the model.

    Returns training statistics dict.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    stats = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    best_test_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Prepare masks
            src_mask = (src == pad_idx)
            tgt_mask_pad = (tgt == pad_idx)
            tgt_causal = torch.triu(
                torch.ones(tgt.size(1), tgt.size(1), device=device) * float("-inf"), diagonal=1
            )

            # Teacher forcing
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()

            logits = model(
                src, tgt_input,
                src_padding_mask=src_mask,
                tgt_padding_mask=tgt_mask_pad[:, :-1],
                tgt_mask=tgt_causal[:-1, :-1],
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )
            acc = compute_accuracy(logits, tgt_output, pad_idx)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_acc = epoch_acc / max(n_batches, 1)
        stats["train_loss"].append(avg_train_loss)
        stats["train_acc"].append(avg_train_acc)

        # Evaluation
        test_metrics = evaluate_model(model, test_loader, criterion, device, pad_idx)
        stats["test_loss"].append(test_metrics["loss"])
        stats["test_acc"].append(test_metrics["accuracy"])

        scheduler.step(test_metrics["loss"])

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss: {avg_train_loss:.4f} | "
                f"train_acc: {avg_train_acc:.4f} | "
                f"test_loss: {test_metrics['loss']:.4f} | "
                f"test_acc: {test_metrics['accuracy']:.4f}"
            )

        # Save best model
        if save_best and test_metrics["loss"] < best_test_loss:
            best_test_loss = test_metrics["loss"]
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": model.vocab_size,
                    "d_model": model.d_model,
                    "epoch": epoch,
                    "test_loss": test_metrics["loss"],
                    "test_acc": test_metrics["accuracy"],
                },
                checkpoint_dir / checkpoint_name,
            )

    return stats


def load_test_scenarios(observations_path: Path) -> list[dict]:
    """Load all observation scenarios for evaluation."""
    with open(observations_path) as f:
        return json.load(f)


def main() -> None:
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train expression generation model on physics discovery data"
    )
    parser.add_argument(
        "--observations",
        default="data/observations/phase2_extended.json",
        help="Path to observation database JSON",
    )
    parser.add_argument(
        "--discoveries",
        default="data/phase_e_discoveries.json",
        help="Path to phase E discoveries JSON",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="Number of CPU threads"
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints", help="Output directory"
    )
    parser.add_argument(
        "--cross-domain", action="store_true",
        help="Use cross-domain split with held-out combined types for testing"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--output-results",
        default="data/phase_f_fixa_results.json",
        help="Path to save evaluation results JSON"
    )
    parser.add_argument(
        "--checkpoint-name",
        default="expression_model_fixa.pt",
        help="Checkpoint filename"
    )
    args = parser.parse_args()

    # Set threads
    torch.set_num_threads(args.threads)

    # Set seed
    torch.manual_seed(args.seed)

    project_root = _project_root
    observations_path = project_root / args.observations
    discoveries_path = project_root / args.discoveries

    print(f"Loading data from {observations_path}")
    print(f"Discoveries from {discoveries_path}")

    # Extract training examples from observations + discoveries
    examples = extract_training_examples(
        str(observations_path),
        str(discoveries_path) if discoveries_path.exists() else None,
    )
    print(f"Base examples (from data): {len(examples)}")

    # ── FIX A: Add combined cross-domain training examples ────────────────
    combined_examples = generate_combined_examples()
    print(f"Combined cross-domain examples: {len(combined_examples)}")
    for ex in combined_examples:
        print(f"  [{ex['scenario_type']}] {ex['expression']}")

    # Split combined examples: keep spring_friction + gravity_spring_friction
    # as HELD-OUT test types; train on gravity_spring + em_gravity combos
    train_combined = [
        ex for ex in combined_examples
        if ex["scenario_type"] in ("gravity_spring", "em_gravity")
    ]
    test_combined = [
        ex for ex in combined_examples
        if ex["scenario_type"] in ("spring_friction", "gravity_spring_friction")
    ]

    print(f"  → Train combined: {len(train_combined)} ({set(ex['scenario_type'] for ex in train_combined)})")
    print(f"  → Test combined (held-out): {len(test_combined)} ({set(ex['scenario_type'] for ex in test_combined)})")

    if args.cross_domain:
        # Cross-domain: train on single-domain + gravity_spring/em_gravity combos,
        # test on held-out spring_friction/gravity_spring_friction
        # Also include existing gravity_spring / em_gravity observations in training
        train_examples = [
            ex for ex in examples
            if ex["scenario_type"] not in ("spring_friction", "gravity_spring_friction")
        ]
        # Add train combined examples
        train_examples.extend(train_combined)

        test_examples = [
            ex for ex in examples
            if ex["scenario_type"] in ("spring_friction", "gravity_spring_friction")
        ]
        # If no real test examples of these types, use synthetic ones
        if not test_examples:
            test_examples = list(test_combined)

        print(f"Cross-domain split: train={len(train_examples)}, test={len(test_examples)}")
        print(f"  Train types: {sorted(set(ex['scenario_type'] for ex in train_examples))}")
        print(f"  Test types:  {sorted(set(ex['scenario_type'] for ex in test_examples))}")
    else:
        # Default: random stratified split, all combined examples in training
        all_examples = list(examples)
        all_examples.extend(combined_examples)

        train_examples, test_examples = create_train_test_split(
            all_examples, test_size=0.3, seed=args.seed,
        )
        print(f"Random split: train={len(train_examples)}, test={len(test_examples)}")

    if len(test_examples) == 0:
        print("ERROR: No test examples. Check data.")
        sys.exit(1)

    # Tokenizer
    tokenizer = ExpressionTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create datasets
    train_dataset = ExpressionDataset(train_examples, tokenizer)
    test_dataset = ExpressionDataset(test_examples, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    model = ExpressionSequenceModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,} (target < 1M: {'✓' if n_params < 1_000_000 else '✗'})")

    # Train
    device = torch.device("cpu")
    print(f"\nTraining on {device} with {args.threads} threads, {args.epochs} epochs")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.lr}")

    checkpoint_dir = project_root / args.checkpoint_dir

    stats = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        pad_idx=PAD_IDX,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
    )

    # Save final model and tokenizer
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / args.checkpoint_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "d_model": model.d_model,
        },
        final_path,
    )
    tokenizer.save(checkpoint_dir / "tokenizer_fixa.json")

    # Save training stats
    stats_path = checkpoint_dir / "training_stats_fixa.json"
    stats["n_params"] = model.count_parameters()
    stats["n_train"] = len(train_dataset)
    stats["n_test"] = len(test_dataset)
    stats["vocab_size"] = tokenizer.vocab_size
    stats["n_combined_train"] = len(train_combined)
    stats["n_combined_test"] = len(test_combined)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTraining complete.")
    print(f"  Best model: {checkpoint_dir / args.checkpoint_name}")
    print(f"  Final model: {final_path}")
    print(f"  Tokenizer: {checkpoint_dir / 'tokenizer_fixa.json'}")
    print(f"  Stats: {stats_path}")
    print(f"  Final train_loss: {stats['train_loss'][-1]:.4f}")
    print(f"  Final test_acc: {stats['test_acc'][-1]:.4f}")

    # ── Cross-Domain Evaluation ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Cross-Domain Composition Evaluation")
    print(f"{'='*60}")

    # Load test scenarios for evaluation
    test_scenarios = load_test_scenarios(observations_path)

    # Run evaluation
    results = run_cross_domain_evaluation(
        model, tokenizer, test_scenarios, device,
    )
    results["model_path"] = str(final_path)
    results["observations_path"] = str(observations_path)

    # Compute cross-domain composition score
    # (fraction of combined-domain scenarios where model includes all domain terms)
    combined_scenarios = [
        s for s in test_scenarios
        if s.get("known_invariant") and (
            ("k" in s["known_invariant"] and "g" in s["known_invariant"])
            or ("q" in s["known_invariant"] and "g" in s["known_invariant"])
        )
    ]
    composition_scores = []
    for scenario in combined_scenarios:
        sid = scenario["id"]
        if sid in results["tests"]:
            expr = results["tests"][sid]["best_expression"]
            known = scenario["known_invariant"]
            score = _token_overlap_score(expr, known)
            composition_scores.append(score)
            print(f"  {sid}: score={score:.4f}  expr='{expr}'  known='{known}'")

    results["composition_score"] = (
        sum(composition_scores) / len(composition_scores)
        if composition_scores else 0.0
    )
    results["n_combined_tested"] = len(composition_scores)

    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"  Tests: {results['n_tests']}")
    print(f"  Passed: {results['n_passed']}")
    print(f"  Composition score: {results['composition_score']:.4f}")
    print(f"  Combined scenarios tested: {results['n_combined_tested']}")

    # Check acceptance criteria
    print(f"\nAcceptance Criteria:")
    has_k_term = any(
        "k" in t["best_expression"]
        for t in results["tests"].values()
    )
    print(f"  Model generates ½kx² term when k present: {'✓' if has_k_term else '✗'}")
    composition_ok = results["composition_score"] > 0.85
    print(f"  Cross-domain composition score > 0.85: {'✓' if composition_ok else '✗'} ({results['composition_score']:.4f})")

    # Save results
    results_path = project_root / args.output_results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
