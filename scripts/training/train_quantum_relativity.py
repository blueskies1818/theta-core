#!/usr/bin/env python3
"""Train quantum + relativistic domain template generators.
Add to complete 7-domain PerDomainComposer.

Trains:
  1. Domain classifier (6 domains: gravity, spring, em, thermal, quantum, relativistic)
  2. Quantum TemplateGenerator (~50K params) on quantum_synthetic.json
  3. Relativistic TemplateGenerator (~50K params) on relativity_synthetic.json
  4. Composes full 7-domain pipeline with all existing + new templates

Output:
  checkpoints/quantum_template.pt
  checkpoints/relativity_template.pt
  checkpoints/domain_classifier.pt (6-domain)
  data/phase_f_7domain_results.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from src.physics.composer import (
    DomainClassifier,
    DomainTemplateGenerator,
    ExpressionComposer,
    PerDomainComposer,
    DOMAINS,
    COLLISION_DOMAIN,
    DOMAIN_TEMPLATES,
    DOMAIN_QUANTITIES,
    NUM_DOMAIN_CLASSES,
    assign_domain_labels,
    extract_domain_examples,
    _assign_domain_labels_from_keys,
    quantity_set_to_features,
    quantities_to_tensor,
    quantities_to_features,
    expression_to_tensor,
    detokenize_expression,
    tokenize_expression,
    TEMPLATE_VOCAB_SIZE,
    TEMPLATE_PAD_IDX,
    TEMPLATE_SOS_IDX,
    TEMPLATE_EOS_IDX,
    TEMPLATE_UNK_IDX,
    save_composer,
    save_domain_classifier,
    save_domain_generator,
    load_domain_generator,
    load_domain_classifier,
    load_composer,
)


# ── Domain Classifier Training (6-domain) ──────────────────────────────────


class DomainClassifierDataset(torch.utils.data.Dataset):
    """Dataset for 6-domain classifier: quantity features → domain labels."""

    def __init__(self, observations_paths: list[str | Path]):
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for path in observations_paths:
            with open(path) as f:
                data = json.load(f)
            for obs in data:
                qty_symbols = list(obs["quantities"].keys())
                all_symbols = set(qty_symbols) | set(obs.get("parameters", {}).keys())
                features = quantity_set_to_features(list(all_symbols))
                labels = torch.tensor(
                    _assign_domain_labels_from_keys(all_symbols), dtype=torch.float
                )
                self.samples.append((features, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def train_domain_classifier(
    observations_paths: list[str | Path],
    checkpoint_dir: str | Path,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
) -> tuple[DomainClassifier, dict]:
    """Train the 6-domain classifier on all observation data."""
    print("\n" + "=" * 60)
    print("Training Domain Classifier (6 domains)")
    print("=" * 60)

    dataset = DomainClassifierDataset(observations_paths)
    print(f"  Training examples: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DomainClassifier()
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    stats = {"epoch_loss": []}
    model.train()
    model.to(device)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model.forward(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        stats["epoch_loss"].append(avg_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} — loss: {avg_loss:.4f}")

    # Save
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "domain_classifier.pt"
    save_domain_classifier(model, str(ckpt_path))
    print(f"  Saved: {ckpt_path}")

    # Evaluate
    model.eval()
    print("\n  Domain classifier evaluation:")
    test_sets = [
        (["m", "g", "h", "v"], "gravity-only"),
        (["m", "k", "x", "v"], "spring-only"),
        (["m", "g", "h", "v", "q", "E"], "em+gravity"),
        (["P", "V", "T"], "thermal-only"),
        (["hbar", "m", "L", "n", "E"], "quantum"),
        (["c", "v", "t", "x", "gamma"], "relativistic"),
        (["m", "k", "g", "h", "v"], "gravity+spring"),
        (["m", "g", "hbar", "c"], "mixed quantum+relativistic"),
        (["m", "k", "g", "h", "v", "q", "E", "P", "V", "T", "hbar", "c"], "all six"),
    ]
    for qty_symbols, desc in test_sets:
        features = quantity_set_to_features(qty_symbols).unsqueeze(0).to(device)
        probs = model.predict_proba(features).squeeze(0)
        score_str = ", ".join(
            f"{DOMAINS[i]}={probs[i].item():.3f}" for i in range(NUM_DOMAIN_CLASSES)
        )
        print(f"    {desc:30s}: {score_str}")

    return model, stats


# ── Domain Template Generator Training ─────────────────────────────────────


class DomainTemplateDataset(torch.utils.data.Dataset):
    """Dataset for a single domain's template generator."""

    def __init__(
        self,
        examples: list[dict],
        src_max_len: int = 10,
        tgt_max_len: int = 40,
    ):
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for ex in examples:
            qty_symbols = sorted(ex["quantities"].keys())
            src = quantities_to_tensor(qty_symbols, max_len=src_max_len)
            tgt = expression_to_tensor(ex["expression"], max_len=tgt_max_len)
            self.samples.append((src, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def train_domain_generator(
    observations_path: str | Path,
    domain: str,
    checkpoint_dir: str | Path,
    epochs: int = 80,
    lr: float = 1e-3,
    batch_size: int = 4,
    device: torch.device = torch.device("cpu"),
    d_model: int = 40,
    nhead: int = 2,
) -> tuple[DomainTemplateGenerator, dict]:
    """Train a single domain's template generator."""
    print(f"\n{'=' * 60}")
    print(f"Training {domain.capitalize()} Template Generator")
    print("=" * 60)

    examples = extract_domain_examples(observations_path, domain)
    print(f"  Training examples: {len(examples)}")

    if len(examples) == 0:
        print(f"  WARNING: No training examples for domain '{domain}'")
        model = DomainTemplateGenerator(d_model=d_model, nhead=nhead)
        return model, {"error": "no_examples"}

    from collections import Counter
    expr_counts = Counter(ex["expression"] for ex in examples)
    print(f"  Unique expressions: {len(expr_counts)}")
    for expr, count in expr_counts.most_common(10):
        print(f"    [{count:3d}] {expr}")

    dataset = DomainTemplateDataset(examples, src_max_len=10, tgt_max_len=40)
    dataloader = DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=True
    )

    model = DomainTemplateGenerator(
        d_model=d_model, nhead=nhead, max_src_len=10, max_tgt_len=40
    )
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=TEMPLATE_PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    stats = {"train_loss": [], "train_acc": []}
    model.train()
    model.to(device)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0

        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = (src == TEMPLATE_PAD_IDX)
            tgt_mask_pad = (tgt_input == TEMPLATE_PAD_IDX)
            tgt_causal = torch.triu(
                torch.ones(tgt_input.size(1), tgt_input.size(1), device=device)
                * float("-inf"),
                diagonal=1,
            )

            logits = model.forward(
                src, tgt_input,
                src_padding_mask=src_mask,
                tgt_padding_mask=tgt_mask_pad,
                tgt_mask=tgt_causal,
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            preds = logits.argmax(dim=-1)
            mask = tgt_output != TEMPLATE_PAD_IDX
            correct = (preds[mask] == tgt_output[mask]).sum().item()
            total = mask.sum().item()
            epoch_correct += correct
            epoch_total += total
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_correct / max(epoch_total, 1) if epoch_total > 0 else 0.0
        stats["train_loss"].append(avg_loss)
        stats["train_acc"].append(avg_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} — "
                f"loss: {avg_loss:.4f}, acc: {avg_acc:.4f}"
            )

    # Save
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{domain}_template.pt"
    save_domain_generator(model, str(ckpt_path))
    print(f"  Saved: {ckpt_path}")

    # Evaluate
    model.eval()
    print(f"\n  {domain.capitalize()} generation samples:")
    for ex in examples[:5]:
        sample_qty = sorted(ex["quantities"].keys())
        src = quantities_to_tensor(sample_qty, max_len=10).unsqueeze(0).to(device)
        src_mask = (src == TEMPLATE_PAD_IDX)
        with torch.no_grad():
            seqs = model.generate(src, src_padding_mask=src_mask, max_len=40)
            generated = detokenize_expression(seqs[0])
        expected = ex["expression"]
        match = "✓" if generated == expected else ""
        print(f"    Input: {sample_qty}")
        print(f"    Generated: {generated}")
        print(f"    Expected:  {expected} {match}")
        print()

    return model, stats


# ── 7-Domain Composition Tests ────────────────────────────────────────────


def test_7domain_composition(
    composer: PerDomainComposer,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Test 7-domain composition across all domain combinations."""
    composer.to(device)
    composer.eval()

    test_cases = [
        # Quantum only
        {
            "name": "Quantum: particle-in-box",
            "quantities": ["hbar", "m", "L", "n", "E"],
            "expected_terms": ["E", "n^2"],
        },
        # Relativistic only
        {
            "name": "Relativistic: spacetime interval",
            "quantities": ["c", "t", "x", "v", "tau", "gamma"],
            "expected_terms": ["c*t", "x"],
        },
        # Quantum + gravity (particle in gravitational well)
        {
            "name": "Quantum + gravity: particle in well",
            "quantities": ["m", "g", "h", "hbar", "n", "E", "L", "v"],
            "expected_terms": ["m*g*h", "0.5*m*v^2", "E"],
        },
        # Relativistic + EM (charged particle near c)
        {
            "name": "Relativistic + EM: charged near c",
            "quantities": ["c", "v", "m", "E", "p", "q", "B"],
            "expected_terms": ["E^2", "p*c", "0.5*m"],
        },
        # All 6 domains (excluding collision)
        {
            "name": "All 6 domains: classical+quantum+relativistic",
            "quantities": ["m", "g", "h", "v", "k", "q", "E", "P", "V", "T", "hbar", "c", "p"],
            "expected_terms": ["m*g*h", "0.5*m*v^2", "P*V"],
        },
        # Existing baseline tests
        {
            "name": "Gravity only (baseline)",
            "quantities": ["m", "g", "h", "v"],
            "expected_terms": ["m*g*h", "0.5*m*v^2"],
        },
        {
            "name": "Spring only (baseline)",
            "quantities": ["m", "k", "h", "v"],
            "expected_terms": ["0.5*k*h^2", "0.5*m*v^2"],
        },
        {
            "name": "EM only (baseline)",
            "quantities": ["m", "q", "E", "x", "vx", "vy"],
            "expected_terms": ["0.5*m", "q*E"],
        },
    ]

    results = {"test_cases": [], "summary": {}}

    print("\n" + "=" * 60)
    print("7-Domain Composition Tests")
    print("=" * 60)

    for tc in test_cases:
        composed, active = composer.forward(tc["quantities"])
        print(f"\n  {tc['name']}:")
        print(f"    Quantities: {tc['quantities']}")
        print(f"    Active domains: {active}")
        print(f"    Composed: {composed}")

        term_hits = 0
        term_misses = 0
        for term in tc["expected_terms"]:
            found = term.replace("*", "") in composed.replace(" ", "").replace("*", "")
            if found:
                term_hits += 1
            else:
                term_misses += 1

        results["test_cases"].append({
            "name": tc["name"],
            "quantities": tc["quantities"],
            "active_domains": active,
            "composed_expression": composed,
            "expected_terms": tc["expected_terms"],
            "term_hits": term_hits,
            "term_misses": term_misses,
            "total_terms": len(tc["expected_terms"]),
        })
        print(f"    Term match: {term_hits}/{len(tc['expected_terms'])}")

    total_hits = sum(tc["term_hits"] for tc in results["test_cases"])
    total_terms = sum(tc["total_terms"] for tc in results["test_cases"])
    results["summary"] = {
        "total_test_cases": len(test_cases),
        "total_term_hits": total_hits,
        "total_term_misses": total_terms - total_hits,
        "total_terms": total_terms,
        "accuracy": total_hits / total_terms if total_terms > 0 else 0.0,
    }
    print(f"\n  Summary: {total_hits}/{total_terms} term matches "
          f"({results['summary']['accuracy']:.1%})")

    return results


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train quantum + relativistic template generators for 7-domain composer"
    )
    parser.add_argument(
        "--quantum-observations",
        default="data/observations/quantum_synthetic.json",
        help="Path to quantum observation database",
    )
    parser.add_argument(
        "--relativity-observations",
        default="data/observations/relativity_synthetic.json",
        help="Path to relativity observation database",
    )
    parser.add_argument(
        "--mechanics-observations",
        default="data/observations/mechanics_synthetic.json",
        help="Path to mechanics observation database",
    )
    parser.add_argument(
        "--em-observations",
        default="data/observations/em_synthetic.json",
        help="Path to EM observation database",
    )
    parser.add_argument(
        "--thermal-observations",
        default="data/observations/thermal_synthetic.json",
        help="Path to thermal observation database",
    )
    parser.add_argument(
        "--epochs-classifier", type=int, default=30,
        help="Epochs for 6-domain classifier",
    )
    parser.add_argument(
        "--epochs-quantum", type=int, default=80,
        help="Epochs for quantum template generator",
    )
    parser.add_argument(
        "--epochs-relativity", type=int, default=80,
        help="Epochs for relativity template generator",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--d-model", type=int, default=40,
        help="Template generator hidden dim",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints", help="Output directory"
    )
    parser.add_argument(
        "--results-path",
        default="data/phase_f_7domain_results.json",
        help="Path for results JSON",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--skip-classifier", action="store_true",
        help="Skip classifier training (use existing checkpoint)",
    )
    parser.add_argument(
        "--skip-quantum", action="store_true",
        help="Skip quantum generator training",
    )
    parser.add_argument(
        "--skip-relativity", action="store_true",
        help="Skip relativity generator training",
    )
    parser.add_argument(
        "--skip-composition-tests", action="store_true",
        help="Skip 7-domain composition tests",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    project_root = _project_root
    quantum_path = project_root / args.quantum_observations
    relativity_path = project_root / args.relativity_observations
    mechanics_path = project_root / args.mechanics_observations
    em_path = project_root / args.em_observations
    thermal_path = project_root / args.thermal_observations
    checkpoint_dir = project_root / args.checkpoint_dir

    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # ── Step 1: Train 6-domain classifier ──────────────────────────────────
    if not args.skip_classifier:
        obs_paths = [
            str(mechanics_path), str(em_path), str(thermal_path),
            str(quantum_path), str(relativity_path),
        ]
        classifier, clf_stats = train_domain_classifier(
            observations_paths=obs_paths,
            checkpoint_dir=str(checkpoint_dir),
            epochs=args.epochs_classifier,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        print("\nSkipping classifier training (using existing checkpoint)")
        clf_path = checkpoint_dir / "domain_classifier.pt"
        if clf_path.exists():
            classifier = load_domain_classifier(str(clf_path))
            clf_stats = {"skipped": True}
        else:
            print("  WARNING: No existing classifier checkpoint, creating new")
            classifier = DomainClassifier()
            clf_stats = {}

    # ── Step 2: Train quantum template generator ───────────────────────────
    if not args.skip_quantum:
        quantum_gen, quantum_stats = train_domain_generator(
            observations_path=str(quantum_path),
            domain="quantum",
            checkpoint_dir=str(checkpoint_dir),
            epochs=args.epochs_quantum,
            lr=args.lr,
            batch_size=min(args.batch_size, 4),
            device=device,
            d_model=args.d_model,
            nhead=2,
        )
    else:
        print("\nSkipping quantum generator training")
        q_path_ckpt = checkpoint_dir / "quantum_template.pt"
        if q_path_ckpt.exists():
            quantum_gen = load_domain_generator(str(q_path_ckpt))
            quantum_stats = {"skipped": True}
        else:
            quantum_gen = DomainTemplateGenerator(d_model=args.d_model, nhead=2)
            quantum_stats = {}

    # ── Step 3: Train relativity template generator ────────────────────────
    if not args.skip_relativity:
        relativity_gen, relativity_stats = train_domain_generator(
            observations_path=str(relativity_path),
            domain="relativistic",
            checkpoint_dir=str(checkpoint_dir),
            epochs=args.epochs_relativity,
            lr=args.lr,
            batch_size=min(args.batch_size, 4),
            device=device,
            d_model=args.d_model,
            nhead=2,
        )
    else:
        print("\nSkipping relativity generator training")
        r_path_ckpt = checkpoint_dir / "relativistic_template.pt"
        if r_path_ckpt.exists():
            relativity_gen = load_domain_generator(str(r_path_ckpt))
            relativity_stats = {"skipped": True}
        else:
            relativity_gen = DomainTemplateGenerator(d_model=args.d_model, nhead=2)
            relativity_stats = {}

    # ── Step 4: Build full 7-domain composer ───────────────────────────────
    generators: dict[str, DomainTemplateGenerator] = {
        "quantum": quantum_gen,
        "relativistic": relativity_gen,
    }

    # Load existing domain generators
    for domain in ["gravity", "spring", "em", "thermal"]:
        ckpt_path = checkpoint_dir / f"{domain}_template.pt"
        if ckpt_path.exists():
            try:
                generators[domain] = load_domain_generator(str(ckpt_path))
            except RuntimeError as e:
                print(f"  Note: Could not load {domain} checkpoint (vocab mismatch). "
                      f"Using fresh model. Error: {e}")
                generators[domain] = DomainTemplateGenerator(
                    d_model=args.d_model, nhead=2
                )
        else:
            generators[domain] = DomainTemplateGenerator(
                d_model=args.d_model, nhead=2
            )

    # Try to load collision generator
    collision_path = checkpoint_dir / "collision_template.pt"
    if collision_path.exists():
        try:
            generators[COLLISION_DOMAIN] = load_domain_generator(str(collision_path))
        except RuntimeError as e:
            print(f"  Note: Could not load collision checkpoint. {e}")

    composer = PerDomainComposer(classifier, generators)
    composer.to(device)
    print(f"\nFull composer parameters: {composer.count_parameters():,}")

    # ── Step 5: Run 7-domain composition tests ─────────────────────────────
    if not args.skip_composition_tests:
        comp_results = test_7domain_composition(composer, device)
    else:
        comp_results = {"skipped": True}

    # ── Step 6: Save results ───────────────────────────────────────────────
    all_results = {
        "classifier_stats": clf_stats,
        "quantum_stats": quantum_stats,
        "relativity_stats": relativity_stats,
        "composition_results": comp_results,
    }

    results_path = project_root / args.results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  checkpoints/quantum_template.pt")
    print(f"  checkpoints/relativity_template.pt")
    print(f"  checkpoints/domain_classifier.pt (6-domain)")
    print(f"  {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
