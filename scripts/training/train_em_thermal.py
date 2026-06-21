#!/usr/bin/env python3
"""Train EM and Thermal domain template generators.

Trains:
  1. EM TemplateGenerator (~50K params) on em_synthetic.json (50 scenarios)
  2. Thermal TemplateGenerator (~50K params) on thermal_synthetic.json (58 scenarios)
  3. Optionally retrains domain classifier on combined data

Output:
  checkpoints/em_template.pt
  checkpoints/thermal_template.pt
  data/phase_f_5domain_results.json
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
    DOMAIN_TEMPLATES,
    DOMAIN_QUANTITIES,
    COLLISION_DOMAIN,
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


# ── Domain Classifier Training ───────────────────────────────────────────────

class DomainClassifierDataset(torch.utils.data.Dataset):
    """Dataset for domain classifier: quantity features → domain labels."""

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
    """Train the domain classifier on all observation data.

    Returns (trained_model, training_stats).
    """
    print("\n" + "=" * 60)
    print("Training Domain Classifier (4 domains)")
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
        (["m", "k", "g", "h", "v"], "gravity+spring"),
        (["m", "k", "g", "h", "v", "q", "E", "P", "V", "T"], "all four"),
    ]
    for qty_symbols, desc in test_sets:
        features = quantity_set_to_features(qty_symbols).unsqueeze(0).to(device)
        probs = model.predict_proba(features).squeeze(0)
        score_str = ", ".join(
            f"{DOMAINS[i]}={probs[i].item():.3f}" for i in range(len(DOMAINS))
        )
        print(f"    {desc:20s}: {score_str}")

    return model, stats


# ── Domain Template Generator Training ───────────────────────────────────────

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
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 4,
    device: torch.device = torch.device("cpu"),
    d_model: int = 40,
    nhead: int = 2,
) -> tuple[DomainTemplateGenerator, dict]:
    """Train a single domain's template generator.

    Only uses data from the specified domain.
    """
    print(f"\n{'=' * 60}")
    print(f"Training {domain.capitalize()} Template Generator")
    print("=" * 60)

    examples = extract_domain_examples(observations_path, domain)
    print(f"  Training examples: {len(examples)}")

    if len(examples) == 0:
        print(f"  WARNING: No training examples for domain '{domain}'")
        model = DomainTemplateGenerator(d_model=d_model, nhead=nhead)
        return model, {"error": "no_examples"}

    # Show example diversity
    from collections import Counter
    expr_counts = Counter(ex["expression"] for ex in examples)
    print(f"  Unique expressions: {len(expr_counts)}")
    for expr, count in expr_counts.most_common():
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

            # Teacher forcing: input tgt[:, :-1], predict tgt[:, 1:]
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

            # Accuracy
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

    # Evaluate: generate template and compare
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


# ── 5-Domain Composition Tests ──────────────────────────────────────────────

def test_5domain_composition(
    composer: PerDomainComposer,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Test 5-domain composition across all domain pairs.

    Returns results dict with test outcomes.
    """
    composer.to(device)
    composer.eval()

    test_cases = [
        # EM + gravity: charged particle falling
        {
            "name": "EM + gravity: charged particle falling",
            "quantities": ["m", "g", "h", "v", "q", "E"],
            "expected_terms": ["0.5*m*v^2", "m*g*h", "q*E"],
        },
        # Thermal + mechanical: gas expanding against spring
        {
            "name": "Thermal + mechanical: gas against spring",
            "quantities": ["P", "V", "T", "m", "k", "h", "v"],
            "expected_terms": ["P*V", "0.5*k*h^2", "0.5*m*v^2"],
        },
        # EM only
        {
            "name": "EM only: E field",
            "quantities": ["m", "q", "E", "x", "vx", "vy"],
            "expected_terms": ["0.5*m", "q*E"],
        },
        # Thermal only
        {
            "name": "Thermal only: ideal gas",
            "quantities": ["P", "V", "T", "n", "R"],
            "expected_terms": ["P*V", "T"],
        },
        # Gravity only (baseline)
        {
            "name": "Gravity only",
            "quantities": ["m", "g", "h", "v"],
            "expected_terms": ["m*g*h", "0.5*m*v^2"],
        },
        # Spring only (baseline)
        {
            "name": "Spring only",
            "quantities": ["m", "k", "h", "v"],
            "expected_terms": ["0.5*k*h^2", "0.5*m*v^2"],
        },
        # All 5 domains
        {
            "name": "All 5 domains: gravity+spring+em+thermal",
            "quantities": ["m", "g", "h", "v", "k", "q", "E", "P", "V", "T"],
            "expected_terms": ["m*g*h", "0.5*m*v^2", "0.5*k*h^2", "q*E", "P*V"],
        },
        # EM + thermal + gravity
        {
            "name": "EM + thermal + gravity",
            "quantities": ["m", "g", "h", "v", "q", "E", "P", "V", "T"],
            "expected_terms": ["0.5*m*v^2", "m*g*h", "q*E", "P*V"],
        },
    ]

    results = {"test_cases": [], "summary": {}}

    print("\n" + "=" * 60)
    print("5-Domain Composition Tests")
    print("=" * 60)

    for tc in test_cases:
        composed, active = composer.forward(tc["quantities"])
        print(f"\n  {tc['name']}:")
        print(f"    Quantities: {tc['quantities']}")
        print(f"    Active domains: {active}")
        print(f"    Composed: {composed}")

        # Check expected terms appear in composed expression
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train EM + Thermal template generators for 5-domain composer"
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
        "--phase2-observations",
        default="data/observations/phase2_extended.json",
        help="Path to phase2 observation database (for classifier)",
    )
    parser.add_argument(
        "--epochs-classifier", type=int, default=20,
        help="Epochs for domain classifier",
    )
    parser.add_argument(
        "--epochs-em", type=int, default=50,
        help="Epochs for EM template generator",
    )
    parser.add_argument(
        "--epochs-thermal", type=int, default=50,
        help="Epochs for thermal template generator",
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
        default="data/phase_f_5domain_results.json",
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
        "--skip-em", action="store_true",
        help="Skip EM generator training",
    )
    parser.add_argument(
        "--skip-thermal", action="store_true",
        help="Skip thermal generator training",
    )
    parser.add_argument(
        "--skip-composition-tests", action="store_true",
        help="Skip 5-domain composition tests",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    project_root = _project_root
    em_path = project_root / args.em_observations
    thermal_path = project_root / args.thermal_observations
    phase2_path = project_root / args.phase2_observations
    checkpoint_dir = project_root / args.checkpoint_dir

    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # ── Step 1: Train domain classifier ──────────────────────────────────
    if not args.skip_classifier:
        obs_paths = [str(phase2_path), str(em_path), str(thermal_path)]
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

    # ── Step 2: Train EM template generator ──────────────────────────────
    if not args.skip_em:
        em_gen, em_stats = train_domain_generator(
            observations_path=str(em_path),
            domain="em",
            checkpoint_dir=str(checkpoint_dir),
            epochs=args.epochs_em,
            lr=args.lr,
            batch_size=min(args.batch_size, 4),
            device=device,
            d_model=args.d_model,
            nhead=2,
        )
    else:
        print("\nSkipping EM generator training")
        em_path_ckpt = checkpoint_dir / "em_template.pt"
        if em_path_ckpt.exists():
            em_gen = load_domain_generator(str(em_path_ckpt))
            em_stats = {"skipped": True}
        else:
            em_gen = DomainTemplateGenerator(d_model=args.d_model, nhead=2)
            em_stats = {}

    # ── Step 3: Train thermal template generator ─────────────────────────
    if not args.skip_thermal:
        thermal_gen, thermal_stats = train_domain_generator(
            observations_path=str(thermal_path),
            domain="thermal",
            checkpoint_dir=str(checkpoint_dir),
            epochs=args.epochs_thermal,
            lr=args.lr,
            batch_size=min(args.batch_size, 4),
            device=device,
            d_model=args.d_model,
            nhead=2,
        )
    else:
        print("\nSkipping thermal generator training")
        thermal_path_ckpt = checkpoint_dir / "thermal_template.pt"
        if thermal_path_ckpt.exists():
            thermal_gen = load_domain_generator(str(thermal_path_ckpt))
            thermal_stats = {"skipped": True}
        else:
            thermal_gen = DomainTemplateGenerator(d_model=args.d_model, nhead=2)
            thermal_stats = {}

    # ── Step 4: Build full composer ──────────────────────────────────────
    # Create generators dict
    # Note: old checkpoints have different vocab sizes and won't load.
    # Gravity/spring use hardcoded templates as fallback for composition tests.
    generators: dict[str, DomainTemplateGenerator] = {
        "em": em_gen,
        "thermal": thermal_gen,
    }

    # Try to load existing gravity/spring checkpoints
    # If they fail due to vocab mismatch, create fresh ones (fallback templates used)
    for domain in ["gravity", "spring"]:
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

    composer = PerDomainComposer(classifier, generators)
    composer.to(device)
    print(f"\nFull composer parameters: {composer.count_parameters():,}")

    # ── Step 5: Run 5-domain composition tests ──────────────────────────
    if not args.skip_composition_tests:
        comp_results = test_5domain_composition(composer, device)
    else:
        comp_results = {"skipped": True}

    # ── Step 6: Save results ────────────────────────────────────────────
    all_results = {
        "classifier_stats": clf_stats,
        "em_stats": em_stats,
        "thermal_stats": thermal_stats,
        "composition_results": comp_results,
    }

    results_path = project_root / args.results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  checkpoints/em_template.pt")
    print(f"  checkpoints/thermal_template.pt")
    print(f"  {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
