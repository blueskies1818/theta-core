#!/usr/bin/env python3
"""Train the symmetry classifier on all observation databases.

Generates training data from Phase A-E scenarios, trains a small MLP
(~100K params) to classify which symmetries are present in a system
given only its quantity set.

Output: checkpoints/symmetry_classifier.pt
"""

import json
from pathlib import Path

import torch

from src.physics.symmetry import (
    SymmetryClassifier,
    build_symmetry_training_data,
    train_symmetry_classifier,
)

OBSERVATION_FILES = [
    "data/observations/phase1_falling.json",
    "data/observations/phase2_extended.json",
    "data/observations/mechanics_synthetic.json",
    "data/observations/em_synthetic.json",
    "data/observations/thermal_synthetic.json",
]


def main() -> None:
    print("Building training data from observation databases...")

    all_features: list[list[float]] = []
    all_labels: list[list[int]] = []

    for path_str in OBSERVATION_FILES:
        path = Path(path_str)
        if not path.exists():
            print(f"  SKIP {path_str} (not found)")
            continue
        print(f"  Loading {path_str}...")
        features, labels = build_symmetry_training_data(str(path))
        all_features.extend(features)
        all_labels.extend(labels)
        print(f"    → {len(features)} examples")

    # Deduplicate by feature vector
    seen: set[tuple] = set()
    dedup_features: list[list[float]] = []
    dedup_labels: list[list[int]] = []
    for feat, lab in zip(all_features, all_labels):
        key = tuple(feat)
        if key not in seen:
            seen.add(key)
            dedup_features.append(feat)
            dedup_labels.append(lab)

    print(f"\nTotal: {len(dedup_features)} unique examples "
          f"(from {len(all_features)} raw)")

    # Print class distribution
    from src.physics.symmetry import SYMMETRY_CLASS_LABELS
    label_sums = [sum(lab[i] for lab in dedup_labels) for i in range(len(dedup_labels[0]))]
    print("\nClass distribution:")
    for name, count in zip(SYMMETRY_CLASS_LABELS, label_sums):
        print(f"  {name}: {count}/{len(dedup_labels)} ({100*count/max(1,len(dedup_labels)):.1f}%)")

    # Train
    print("\nTraining symmetry classifier...")
    clf = train_symmetry_classifier(
        dedup_features,
        dedup_labels,
        epochs=100,
        learning_rate=0.002,
        checkpoint_path="checkpoints/symmetry_classifier.pt",
    )

    n_params = clf.count_parameters()
    print(f"\nModel: {n_params:,} parameters")
    print(f"Saved to: checkpoints/symmetry_classifier.pt")

    # Quick eval
    print("\nEvaluation on training data:")
    for i in range(min(5, len(dedup_features))):
        probs = clf.predict(dedup_features[i])
        true = dedup_labels[i]
        pred = [1 if p > 0.5 else 0 for p in probs]
        acc = sum(1 for a, b in zip(pred, true) if a == b) / len(true)
        print(f"  Example {i}: true={true}, pred={pred}, acc={acc:.3f}")


if __name__ == "__main__":
    main()
