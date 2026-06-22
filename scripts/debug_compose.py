#!/usr/bin/env python3
"""Debug the composition test failure - replicate the training script's test cases."""
import sys
sys.path.insert(0, '/home/blueman1818/Projects/theta-core')

import torch
from src.physics.composer import (
    DomainClassifier, DomainTemplateGenerator, PerDomainComposer,
    ExpressionComposer, load_domain_generator, load_domain_classifier,
    quantities_to_tensor, quantities_to_features,
    DOMAIN_QUANTITIES, TEMPLATE_VOCAB_SIZE,
)

device = torch.device("cpu")

# Load trained models
classifier = load_domain_classifier("checkpoints/domain_classifier.pt")
em_gen = load_domain_generator("checkpoints/em_template.pt")
thermal_gen = load_domain_generator("checkpoints/thermal_template.pt")

# Fresh gravity/spring
gravity_gen = DomainTemplateGenerator(d_model=40, nhead=2)
spring_gen = DomainTemplateGenerator(d_model=40, nhead=2)

generators = {
    "gravity": gravity_gen,
    "spring": spring_gen,
    "em": em_gen,
    "thermal": thermal_gen,
}

composer = PerDomainComposer(classifier, generators)
composer.to(device)

test_cases = [
    # EM + gravity: charged particle falling
    {
        "name": "EM + gravity: charged particle falling",
        "quantities": ["m", "g", "h", "v", "q", "E"],
    },
    # Thermal + mechanical: gas against spring
    {
        "name": "Thermal + mechanical: gas against spring",
        "quantities": ["P", "V", "T", "m", "k", "h", "v"],
    },
    # EM only
    {
        "name": "EM only: E field",
        "quantities": ["m", "q", "E", "x", "vx", "vy"],
    },
]

for tc in test_cases:
    print(f"\n=== {tc['name']} ===")
    print(f"  Quantities: {tc['quantities']}")
    
    # Check domain classifier
    features = quantities_to_features(tc['quantities']).unsqueeze(0).to(device)
    probs = classifier.predict_proba(features).squeeze(0)
    for i, d in enumerate(["gravity", "spring", "em", "thermal"]):
        print(f"    {d}: {probs[i].item():.4f}")
    
    # Check filtered quantities
    for domain in ["gravity", "spring", "em", "thermal"]:
        filtered = composer._filter_domain_quantities(tc['quantities'], domain)
        if filtered:
            src = quantities_to_tensor(filtered, max_len=8).unsqueeze(0)
            gen = generators.get(domain)
            print(f"    {domain} filtered: {filtered}, max_id={src.max().item()}, gen_vocab={gen.vocab_size if gen else '?'}")
    
    try:
        composed, active = composer.forward(tc['quantities'])
        print(f"  Active: {active}")
        print(f"  Composed: {composed}")
    except Exception as e:
        print(f"  ERROR: {e}")
