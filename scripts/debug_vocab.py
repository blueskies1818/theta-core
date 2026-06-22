#!/usr/bin/env python3
"""Debug token ID ranges."""
import sys
sys.path.insert(0, '/home/blueman1818/Projects/theta-core')

from src.physics.composer import (
    extract_domain_examples, quantities_to_tensor, expression_to_tensor,
    TEMPLATE_VOCAB_SIZE, TEMPLATE_TOKEN_TO_ID, TEMPLATE_UNK_IDX
)

print(f"TEMPLATE_VOCAB_SIZE: {TEMPLATE_VOCAB_SIZE}")
print(f"Sample tokens: {list(TEMPLATE_TOKEN_TO_ID.items())[:10]}")

# Check EM examples
em_ex = extract_domain_examples('data/observations/em_synthetic.json', 'em')
print(f"\nEM examples: {len(em_ex)}")
for ex in em_ex[:2]:
    qty = sorted(ex['quantities'].keys())
    src = quantities_to_tensor(qty, max_len=10)
    tgt = expression_to_tensor(ex['expression'], max_len=40)
    print(f"  Qty: {qty} -> src max: {src.max().item()}")
    print(f"  Expr: {ex['expression'][:60]} -> tgt max: {tgt.max().item()}")

# Check thermal examples
thermal_ex = extract_domain_examples('data/observations/thermal_synthetic.json', 'thermal')
print(f"\nThermal examples: {len(thermal_ex)}")
for ex in thermal_ex[:2]:
    qty = sorted(ex['quantities'].keys())
    src = quantities_to_tensor(qty, max_len=10)
    tgt = expression_to_tensor(ex['expression'], max_len=40)
    print(f"  Qty: {qty} -> src max: {src.max().item()}")
    print(f"  Expr: {ex['expression'][:60]} -> tgt max: {tgt.max().item()}")

# Check if any have max >= vocab_size
all_max_ids = set()
for ex in em_ex + thermal_ex:
    qty = sorted(ex['quantities'].keys())
    src = quantities_to_tensor(qty, max_len=10)
    tgt = expression_to_tensor(ex['expression'], max_len=40)
    all_max_ids.add(src.max().item())
    all_max_ids.add(tgt.max().item())

print(f"\nAll max token IDs seen: {sorted(all_max_ids)}")
print(f"Vocab size: {TEMPLATE_VOCAB_SIZE}")
print(f"Any out of range: {any(i >= TEMPLATE_VOCAB_SIZE for i in all_max_ids)}")
