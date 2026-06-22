#!/usr/bin/env python3
"""Quick test of composer fixes."""
import sys
sys.path.insert(0, '/home/blueman1818/Projects/theta-core')

from src.physics.composer import (
    extract_domain_examples, tokenize_expression, detokenize_expression,
    TEMPLATE_UNK_IDX, DOMAIN_QUANTITY_KEY, DOMAINS,
)

print('DOMAINS:', DOMAINS)
print('EM keys:', DOMAIN_QUANTITY_KEY['em'])
print('Thermal keys:', DOMAIN_QUANTITY_KEY['thermal'])

em_examples = extract_domain_examples('data/observations/em_synthetic.json', 'em')
print(f'EM examples: {len(em_examples)}')
if em_examples:
    for ex in em_examples[:2]:
        print(f'  qty={sorted(ex["quantities"].keys())}, expr={ex["expression"]}')

thermal_examples = extract_domain_examples('data/observations/thermal_synthetic.json', 'thermal')
print(f'Thermal examples: {len(thermal_examples)}')
if thermal_examples:
    for ex in thermal_examples[:2]:
        print(f'  qty={sorted(ex["quantities"].keys())}, expr={ex["expression"]}')

# Test tokenization
for inv in ['abs(epsilon)', 'delta_S/log(V)',
            '0.5*m*(vx^2 + vy^2) - q*E*x',
            '0.5*m1*v1^2 + 0.5*m2*v2^2 + k*q1*q2/(abs(x2-x1))']:
    t = tokenize_expression(inv)
    print(f'Tokenize: {inv} -> {detokenize_expression(t)} (unk={t.count(TEMPLATE_UNK_IDX)})')
