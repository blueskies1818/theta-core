"""Temporary verification script for hydrogen_balmer.json known_invariant."""
import json, math, sys
from pathlib import Path

data = json.loads(Path('data/real_experimental/hydrogen_balmer.json').read_text())
points = data['data_points']

def constancy(values):
    n = len(values)
    mean_val = sum(values) / n
    variance = sum((v - mean_val)**2 for v in values) / n
    std_val = math.sqrt(max(variance, 0.0))
    return 1.0 / (1.0 + std_val / abs(mean_val))

e_lambda_vals = [p['E'] * p['lambda'] for p in points]
e_n2_vals = [p['E'] * (p['n'] ** 2) for p in points]

el_score = constancy(e_lambda_vals)
en_score = constancy(e_n2_vals)

print(f'known_invariant field: {data["known_invariant"]}')
print(f'E*lambda values: {[f"{v:.6e}" for v in e_lambda_vals]}')
print(f'E*lambda constancy: {el_score:.4f}')
print(f'E*n^2 values:   {[f"{v:.6e}" for v in e_n2_vals]}')
print(f'E*n^2 constancy: {en_score:.4f}')
print(f'E*lambda is constant (~0.995): {el_score > 0.99} (actual: {el_score:.4f})')
print(f'E*n^2 varies (~0.57):         {en_score < 0.6}  (actual: {en_score:.4f})')

if el_score < 0.99 or en_score > 0.6:
    print('WARNING: Scores not as expected!', file=sys.stderr)
    sys.exit(1)
else:
    print('OK: Invariant scores confirmed.')
