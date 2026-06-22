# ERA GATE Report: Pre-1905 → Post-1905 Physics Discovery

> Ultimate honesty test. Train on pre-1905 physics, test on post-1905 experimental data. Discover what wasn't taught.

**Date:** 2026-06-21 21:18:59
**Duration:** 101.7s
**Noise level:** MEDIUM (3%)

## Pre-1905 Training Configuration

- **Scenarios:** 162
- **Domains:** {'gravity': 36, 'spring': 18, 'em': 30, 'thermal': 58, 'collision': 20}
- **Post-1905 leakage:** []

### Leakage Verification
- Clean: True
- Pre-1905 domains: ['gravity', 'spring', 'em', 'thermal', 'collision']
- Pre-1905 symmetries: ['galilean', 'u1']

## Summary

| Metric | Value |
|--------|-------|
| Scenarios tested | 5 |
| Breakthroughs | 5 |
| Conservation | 5 |
| Symmetry | 0 |
| Significant | 0 |

### ★ ERA GATE BREACHED ★
- **special_relativity** (conserved_quantity): `E*tau` (constancy=1.0000)
- **general_relativity** (conserved_quantity): `0/delta_phi_obs` (constancy=1.0000)
- **quantum_hydrogen** (conserved_quantity): `n*h` (constancy=1.0000)
- **wave_particle_duality** (conserved_quantity): `h/p` (constancy=1.0000)
- **uncertainty_principle** (conserved_quantity): `delta_p*hbar` (constancy=1.0000)

## Per-Scenario Results
### special_relativity
- Domain: relativistic (post-1905)
- Description: Muon lifetime measurements demonstrating time dilation.
- Expected invariant: `E^2 - (p*c)^2`
- Expected symmetry: `lorentz_invariant`

| Metric | Value |
|--------|-------|
| Best expression | `E*tau` |
| Best constancy | 1.000000 |
| Noise floor | 0.850000 |
| Noise threshold | 0.893804 |
| Breakthrough | True |
| Type | conserved_quantity |
| p-value | 0.0006163428257339731 |

**Symmetry:**
- Known matched: []
- Discovered: ['time_translation']
- Group: ℝ (time translation)
- Score: 1.0000

| Expression | Constancy | ± Error | Floor | Gate | p |
|-----------|----------|---------|-------|------|---|
| `E*tau` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0006163428257339731 |
| `E/v` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0006163428257339731 |
| `v^0` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0006163428257339731 |
| `0/v` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0006163428257339731 |
| `tau0/v` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0006163428257339731 |
| `E/c` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0006163428257339731 |
| `E*tau0` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0006163428257339731 |
| `E*E` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0006163428257339731 |
| `tau/v` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0006163428257339731 |
| `E^2` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0006163428257339731 |

### general_relativity
- Domain: relativistic (post-1905)
- Description: Mercury perihelion precession — 43 arcsec/century anomaly.
- Expected invariant: `delta_phi_obs*a*(1-e^2)`
- Expected symmetry: `schwarzschild`

| Metric | Value |
|--------|-------|
| Best expression | `0/delta_phi_obs` |
| Best constancy | 1.000000 |
| Noise floor | 0.500000 |
| Noise threshold | 0.800000 |
| Breakthrough | True |
| Type | conserved_quantity |
| p-value | 5.733031436250258e-07 |

**Symmetry:**
- Known matched: []
- Discovered: ['time_translation', 'space_translation_x', 'space_translation_y', 'space_translation_z', 'rotation_xy', 'rotation_xz', 'rotation_yz', 'boost_x', 'boost_y', 'boost_z', 'u1_phase']
- Group: lorentz_poincare_like
- Score: 0.0000

| Expression | Constancy | ± Error | Floor | Gate | p |
|-----------|----------|---------|-------|------|---|
| `0/delta_phi_obs` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `e^0` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `a^0` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `G/a` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `0/e` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `0^e` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `0/a` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `e*phi` | 0.7567 | 0.0002 | 0.5000 | ✗ | 0.010260233649237227 |
| `delta_phi_newton/delta_phi_obs` | 0.7127 | 0.0112 | 0.5000 | ✗ | 0.03345333250604998 |
| `c/phi` | 0.6864 | 0.0005 | 0.5000 | ✗ | 0.06237977149557539 |

### quantum_hydrogen
- Domain: quantum (post-1905)
- Description: Hydrogen Balmer series — quantized energy levels.
- Expected invariant: `E*n^2`
- Expected symmetry: `so4_dynamical`

| Metric | Value |
|--------|-------|
| Best expression | `n*h` |
| Best constancy | 1.000000 |
| Noise floor | 0.850000 |
| Noise threshold | 0.850000 |
| Breakthrough | True |
| Type | conserved_quantity |
| p-value | N/A |

**Symmetry:**
- Known matched: []
- Discovered: ['time_translation', 'space_translation_x', 'space_translation_y', 'space_translation_z', 'rotation_xy', 'rotation_xz', 'rotation_yz', 'boost_x', 'boost_y', 'boost_z', 'u1_phase']
- Group: lorentz_poincare_like
- Score: 0.0000

| Expression | Constancy | ± Error | Floor | Gate | p |
|-----------|----------|---------|-------|------|---|
| `n*h` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `lambda*h` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `E*h` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `h*n` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `E*lambda` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `E/n^2` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `lambda*lambda` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `E/c` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `lambda^2` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `h*E` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |

### wave_particle_duality
- Domain: quantum (post-1905)
- Description: Double-slit electron interference — wave-particle duality.
- Expected invariant: `lambda*p`
- Expected symmetry: `debroglie`

| Metric | Value |
|--------|-------|
| Best expression | `h/p` |
| Best constancy | 1.000000 |
| Noise floor | 0.850000 |
| Noise threshold | 0.850000 |
| Breakthrough | True |
| Type | conserved_quantity |
| p-value | N/A |

**Symmetry:**
- Known matched: []
- Discovered: ['time_translation', 'space_translation_x', 'space_translation_y', 'space_translation_z', 'rotation_xy', 'rotation_xz', 'rotation_yz', 'boost_x', 'boost_y', 'boost_z', 'u1_phase']
- Group: lorentz_poincare_like
- Score: 0.0000

| Expression | Constancy | ± Error | Floor | Gate | p |
|-----------|----------|---------|-------|------|---|
| `h/p` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `p^2` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `h*V` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `h/lambda` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `V*h` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `h*lambda` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `d*lambda` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `d_y*lambda` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `d_y*h` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `lambda*lambda` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |

### uncertainty_principle
- Domain: quantum (post-1905)
- Description: Position-momentum uncertainty measurements.
- Expected invariant: `delta_x*delta_p`
- Expected symmetry: `heisenberg`

| Metric | Value |
|--------|-------|
| Best expression | `delta_p*hbar` |
| Best constancy | 1.000000 |
| Noise floor | 0.500000 |
| Noise threshold | 0.800000 |
| Breakthrough | True |
| Type | conserved_quantity |
| p-value | 5.733031436250258e-07 |

**Symmetry:**
- Known matched: []
- Discovered: ['time_translation', 'space_translation_x', 'space_translation_y', 'space_translation_z', 'rotation_xy', 'rotation_xz', 'rotation_yz', 'boost_x', 'boost_y', 'boost_z', 'u1_phase']
- Group: lorentz_poincare_like
- Score: 0.0000

| Expression | Constancy | ± Error | Floor | Gate | p |
|-----------|----------|---------|-------|------|---|
| `delta_p*hbar` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `hbar*n` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `delta_p^2` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `delta_x*hbar` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `delta_p*delta_p` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `delta_x*delta_p` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `delta_p*delta_x` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `hbar/n` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `a*delta_p` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `hbar*delta_p` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |

---
*Generated by ERA GATE pipeline*