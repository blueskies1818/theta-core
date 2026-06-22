# ERA GATE Report: Pre-1905 → Post-1905 Physics Discovery

> Ultimate honesty test. Train on pre-1905 physics, test on post-1905 experimental data. Discover what wasn't taught.

**Date:** 2026-06-21 21:31:52
**Duration:** 113.5s
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
- **special_relativity** (conserved_quantity): `E*E` (constancy=1.0000)
- **general_relativity** (conserved_quantity): `-1^delta_phi_gr` (constancy=1.0000)
- **quantum_hydrogen** (conserved_quantity): `h*n` (constancy=1.0000)
- **wave_particle_duality** (conserved_quantity): `V*m_e` (constancy=1.0000)
- **uncertainty_principle** (conserved_quantity): `hbar/n` (constancy=1.0000)

## Per-Scenario Results
### special_relativity
- Domain: relativistic (post-1905)
- Description: Muon lifetime measurements demonstrating time dilation.
- Expected invariant: `E^2 - (p*c)^2`
- Expected symmetry: `lorentz_invariant`

| Metric | Value |
|--------|-------|
| Best expression | `E*E` |
| Best constancy | 1.000000 |
| Noise floor | 0.850000 |
| Noise threshold | 0.894887 |
| Breakthrough | True |
| Type | conserved_quantity |
| p-value | 0.0008325212979045649 |

**Symmetry:**
- Known matched: []
- Discovered: ['time_translation']
- Group: ℝ (time translation)
- Score: 1.0000

| Expression | Constancy | ± Error | Floor | Gate | p |
|-----------|----------|---------|-------|------|---|
| `E*E` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0008325212979045649 |
| `E*tau0` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0008325212979045649 |
| `E*tau` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0008325212979045649 |
| `E^2` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0008325212979045649 |
| `E/c` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0008325212979045649 |
| `E/v` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0008325212979045649 |
| `tau/v` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0008325212979045649 |
| `tau0/v` | 1.0000 | 0.0000 | 0.8500 | ✓ | 0.0008325212979045649 |
| `v` | 0.9175 | 0.0234 | 0.8500 | ✓ | 0.1325864147598863 |
| `c*v` | 0.9175 | 0.0234 | 0.8500 | ✓ | 0.1325864147598863 |

### general_relativity
- Domain: relativistic (post-1905)
- Description: Mercury perihelion precession — 43 arcsec/century anomaly.
- Expected invariant: `delta_phi_obs*a*(1-e^2)`
- Expected symmetry: `schwarzschild`

| Metric | Value |
|--------|-------|
| Best expression | `-1^delta_phi_gr` |
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
| `-1^delta_phi_gr` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `delta_phi_gr/M_sun` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `delta_phi_gr^0` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `1^delta_phi_gr` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `0/delta_phi_gr` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `G/a` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `e*phi` | 0.7566 | 0.0001 | 0.5000 | ✗ | 0.010301336327776633 |
| `delta_phi_newton/delta_phi_obs` | 0.7147 | 0.0160 | 0.5000 | ✗ | 0.03182965539280236 |
| `c/phi` | 0.6868 | 0.0003 | 0.5000 | ✗ | 0.06183086671471383 |
| `M_sun/phi` | 0.6868 | 0.0003 | 0.5000 | ✗ | 0.06183086671471383 |

### quantum_hydrogen
- Domain: quantum (post-1905)
- Description: Hydrogen Balmer series — quantized energy levels.
- Expected invariant: `E*n^2`
- Expected symmetry: `so4_dynamical`

| Metric | Value |
|--------|-------|
| Best expression | `h*n` |
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
| `h*n` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `E/c` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `1^n` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `E*lambda` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `h/lambda` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `E/n` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `E*E` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `lambda*lambda` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `h/n` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `lambda^2` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |

### wave_particle_duality
- Domain: quantum (post-1905)
- Description: Double-slit electron interference — wave-particle duality.
- Expected invariant: `lambda*p`
- Expected symmetry: `debroglie`

| Metric | Value |
|--------|-------|
| Best expression | `V*m_e` |
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
| `V*m_e` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `L*p` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `d_y*p` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `V*e` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `h/p` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `d*lambda` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `h*lambda` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `d_y*lambda` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `lambda*e` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |
| `p^2` | 1.0000 | 0.0000 | 0.8500 | ✓ | - |

### uncertainty_principle
- Domain: quantum (post-1905)
- Description: Position-momentum uncertainty measurements.
- Expected invariant: `delta_x*delta_p`
- Expected symmetry: `heisenberg`

| Metric | Value |
|--------|-------|
| Best expression | `hbar/n` |
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
| `hbar/n` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `a*hbar` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `a*delta_p` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `hbar*n` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `delta_x*hbar` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `delta_p*delta_x` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `delta_p*delta_p` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `delta_p^2` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `delta_p*hbar` | 1.0000 | 0.0000 | 0.5000 | ✓ | 5.733031436250258e-07 |
| `n+1` | 0.9959 | 0.0000 | 0.5000 | ✓ | 7.087885194323462e-07 |

---
*Generated by ERA GATE pipeline*