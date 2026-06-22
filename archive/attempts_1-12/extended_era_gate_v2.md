# Extended ERA GATE — Post-1905 Hidden Variable Discoveries

*Generated: 2026-06-22 04:52:03*

---

## Summary

| Category | Tested | Verified | Rate |
|----------|--------|----------|------|
| Quantum | 4 | 4 | 100% |
| Relativistic | 4 | 2 | 50% |
| Classical (FP check) | 2 | 2 | 100% |
| **Total** | **10** | **6** | **75%** |

Discovery threshold: 0.9

### Acceptance Criteria

- New scenarios with verified discoveries: 6/8 (PASS — need >= 5)
- False positives on classical: 0 (PASS — need 0)

---

## Quantum Scenarios

### angular_momentum — ✅ VERIFIED

- **Name**: Angular momentum E = E0*n^2
- **Domain**: quantum
- **Known invariant**: `E / n^2`
- **Hidden variable**: `n`
- **Error shape**: linear (confidence=1.0000, CV=0.0000)
- **Baseline** (without hidden var): `0.5*E` score=0.5598 not discovered
- **Augmented** (with hidden var): `E/n/n` score=1.0000 discovered
- **Proposal**: `half_integer` → `identity` (conf=0.9983)
- **Noise gate**: floor=0.6811, threshold=0.6811, passes=YES
- **All proposals (3):**
  1. `half_integer` → `identity` (conf=0.9983): Error shape suggests half_integer with identity in quantum domain (quantities: E)
  2. `integer_n` → `identity` (conf=0.6500): Linear residuals suggest counting/index pattern
  3. `integer_n` → `squared` (conf=0.4000): Generic hidden integer variable (fallback)
- **Time**: 2.43s

### spin_measurement — ✅ VERIFIED

- **Name**: Spin measurement E = E0*n
- **Domain**: quantum
- **Known invariant**: `E / n`
- **Hidden variable**: `n`
- **Error shape**: linear (confidence=1.0000, CV=0.0000)
- **Baseline** (without hidden var): `0.5*E+E-E` score=0.6782 not discovered
- **Augmented** (with hidden var): `E/n` score=0.9733 discovered
- **Proposal**: `half_integer` → `identity` (conf=0.9983)
- **Noise gate**: floor=0.6733, threshold=0.6733, passes=YES
- **All proposals (3):**
  1. `half_integer` → `identity` (conf=0.9983): Error shape suggests half_integer with identity in quantum domain (quantities: E)
  2. `integer_n` → `identity` (conf=0.6500): Linear residuals suggest counting/index pattern
  3. `integer_n` → `squared` (conf=0.4000): Generic hidden integer variable (fallback)
- **Time**: 2.10s

### blackbody_peak — ✅ VERIFIED

- **Name**: Blackbody E_photon/T = const (Wien)
- **Domain**: quantum
- **Known invariant**: `E_photon / T`
- **Hidden variable**: `none`
- **Error shape**:  (confidence=0.0000, CV=0.0000)
- **Baseline** (without hidden var): `E_photon/T` score=1.0000 discovered
- **Augmented** (with hidden var): `E_photon/T` score=1.0000 discovered
- **Proposal**: `` → `` (conf=0.0000)
- **Noise gate**: floor=0.6044, threshold=0.6044, passes=YES
- **Time**: 0.43s

### photoelectric — ✅ VERIFIED

- **Name**: Photoelectric K_max = h*f - phi
- **Domain**: quantum
- **Known invariant**: `K_max + phi`
- **Hidden variable**: `phi`
- **Error shape**:  (confidence=0.0000, CV=0.0000)
- **Baseline** (without hidden var): `f/f` score=1.0000 discovered
- **Augmented** (with hidden var): `f/f` score=1.0000 discovered
- **Proposal**: `` → `` (conf=0.0000)
- **Noise gate**: floor=0.5941, threshold=0.7800, passes=YES
- **Time**: 1.38s

---

## Relativistic Scenarios

### velocity_addition — ✅ VERIFIED

- **Name**: Velocity addition (u'+v)/(1+u'v/c²)
- **Domain**: relativistic
- **Known invariant**: `(u+v) / (1+u*v/c^2)`
- **Hidden variable**: `c`
- **Error shape**:  (confidence=0.0000, CV=0.0000)
- **Baseline** (without hidden var): `u+u` score=1.0000 discovered
- **Augmented** (with hidden var): `u+u` score=1.0000 discovered
- **Proposal**: `` → `` (conf=0.0000)
- **Noise gate**: floor=0.8500, threshold=0.9500, passes=YES
- **Time**: 0.47s

### relativistic_momentum — ✅ VERIFIED

- **Name**: Relativistic momentum p = gamma*m*v
- **Domain**: relativistic
- **Known invariant**: `E^2 - (p*c)^2`
- **Hidden variable**: `gamma`
- **Error shape**: linear (confidence=1.0000, CV=0.0000)
- **Baseline** (without hidden var): `0.5*E/p*p` score=0.5762 not discovered
- **Augmented** (with hidden var): `E/gamma` score=1.0000 discovered
- **Proposal**: `continuous_ratio` → `ratio` (conf=1.0000)
- **Noise gate**: floor=0.6495, threshold=0.6495, passes=YES
- **All proposals (4):**
  1. `continuous_ratio` → `ratio` (conf=1.0000): Error shape suggests continuous_ratio with ratio in relativistic domain (quantities: m, v, E, c, p)
  2. `continuous_ratio` → `identity` (conf=0.6500): Relativistic linear residuals — propose continuous multiplier gamma
  3. `integer_n` → `identity` (conf=0.6500): Linear residuals suggest counting/index pattern
  4. `integer_n` → `squared` (conf=0.4000): Generic hidden integer variable (fallback)
- **Time**: 4.37s

### time_dilation — ⚠️ NO DISCOVERY

- **Name**: Time dilation delta_t = gamma*delta_tau
- **Domain**: relativistic
- **Known invariant**: `(c*t)^2 - x^2`
- **Hidden variable**: `gamma`
- **Error shape**: linear (confidence=1.0000, CV=0.0000)
- **Baseline** (without hidden var): `` score=0.0000 not discovered
- **Augmented** (with hidden var): `` score=0.0000 not discovered
- **Proposal**: `continuous_ratio` → `ratio` (conf=1.0000)
- **Noise gate**: floor=0.0000, threshold=0.0000, passes=NO
- **All proposals (4):**
  1. `continuous_ratio` → `ratio` (conf=1.0000): Error shape suggests continuous_ratio with ratio in relativistic domain (quantities: v, t, x, c, tau)
  2. `continuous_ratio` → `identity` (conf=0.6500): Relativistic linear residuals — propose continuous multiplier gamma
  3. `integer_n` → `identity` (conf=0.6500): Linear residuals suggest counting/index pattern
  4. `integer_n` → `squared` (conf=0.4000): Generic hidden integer variable (fallback)
- **Time**: 4.04s

### doppler_shift — ⚠️ NO DISCOVERY

- **Name**: Relativistic Doppler shift
- **Domain**: relativistic
- **Known invariant**: `f / sqrt((1-beta)/(1+beta))`
- **Hidden variable**: `gamma`
- **Error shape**: linear (confidence=1.0000, CV=0.0000)
- **Baseline** (without hidden var): `` score=0.0000 not discovered
- **Augmented** (with hidden var): `` score=0.0000 not discovered
- **Proposal**: `continuous_ratio` → `ratio` (conf=1.0000)
- **Noise gate**: floor=0.0000, threshold=0.0000, passes=NO
- **All proposals (4):**
  1. `continuous_ratio` → `ratio` (conf=1.0000): Error shape suggests continuous_ratio with ratio in relativistic domain (quantities: v, t, c, f)
  2. `continuous_ratio` → `identity` (conf=0.6500): Relativistic linear residuals — propose continuous multiplier gamma
  3. `integer_n` → `identity` (conf=0.6500): Linear residuals suggest counting/index pattern
  4. `integer_n` → `squared` (conf=0.4000): Generic hidden integer variable (fallback)
- **Time**: 3.07s

---

## Classical Verification (False Positive Check)

These scenarios should already be discoverable without hidden variables.
If they fail, something is broken. If they succeed, we confirm the system
doesn't hallucinate hidden variables where none are needed.

### simple_pendulum — ✅ PASS (no FP)

- **Name**: Simple pendulum (L=1.0m, theta0=10°)
- **Domain**: gravity
- **Known invariant**: `m*g*h + 0.5*m*v^2`
- **Baseline**: `0.5*h/h*m*g*L` score=1.0000
- **Noise gate**: PASS

### mass_spring — ✅ PASS (no FP)

- **Name**: Mass-spring (k=10.0, m=1.0, A=0.5)
- **Domain**: spring
- **Known invariant**: `0.5*k*x^2 + 0.5*m*v^2`
- **Baseline**: `-1*k*x*x+-1*v*m*v` score=1.0000
- **Noise gate**: PASS

---

## Assessment

### Verdict: PASS

The system was trained exclusively on pre-1905 physics. Extended hidden
variable discovery tests probe its ability to diagnose missing variables
across quantum (quantum numbers, spin, spectroscopic patterns) and
relativistic (gamma factor, velocity addition, Doppler shift) domains.

### Key findings

- ✅ **angular_momentum**: E/n/n (score=1.0000)
- ✅ **spin_measurement**: E/n (score=0.9733)
- ✅ **blackbody_peak**: E_photon/T (score=1.0000)
- ✅ **photoelectric**: f/f (score=1.0000)
- ✅ **velocity_addition**: u+u (score=1.0000)
- ✅ **relativistic_momentum**: E/gamma (score=1.0000)
- ❌ **time_dilation**: failed (best=, score=0.0000)
- ❌ **doppler_shift**: failed (best=, score=0.0000)
- ✅ **simple_pendulum**: 0.5*h/h*m*g*L (score=1.0000)
- ✅ **mass_spring**: -1*k*x*x+-1*v*m*v (score=1.0000)