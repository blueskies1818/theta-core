# Phase 2 Data Collection & Correspondence Wiring Review

**Date:** 2026-06-03
**Status:** Complete — data encoded, correspondence layer wired into training loop
**Branch:** `main`
**Covers:** ROADMAP 2.5 (frontier map), 2.7 (failure coordinates), 2.6 (experimental data encoding, partial)

---

## 1. Executive Summary

**Goal:** Build the correspondence layer — the compass that tells the GNN+MCTS explorer *where* to search and *what* to solve. Without this, the explorer has no sense of which mathematical territory matters and which proofs are significant.

**Verdict: The correspondence layer is built, tested, and wired into the training loop.** The frontier map (13 zones), failure coordinate system (12 points), and physical constants database (192 entries with temporal gating) are all functional. The reward integration module connects them to the GRPO training loop in the explorer trainer. Tier 2 data files (GW150914 strain, Pantheon+ SNIa, Planck CMB) have been downloaded and verified.

The explorer now has a compass. Rewards are shaped by frontier zone (Planck breakdown = 3.0×, thermodynamics = 0.1×) and failure behavior (resolving a singularity = +2.5 bonus, reproducing one = −1.5 penalty).

---

## 2. What Was Built

### 2.1 Frontier Map (`src/correspondence/frontier.py`, `configs/frontier_map.yaml`)

ROADMAP 2.5. Machine-readable map of mathematical-physical space with three zones.

| Metric | Value |
|--------|-------|
| Total zones | 13 |
| Established zones | 4 (gr_classical, standard_model, qed, thermodynamics) |
| Uncertain zones | 4 (quantum_gravity, dark_matter, dark_energy, inflation) |
| Breakdown zones | 5 (planck_breakdown, black_hole_singularity, big_bang_singularity, qft_divergence, gr_qft_incompatibility) |
| Boundary condition types | 7 (energy_scale, length_scale, curvature, gauge_group, theorem_dependent, topological, conservation) |
| Lines of code | 577 (frontier.py) |

**Reward shaping:** Established zones get multipliers < 1.0 (de-prioritize re-discovery). Breakdown zones get multipliers up to 3.0× (pull explorer toward the frontier). The `classify()` method uses boundary conditions with explicit energy scales, gauge groups, and theorem dependencies — falling back to keyword matching when conditions aren't met.

### 2.2 Failure Coordinates (`src/correspondence/failure_points.py`, `configs/failure_coordinates.yaml`)

ROADMAP 2.7. Formal encoding of exact conditions where current theories produce infinities.

| Metric | Value |
|--------|-------|
| Total failure points | 12 |
| Catastrophic (literal infinities) | 4 (Planck divergence, black hole singularity, Big Bang t=0, non-renormalizable QG) |
| Pathological (mathematically valid, physically wrong) | 3 (cosmological constant, hierarchy, strong CP) |
| Incomplete (theory is silent) | 3 (dark matter identity, neutrino masses, baryon asymmetry) |
| Tension (predictions contradict) | 2 (GR-QFT incompatibility, black hole information paradox) |
| Lines of code | 437 (failure_points.py) |

**Reward calibration:** Resolving all catastrophic failures = +10.0. Reproducing one = −1.5. The `estimate_reward_modifier()` method computes net bonus/penalty from resolved/reproduced sets.

### 2.3 Physical Constants Database (`src/data/physical/constants.py`)

ROADMAP 2.6 (partial). All Tier 1 physical constants with discovery years for temporal gating.

| Section | Entries | Highlights |
|---------|---------|-----------|
| Fundamental constants | 21 | c, G, h, e, m_e, m_p, α, k_B, R_∞, Planck units |
| Particle properties | 23 | All SM particles + key hadrons |
| Spectral lines | 18 | H Lyman/Balmer/Paschen, He, Na D, 21 cm |
| Cosmological params | 14 | H₀, Ω_m, Ω_Λ, n_s, σ_8, S_8, t₀ |
| Nuclear properties | 17 | BBN nuclei, Fe peak, magic numbers |
| Neutrino params | 8 | Δm², mixing angles, δ_CP, m_ββ limit |
| Anomalies | 8 | H₀ tension, g-2, S_8, CDF W, CC problem |
| Flavor physics | 19 | CKM magnitudes, β/α/γ, mixing, ε_K, ε'/ε |
| More hadrons | 13 | ρ, ω, φ, η, η', J/ψ, Υ, D⁰, B⁰, B_s, Λ⁰, Σ⁺, Ξ⁻ |
| Thermodynamic | 14 | c_p, latent heats, critical points, R, V_m |
| GR solar system tests | 9 | Mercury, Eddington, Cassini, LLR, GP-B, Hulse-Taylor, double pulsar, EHT |
| Equivalence principle | 5 | Eötvös, Eöt-Wash, MICROSCOPE, LLR Nordtvedt |
| Direct detection | 6 | LZ, XENONnT, ADMX, KamLAND-Zen 0νββ, Super-K p, nEDM |
| Periodic table | 16 | H through U with ionization energies, electronegativities |
| **Total** | **192** | |

Every entry carries a `discovery_year`. Filtering to ≤1904 reveals only the electron (no photon, no quarks). Filtering to ≤1964 reveals 12 particles but no quarks. Filtering to ≤2026 gives all 192 entries plus 8 current anomalies.

### 2.4 Temporal Gating (`ERA_CUTOFFS`)

The database supports chronology-gated evaluation — the key innovation from the data plan:

| Era | Cutoff | Entries | Key missing |
|-----|--------|---------|-------------|
| Classical | ≤1860 | 37 | Maxwell, QM, relativity |
| Classical crisis | ≤1900 | 59 | Photon, relativity, QM |
| Pre-relativity | ≤1904 | 59 | Special relativity |
| Pre-GR | ≤1914 | 68 | General relativity |
| Old quantum | ≤1925 | 82 | Matrix/wave mechanics |
| Pre-QED | ≤1946 | 93 | Lamb shift, g-2 |
| Pre-SM | ≤1965 | 118 | Electroweak, QCD |
| SM construction | ≤1975 | 125 | τ, b, W/Z discovery |
| SM confirmed | ≤1995 | 143 | Top, neutrino oscillations |
| Precision era | ≤2010 | 173 | LHC, Planck, GW |
| Modern | ≤2026 | 192 | (none) |

This enables the strongest validation: train on pre-1905 data, test whether the system discovers special relativity.

### 2.5 Reward Integration (`src/correspondence/reward_integration.py`)

The bridge between the correspondence layer and the GRPO training loop.

| Component | Role |
|-----------|------|
| `CorrespondenceRewardModifier` | Wraps frontier map + failure coords into a single reward modifier |
| `apply(rewards, proofs, statements)` | Classifies each proof, computes zone multiplier + failure modifier, returns modified rewards |
| `_classify_proof()` | Keyword + condition-based zone classification from theorem statement + proof text |
| `_check_failure_points()` | Detects failure point mentions and resolution claims |
| `create_default_modifier()` | Factory that loads from YAML configs |

**Reward formula:**
```
modified_reward = base_reward × [1.0 + scale × (zone_multiplier − 1.0)]
                  + bonus_scale × max(0, failure_modifier)
                  − penalty_scale × max(0, −failure_modifier)
```

Failed proofs (reward < 1.0) are not modified — the correspondence layer only shapes successful proof rewards.

### 2.6 Explorer Trainer Wiring

Modified `src/explorer/explorer_trainer.py`:

- **Constructor**: Accepts optional `CorrespondenceRewardModifier`. Auto-loads from default configs if `use_correspondence=True`.
- **Phase D2**: New step between reward computation (Phase D) and advantage calculation (Phase E) that applies the correspondence modifier.
- **Logging**: Training logs now show correspondence stats (breakdown/established/uncertain hits, resolutions, reproductions).
- **`ExplorerConfig`**: New fields `use_correspondence`, `correspondence_energy_scale`, `correspondence_gauge_group`.

### 2.7 Tier 2 Data Downloads

| File | Size | Contents | Source |
|------|------|----------|--------|
| `GW150914_H1_4KHZ_strain.hdf5` | 1.0 MB | LIGO Hanford strain, 4 kHz, 32s around event | GWOSC via `gwpy` |
| `GW150914_L1_4KHZ_strain.hdf5` | 964 KB | LIGO Livingston strain | GWOSC via `gwpy` |
| `pantheon_plus_SH0ES.dat` | 566 KB | 1701 SNe Ia, z=0.001–2.26, 47 columns | GitHub (PantheonPlusSH0ES) |
| `planck_2018_CMB_TT_binned.txt` | 7 KB | Planck TT power spectrum, 83 bins | ESA Planck Legacy Archive |

**Not downloaded** (requires browser authentication or manual steps):
- Planck TE/EE/lensing spectra — PLA registration needed (free)
- LEP Z-pole JSON — HEPData API returns HTML redirect
- SDSS BAO measurements — requires SVN checkout

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  CORRESPONDENCE LAYER                        │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                  │
│  │  Frontier Map    │  │  Failure Coords  │                  │
│  │  (13 zones)      │  │  (12 points)     │                  │
│  │  ZoneType × 3    │  │  Severity × 4    │                  │
│  │  BoundaryCond × 7│  │  Regime × 7      │                  │
│  └────────┬─────────┘  └────────┬─────────┘                  │
│           │                     │                            │
│           └──────────┬──────────┘                            │
│                      ▼                                       │
│  ┌──────────────────────────────────────┐                    │
│  │  CorrespondenceRewardModifier        │                    │
│  │  • classify(proof) → zone            │                    │
│  │  • check_failures(proof) → (R, P)    │                    │
│  │  • apply(rewards) → modified_rewards │                    │
│  └──────────────────┬───────────────────┘                    │
│                     │                                        │
├─────────────────────┼────────────────────────────────────────┤
│                     ▼                                        │
│  ┌──────────────────────────────────────┐                    │
│  │  Physical Constants DB               │                    │
│  │  192 entries × discovery_year        │                    │
│  │  Temporal gating (ERA_CUTOFFS)       │                    │
│  └──────────────────────────────────────┘                    │
│                                                              │
│  ┌──────────────────────────────────────┐                    │
│  │  Tier 2 Data Files                   │                    │
│  │  GW150914 (HDF5, 2 MB)               │                    │
│  │  Pantheon+ (ASCII, 566 KB)           │                    │
│  │  Planck TT (ASCII, 7 KB)             │                    │
│  └──────────────────────────────────────┘                    │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  EXPLORER TRAINER                           │
│                                                             │
│  Phase A: GNN embeddings                                    │
│  Phase B: MCTS proof search                                 │
│  Phase C: Proof checker                                     │
│  Phase D: Base rewards (correctness + curiosity + length)   │
│  Phase D2: ◄── CorrespondenceRewardModifier.apply()         │
│            • Zone multiplier (3.0× Planck, 0.1× thermo)     │
│            • Failure bonus (+2.5 resolve, −1.5 reproduce)   │
│  Phase E: Group-relative advantages                         │
│  Phase F: GRPO loss → backward                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. File Inventory

### New files

```
src/correspondence/
├── frontier.py                  # Frontier map data structures + factory (577 lines)
├── failure_points.py            # Failure coordinate system + factory (437 lines)
├── reward_integration.py        # Reward modifier for training loop (295 lines)
└── __init__.py                  # Updated exports (91 lines)

src/data/physical/
├── __init__.py                  # Package exports (47 lines)
└── constants.py                 # 192 entries × discovery_year (1950+ lines)

configs/
├── frontier_map.yaml            # 13 zones with boundary conditions (483 lines)
└── failure_coordinates.yaml     # 12 failure points (334 lines)

data/physical/downloads/
├── GW150914_H1_4KHZ_strain.hdf5 # LIGO Hanford strain (1.0 MB)
├── GW150914_L1_4KHZ_strain.hdf5 # LIGO Livingston strain (964 KB)
├── pantheon_plus_SH0ES.dat      # 1701 SNe Ia (566 KB)
├── pantheon_plus_README.md      # Column documentation (4 KB)
└── planck_2018_CMB_TT_binned.txt # Planck TT spectrum (7 KB)

docs/
└── data_required.md             # Full data inventory by era (108 lines)
```

### Modified files

```
src/correspondence/__init__.py   # Added frontier, failure_points, reward_integration exports
src/explorer/explorer_trainer.py # Added correspondence modifier in constructor + Phase D2 + logging + ExplorerConfig fields
```

### Existing files (unchanged, still used)

```
src/correspondence/limits.py     # ExperimentalDomain, LimitRegime, CorrespondenceResult
src/explorer/gnn_encoder.py      # GAT implementation
src/explorer/mcts.py             # PUCT proof search
src/explorer/structure_generator.py # Structure templates + mutations
src/reward/base.py               # compute_rewards_batch, compute_group_advantages
```

---

## 5. Integration Test Results

All tests pass:

| Test | Result |
|------|--------|
| Frontier map loads (13 zones) | ✓ |
| Failure coords load (12 points) | ✓ |
| Energy scale classification (Planck → breakdown) | ✓ |
| Gauge group classification (SM → standard_model) | ✓ |
| Gauge group classification (U(1) → qed) | ✓ |
| Keyword fallback (planck → breakdown) | ✓ |
| Keyword fallback (standard model → established) | ✓ |
| Keyword fallback (dark matter → uncertain) | ✓ |
| Failure resolution detection (Planck divergence) | ✓ |
| Failure resolution detection (bounce solution) | ✓ |
| Reward modifiers (resolve Planck+GR/QFT = +6.0) | ✓ |
| Reward modifiers (reproduce BH singularity = −1.5) | ✓ |
| Reward modifiers (reproduce BH+CC = −2.5) | ✓ |
| create_default_modifier() factory | ✓ |
| ExplorerConfig has correspondence fields | ✓ |
| Explorer trainer imports CorrespondenceRewardModifier | ✓ |
| Phase D2 present in training loop | ✓ |
| Temporal gating (≤1904: electron only, no photon) | ✓ |
| YAML round-trip (frontier map 13 zones preserved) | ✓ |
| YAML round-trip (failure coords 12 points preserved) | ✓ |
| GW150914 HDF5 strain validates (131,072 samples, 32s) | ✓ |
| Pantheon+ data validates (1,701 SNe Ia, z=0.001–2.26) | ✓ |
| Planck TT spectrum validates (83 bins) | ✓ |

---

## 6. Deviations from ROADMAP

### Deviation 1: 2.5, 2.6, 2.7 built as integrated layer

**ROADMAP says:** Three separate sub-phases (2.5 frontier map, 2.6 experimental checks, 2.7 failure coordinates).

**What we did:** Built all three as a single correspondence layer with shared infrastructure. The frontier map and failure coordinates share boundary condition types. The reward integration module uses both together. The constants database supports both.

**Why:** The three are interdependent. The frontier map defines zones; failure points are the anchors of breakdown zones. Reward shaping needs both simultaneously — a zone multiplier without failure awareness would pull toward breakdown without distinguishing resolution from reproduction. Building them together avoided rework.

### Deviation 2: 2.6 split into encodable vs. data-dependent

**ROADMAP says:** Full experimental reproduction checks including numerical comparison against collider cross-sections and GW strain data.

**What we did:** Built the formal encoding portion (192 constants with temporal gating, frontier zone boundaries, failure conditions). Downloaded 3 of 12 Tier 2 datasets. Deferred numerical distributional comparisons to Phase 3.

**Why:** The ROADMAP itself acknowledges that measurement data acquisition is Phase 3.3. The formal encodings (constants, zone boundaries, failure conditions) are buildable now and provide immediate value for reward shaping. Numerical curve-fitting against real data distributions requires the Phase 3 scoring pipeline.

### Deviation 3: Temporal gating added (not in ROADMAP)

**What we did:** Every data point carries a `discovery_year`. The `get_data_up_to_year(year)` function implements chronology-gated evaluation.

**Why:** This transforms the system from "can it fit known physics?" to "can it discover physics the way humans did, from the same experimental constraints?" — a much stronger test of whether the architecture does genuine science. This emerged from the data collection discussion and is now a first-class evaluation framework.

### Deviation 4: explorer_trainer.py modified for correspondence

**What we did:** Added Phase D2 to the training loop and `CorrespondenceRewardModifier` to the constructor.

**Why:** The ROADMAP describes the frontier map as "guides exploration" without specifying the exact integration point. The natural hook is between reward computation and advantage calculation — after we know whether the proof is valid, before we compute the policy gradient.

---

## 7. What's Not Done

### Tier 2: Remaining datasets

| Dataset | Status | Blocker |
|---------|--------|---------|
| Planck TE/EE/lensing | Not downloaded | PLA authentication |
| LEP Z-pole JSON | Not downloaded | HEPData API redirect |
| SDSS BAO | Not downloaded | SVN checkout |
| GW170817 strain | Not downloaded | Not attempted yet |
| LHC Higgs couplings JSON | Not downloaded | Not attempted yet |

These are not blocking — the constants database and existing downloads are sufficient for the correspondence layer to function. The remaining datasets are needed for Phase 3 numerical comparisons.

### Phase 2.8: Scale to 3B parameters

Not started. Blocked on hardware (Intel Arc B70 34 GB). Current GNN is 856K params at 256-dim. Scaling to 3B would require cloud GPU (A100 80GB).

### Phase 3: Physical grounding

Not started. This is the next major phase — measurement data pipelines, numerical prediction scoring, domain holdout evaluation.

---

## 8. Next Steps

### Immediate (Phase 2 closure)
1. **End-to-end training run** — Run the explorer trainer with correspondence modifier enabled on a small theorem set. Verify that zone multipliers and failure modifiers actually affect the reward distribution and gradient signal.
2. **Temporal gating evaluation** — Filter constants to ≤1904. Run MCTS on classical physics theorems. Does it prioritize Maxwell's equations and the blackbody problem?

### Phase 3 transition
3. **Download remaining Tier 2** — Register at PLA, get GW170817, LHC Higgs JSON
4. **Build measurement parsing pipelines** — HDF5 → strain time series, ASCII → Hubble diagram, FITS → angular power spectrum
5. **Numerical comparison scorer** — Chi-squared / likelihood comparison of structure predictions against data distributions
6. **Wire physical scorer into reward pipeline** — Replace keyword-based failure detection with quantitative comparison

### Longer term
7. **3B parameter scaling** — Requires cloud GPU
8. **Phase 3.5 domain holdout** — Train on GW + particle + spectroscopic, hold out cosmology

---

## 9. Summary

The correspondence layer is built and wired. The explorer now has:
- **A compass** — 13 frontier zones with reward multipliers (3.0× → Planck breakdown, 0.1× → thermodynamics)
- **Negative waypoints** — 12 failure coordinates with resolution bonuses and reproduction penalties
- **A memory** — 192 physical constants with temporal gating, enabling chronology-gated evaluation
- **A steering mechanism** — `CorrespondenceRewardModifier` plugged into Phase D2 of the GRPO training loop

The explorer will now prioritize proofs that push toward the breakdown zone — the Planck scale, black hole singularities, the GR-QFT interface — while de-prioritizing re-derivations of established physics. The temporal gating infrastructure enables the strongest possible validation: restrict to historical data and test whether the system rediscovers the next generation of physics.

---

*Generated 2026-06-03. Correspondence layer complete. Ready for end-to-end training with reward shaping.*
