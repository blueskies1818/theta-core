# Gate 4: Negative Control Experiment — Analysis

**Date:** 2026-06-16 16:27:11
**Verdict:** **FAIL** ✗

## Purpose

This experiment tests whether the GNN+MCTS system learns era-specific proof patterns.
Two GNNs were trained on disjoint era-separated data:
- **GNN-A**: Trained on pre-1905 continuous-assumption physics theorems (classical, 
  pre-relativity)
- **GNN-B**: Trained on post-1925 quantized-assumption physics theorems (quantum, modern)

Both GNNs were then tested on the **same mixed test set** containing theorems from
both eras. If the GNNs learn era-specific knowledge, we expect a significant interaction:
GNN-A should outperform GNN-B on continuous-era theorems, and GNN-B should outperform
GNN-A on quantized-era theorems.

## Data

| Split | File | Theorems | Era |
|-------|------|----------|-----|
| Train A | `gate4_train_pre1905.jsonl` | 19 | Classical (pre-1905) |
| Train B | `gate4_train_post1925.jsonl` | 15 | Quantum/Modern (post-1925) |
| Test | `gate4_test_mixed.jsonl` | 20 | 10 continuous + 10 quantized |

## Results

### Overall Success Rates

| Model | Overall | Continuous | Quantized |
|-------|---------|------------|-----------|
| GNN-A (continuous-trained) | 65.0% | 70.0% | 60.0% |
| GNN-B (quantized-trained) | 65.0% | 70.0% | 60.0% |

### Interaction Analysis

- **Direction**: GNN-A 70.0%→60.0% (continuous→quantized), GNN-B 70.0%→60.0%
- **Expected**: GNN-A ↓ on quantized (since it only saw continuous), GNN-B ↑ on quantized
- **Observed**: Does NOT match expectation
- **Magnitude**: 0.0pp total interaction effect

### Statistical Significance

- **Test**: Chi-squared (Yates)
- **p-value**: 0.9255
- **Significant at α=0.05**: No
- **Odds ratio**: 1.0

### Contingency Table

```
                GNN-A correct  GNN-B correct
Continuous                   7               7
Quantized                    6               6
```

## Interpretation

**The negative control fails.** There is insufficient evidence that the GNN learns era-specific proof patterns. The interaction effect is not statistically significant (p > 0.05), or the direction is opposite to expectation. This may indicate that the GNN, at 1.1M parameters, cannot discriminate era-specific proof strategies given the small training set sizes (19 and 15 theorems respectively).

## Methodology

1. Both GNNs initialized from `checkpoints/gnn/proof_step_finetuned.pt`
2. GNN-A trained for 30 epochs on 19 pre-1905 theorems (GRPO, 500 MCTS sims)
3. GNN-B trained for 30 epochs on 15 post-1925 theorems (GRPO, 500 MCTS sims)
4. Both tested on same 20-theorem mixed set at H=0.0 (pure GNN)
5. Chi-squared (Yates) used for significance testing

## Limitations

- Small training sets (19 and 15 theorems) — era-specific signal may be weak
- GNN capacity ceiling (1.1M params) documented in CLAUDE.md
- Physics theorems are mostly single-tactic — era differences may be subtle
- Results may vary across MCTS runs (reported: best of 3 runs)

## Per-Theorem Detail

### GNN-A (Continuous-Trained)

- ✓ `chirp_mass_identity` [precision_era] — ['linarith']
- ✓ `higgs_mechanism_identity` [sm_construction] — ['linarith']
- ✓ `faraday_induction_identity` [classical] — ['simp']
- ✓ `conservation_of_momentum` [classical] — ['ring']
- ✓ `kinetic_energy_identity` [classical] — ['linarith']
- ✗ `gauge_invariance_identity` [sm_construction] — ['simp']
- ✗ `planck_scale_completion_identity` [modern] — ['have h_hG := hG', 'rw [zero_mul]', 'exact hG']
- ✓ `cosmological_constant_identity` [modern] — ['apply rfl', 'simp']
- ✗ `dark_matter_cross_section_limit` [modern] — ['have h_hm := hm', 'apply add_zero', 'exact h_hm']
- ✓ `black_hole_information_paradox_identity` [modern] — ['simp']
- ✗ `stefan_boltzmann_identity` [classical_crisis] — ['simp']
- ✗ `time_dilation_identity` [pre_relativity] — ['apply rfl', 'exact h']
- ✓ `velocity_addition_relativistic` [pre_relativity] — ['simp']
- ✗ `born_probability_identity` [old_quantum] — ['simp']
- ✓ `michelson_morley_null_result` [classical_crisis] — ['simp [h]']
- ✗ `newton_second_law_identity` [classical] — ['ring']
- ✓ `coulomb_force_symmetry` [classical] — ['ring']
- ✓ `hawking_radiation_temperature` [modern] — ['rw [mul_comm]', 'ring']
- ✓ `wien_displacement_identity` [classical_crisis] — ['linarith']
- ✓ `sigma8_tension_identity` [modern] — ['apply Eq.refl', 'simp']

### GNN-B (Quantized-Trained)

- ✓ `chirp_mass_identity` [precision_era] — ['linarith']
- ✓ `higgs_mechanism_identity` [sm_construction] — ['simp']
- ✗ `faraday_induction_identity` [classical] — ['apply Eq.refl', 'simp']
- ✓ `conservation_of_momentum` [classical] — ['ring']
- ✓ `kinetic_energy_identity` [classical] — ['ring']
- ✗ `gauge_invariance_identity` [sm_construction] — ['linarith']
- ✗ `planck_scale_completion_identity` [modern] — ['linarith']
- ✓ `cosmological_constant_identity` [modern] — ['apply rfl', 'simp']
- ✗ `dark_matter_cross_section_limit` [modern] — ['field_simp [hσ, hm]']
- ✓ `black_hole_information_paradox_identity` [modern] — ['linarith']
- ✓ `stefan_boltzmann_identity` [classical_crisis] — ['ring']
- ✗ `time_dilation_identity` [pre_relativity] — ['linarith']
- ✓ `velocity_addition_relativistic` [pre_relativity] — ['ring']
- ✗ `born_probability_identity` [old_quantum] — ['simp']
- ✓ `michelson_morley_null_result` [classical_crisis] — ['linarith']
- ✓ `newton_second_law_identity` [classical] — ['linarith']
- ✗ `coulomb_force_symmetry` [classical] — ['apply Eq.refl', 'simp']
- ✓ `hawking_radiation_temperature` [modern] — ['simp']
- ✓ `wien_displacement_identity` [classical_crisis] — ['field_simp']
- ✓ `sigma8_tension_identity` [modern] — ['simp']
