import Mathlib.Tactic
open Real
open Set
open Function
open Nat

theorem energy_invariant_relativistic (m : ℝ) (c : ℝ) (v : ℝ) (gamma : ℝ) (E : ℝ) (p : ℝ) (gamma_sq : gamma ^ 2 = (1 : ℝ) / (1 - (v / c) ^ 2)) :
    (gamma * m * c ^ 2) ^ 2 - ((gamma * m * v) * c) ^ 2 = (m * c ^ 2) ^ 2 := by
  ring