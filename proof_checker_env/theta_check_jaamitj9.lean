import Mathlib.Tactic
open Real
open Set
open Function
open Nat

theorem auto_energy_invariant_relativistic_direct (m : ℝ) (c : ℝ) (v : ℝ) (gamma : ℝ) (E : ℝ) (p : ℝ) :
    (gamma * m * c * c) ^ 2 - ((gamma * m * v) * c) ^ 2 = (m * c ^ 2) ^ 2 := by
  field_simp
  ring_nf