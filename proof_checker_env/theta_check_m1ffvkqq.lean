import Mathlib.Tactic
open Real
open Set
open Function
open Nat

theorem energy_invariant_relativistic_direct (m : ℝ) (c : ℝ) (v : ℝ) (gamma : ℝ) (E : ℝ) (p : ℝ) (h_gamma_id : gamma ^ 2 * (c ^ 2 - v ^ 2) = c ^ 2) :
    (gamma * m * c * c) ^ 2 - ((gamma * m * v) * c) ^ 2 = (m * c ^ 2) ^ 2 := by
  have h_factor : (gamma * m * c * c) ^ 2 - ((gamma * m * v) * c) ^ 2 = (m * c) ^ 2 * (gamma ^ 2 * (c ^ 2 - v ^ 2)) := by ring
  rw [h_factor]
  rw [h_gamma_id]
  ring