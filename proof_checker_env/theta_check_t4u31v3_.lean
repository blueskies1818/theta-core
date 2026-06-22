import Mathlib.Tactic
open Real
open Set
open Function
open Nat

theorem auto_energy_conservation_em_e_field (m : ℝ) (q : ℝ) (E : ℝ) (x0 : ℝ) (y0 : ℝ) (vx0 : ℝ) (vy0 : ℝ) (t : ℝ) :
    (1/2) * m * ((vx0 + (q * E / m) * t) ^ 2 + (vy0) ^ 2) - q * E * (x0 + vx0 * t + (1/2) * (q * E / m) * t ^ 2) = (1/2) * m * (vx0 ^ 2 + vy0 ^ 2) - q * E * x0 := by
  ring_nf