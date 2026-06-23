import Mathlib.Tactic
open Real
open Set
open Function
open Nat

theorem energy_conservation_pendulum (m : ℝ) (g : ℝ) (L : ℝ) (theta : ℝ) (theta0 : ℝ) :
    m * g * L * (1 - cos theta) + m * g * L * (cos theta - cos theta0) = m * g * L * (1 - cos theta0) := by
  ring