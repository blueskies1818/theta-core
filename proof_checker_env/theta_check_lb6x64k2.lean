import Mathlib.Tactic
open Real
open Set
open Function
open Nat

example (a b x y : ℝ) : (a*x + b*y)^2 ≤ (a^2 + b^2)*(x^2 + y^2) := by simp