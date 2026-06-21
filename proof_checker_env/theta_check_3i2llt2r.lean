import Mathlib.Tactic
open Real
open Set
open Function
open Nat

example (v c : ℝ) (hc : c > 0) (hv : |v| < c) : v^2 / c^2 < 1 := by apply field