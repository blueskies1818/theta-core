import Mathlib.Tactic
open Real
open Set
open Function
open Nat

example (p q : Polynomial ℝ) (hp : p.natDegree > q.natDegree) (hq : q ≠ 0) : (p + q).natDegree = p.natDegree := by
  rw [Nat.cast_add]
  exact degree_lt_degree