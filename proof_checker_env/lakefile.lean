import Lake
open Lake DSL

package proof_checker_env where
  leanOptions := #[⟨`pp.unicode.fun, true⟩]

@[default_target]
lean_lib ProofChecker where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
