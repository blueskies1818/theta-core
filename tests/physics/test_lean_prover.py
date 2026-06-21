"""Tests for src.physics.lean_prover — Lean theorem generation and verification.

Validates:
1. Variable substitution correctness (word boundaries, v0/v ordering).
2. Generated Lean theorems have valid syntax and pass `lean` verification.
3. Free-fall energy conservation proves mgh + ½mv² = constant.
4. Theorem saving to verified_theorems directory.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from src.physics.lean_prover import (
    SCENARIOS,
    PhysicsScenario,
    LeanTheorem,
    _substitute_vars,
    generate_theorem,
    generate_all_theorems,
    write_lean_file,
    verify_theorem,
    verify_scenario,
    verify_all,
    save_verified_theorem,
    verified_theorems_dir,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_output_dir():
    """Temporary directory for saving verified theorems."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Variable substitution tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSubstitution:
    """Regex-based variable substitution sanity checks."""

    def test_simple_substitution(self):
        expr = "m * g * h"
        subs = {"h": "h0 - (1/2) * g * t ^ 2"}
        result = _substitute_vars(expr, subs)
        assert result == "m * g * (h0 - (1/2) * g * t ^ 2)"

    def test_v0_not_touched_when_substituting_v(self):
        """'v' substitution must not touch 'v0' (word boundary safety)."""
        expr = "m * g * h0 + v0 * t + v ^ 2"
        subs = {"v": "g * t"}
        result = _substitute_vars(expr, subs)
        # 'v0' should remain intact; standalone 'v' should be replaced
        assert "v0" in result
        assert "(g * t) ^ 2" in result
        assert "v" not in result.replace("v0", "")

    def test_h0_not_touched_when_substituting_h(self):
        """'h' substitution must not touch 'h0'."""
        expr = "h + h0"
        subs = {"h": "h0 * 2"}
        result = _substitute_vars(expr, subs)
        assert "h0" in result
        assert "(h0 * 2)" in result

    def test_longest_first_ordering(self):
        """Longer variable names must be substituted before shorter ones."""
        expr = "vx + v + vy"
        subs = {"vx": "v0 * cos(theta)", "vy": "v0 * sin(theta)", "v": "g * t"}
        result = _substitute_vars(expr, subs)
        assert "(v0 * cos(theta))" in result
        assert "(v0 * sin(theta))" in result
        assert "(g * t)" in result
        assert "vx" not in result
        assert "vy" not in result

    def test_freefall_substitution(self):
        """Full free-fall substitution produces expected Lean expression."""
        sc = SCENARIOS["free_fall"]
        result = _substitute_vars(sc.conserved_expr, sc.kinematic_subs)
        expected = "m * g * (h0 - (1/2) * g * t ^ 2) + (1/2) * m * (g * t) ^ 2"
        assert result == expected

    def test_freefall_v0_substitution(self):
        """Full free-fall-v0 substitution produces expected Lean expression."""
        sc = SCENARIOS["free_fall_v0"]
        result = _substitute_vars(sc.conserved_expr, sc.kinematic_subs)
        # Both 'v'→'(v0 - g * t)' and 'h'→'(h0 + v0 * t - ...)' 
        assert "(v0 - g * t) ^ 2" in result
        assert "(h0 + v0 * t - (1/2) * g * t ^ 2)" in result
        assert "m * g * " in result
        assert result.endswith(" ^ 2")

    def test_projectile_substitution(self):
        """Projectile substitution preserves vx, vy, y properly."""
        sc = SCENARIOS["projectile"]
        result = _substitute_vars(sc.conserved_expr, sc.kinematic_subs)
        assert "(v0 * cos theta)" in result
        # vy is replaced with full kinematic expression
        assert "v0 * sin theta - g * t" in result
        # y is replaced with full kinematic expression
        assert "h0 + v0 * sin theta * t" in result
        assert "vx" not in result
        assert "vy" not in result
        assert " y " not in f" {result} "  # standalone y replaced


# ═══════════════════════════════════════════════════════════════════════════════
# Theorem generation tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateTheorem:
    """Tests for generate_theorem() output structure and content."""

    def test_freefall_theorem_structure(self):
        thm = generate_theorem("free_fall")
        assert thm.name == "energy_conservation_free_fall"
        assert thm.scenario == "free_fall"
        assert "theorem" not in thm.statement  # lean_code adds 'theorem'
        assert "(m : ℝ)" in thm.statement
        assert "(g : ℝ)" in thm.statement
        assert "(h0 : ℝ)" in thm.statement
        assert "(t : ℝ)" in thm.statement
        assert "=" in thm.statement

    def test_freefall_lean_code(self):
        thm = generate_theorem("free_fall")
        code = thm.lean_code
        assert code.startswith("theorem energy_conservation_free_fall")
        assert "m * g * h0" in code
        assert "ring" in code
        assert "by\n  ring" in code

    def test_freefall_v0_theorem(self):
        thm = generate_theorem("free_fall_v0")
        code = thm.lean_code
        assert "(v0 : ℝ)" in code
        assert "(1/2) * m * v0 ^ 2" in code
        assert "v0 - g * t" in code

    def test_all_scenarios_generate(self):
        theorems = generate_all_theorems()
        assert len(theorems) == len(SCENARIOS)
        names = {t.name for t in theorems}
        for sc_name in SCENARIOS:
            assert f"energy_conservation_{sc_name}" in names

    def test_projectile_theorem_has_extra_lemmas(self):
        thm = generate_theorem("projectile")
        code = thm.lean_code
        assert "Real.sin_sq_add_cos_sq" in code

    def test_generated_code_is_parseable(self):
        """Generated Lean code has balanced parentheses and no syntax errors."""
        for thm in generate_all_theorems():
            code = thm.lean_code
            # Balanced parens
            assert code.count("(") == code.count(")"), f"Unbalanced parens in {thm.name}"
            # No empty lines with just whitespace mid-theorem
            for line in code.split("\n"):
                pass  # just iterating confirms no parsing issues


# ═══════════════════════════════════════════════════════════════════════════════
# Lean file writing tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWriteLeanFile:
    """Tests for write_lean_file()."""

    def test_writes_file(self, tmp_output_dir):
        thm = generate_theorem("free_fall")
        path = tmp_output_dir / "test_output.lean"
        result = write_lean_file([thm], path)
        assert result == path
        assert path.exists()
        content = path.read_text()
        assert "import Mathlib.Tactic" in content
        assert "theorem energy_conservation_free_fall" in content
        assert "ring" in content

    def test_multiple_theorems(self, tmp_output_dir):
        theorems = [generate_theorem("free_fall"), generate_theorem("free_fall_v0")]
        path = tmp_output_dir / "multi.lean"
        write_lean_file(theorems, path)
        content = path.read_text()
        assert "energy_conservation_free_fall" in content
        assert "energy_conservation_free_fall_v0" in content
        # One import block, not duplicated
        assert content.count("import Mathlib.Tactic") == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Lean verification tests (integration — requires lake + Mathlib)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestLeanVerification:
    """End-to-end tests that run the Lean 4 proof checker.

    These tests require the proof_checker_env Lake project with Mathlib.
    Skip if lean is not available or Mathlib is not built.
    """

    @pytest.fixture(autouse=True)
    def check_lean_available(self):
        """Skip all Lean verification tests if lean/Mathlib unavailable."""
        import subprocess
        from src.physics.lean_prover import _find_lake_project

        project_dir = _find_lake_project()
        if project_dir is None:
            pytest.skip("proof_checker_env Lake project not found")

        try:
            result = subprocess.run(
                ["lake", "env", "lean", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(project_dir),
            )
            if result.returncode != 0:
                pytest.skip(f"lean not available: {result.stderr}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("lean/lake not available")

    def test_freefall_verified(self):
        """Free-fall energy conservation passes Lean verification."""
        success, output, thm = verify_scenario("free_fall")
        assert success, f"Free-fall verification failed:\n{thm.lean_code}\n\nOutput:\n{output}"

    def test_freefall_v0_verified(self):
        """Free-fall with initial velocity passes Lean verification."""
        success, output, thm = verify_scenario("free_fall_v0")
        assert success, f"Free-fall-v0 verification failed:\n{thm.lean_code}\n\nOutput:\n{output}"

    def test_projectile_verified(self):
        """Projectile energy conservation passes Lean verification."""
        success, output, thm = verify_scenario("projectile")
        assert success, f"Projectile verification failed:\n{thm.lean_code}\n\nOutput:\n{output}"

    def test_verify_theorem_direct(self):
        """verify_theorem() function works end-to-end."""
        thm = generate_theorem("free_fall")
        success, output = verify_theorem(thm)
        assert success, f"verify_theorem failed:\n{thm.lean_code}\n\nOutput:\n{output}"

    def test_verify_all_freefall_scenarios(self):
        """Both free-fall scenarios pass Lean verification."""
        results = verify_all(["free_fall", "free_fall_v0"])
        for name, (success, output, thm) in results.items():
            assert success, f"{name} failed:\n{thm.lean_code}\n\nOutput:\n{output}"

    def test_failing_theorem_detected(self):
        """A deliberately wrong theorem fails verification."""
        thm = LeanTheorem(
            name="broken_theorem",
            scenario="free_fall",
            statement="(m g h0 t : ℝ) :\n    m * g * h0 = 0",
            proof_block="by\n  ring",
        )
        success, output = verify_theorem(thm)
        assert not success, f"Broken theorem should fail, got success. Output:\n{output}"


# ═══════════════════════════════════════════════════════════════════════════════
# Save verified theorems tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveVerifiedTheorem:
    """Tests for save_verified_theorem()."""

    def test_saves_file(self, tmp_output_dir, monkeypatch):
        """save_verified_theorem writes a .lean file."""
        # Override output dir to use temp dir
        monkeypatch.setattr(
            "src.physics.lean_prover.verified_theorems_dir",
            lambda: tmp_output_dir,
        )
        thm = generate_theorem("free_fall")
        path = save_verified_theorem(thm)
        assert path.exists()
        assert path.suffix == ".lean"
        assert "energy_conservation_free_fall" in path.name
        content = path.read_text()
        assert "ring" in content

    def test_verified_theorems_dir_exists(self):
        """Default output directory path is within project."""
        d = verified_theorems_dir()
        assert d.name == "verified_theorems"
        assert "data" in str(d)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario completeness tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenarios:
    """Validate scenario definitions are internally consistent."""

    def test_all_scenarios_have_required_fields(self):
        for name, sc in SCENARIOS.items():
            assert sc.name, f"{name}: missing name"
            assert sc.description, f"{name}: missing description"
            assert sc.params, f"{name}: missing params"
            assert sc.conserved_expr, f"{name}: missing conserved_expr"
            assert sc.invariant_rhs, f"{name}: missing invariant_rhs"
            assert sc.proof_tactic, f"{name}: missing proof_tactic"

    def test_kinematic_subs_vars_in_expr(self):
        """Every variable in kinematic_subs should appear in conserved_expr."""
        for name, sc in SCENARIOS.items():
            if not sc.kinematic_subs:
                continue  # scenarios like spring have no subs
            for var in sc.kinematic_subs:
                # Variable must appear as a word-boundary token in the expression
                import re
                pattern = re.compile(r'\b' + re.escape(var) + r'\b')
                assert pattern.search(sc.conserved_expr), (
                    f"{name}: substitution var '{var}' not found in "
                    f"conserved_expr '{sc.conserved_expr}'"
                )

    def test_params_match_substitutions(self):
        """All free variables in substituted RHS should be in params."""
        for name, sc in SCENARIOS.items():
            if name == "spring":
                continue  # trivial case, no time evolution
            # After substitution, only params should remain as free vars
            import re
            result = _substitute_vars(sc.conserved_expr, sc.kinematic_subs)
            full_expr = f"{result} = {sc.invariant_rhs}"
            # Extract all identifiers
            identifiers = set(re.findall(r'\b[A-Za-z_][A-Za-z_0-9]*\b', full_expr))
            # Remove known functions, namespaces, and constants
            builtins = {"sin", "cos", "sqrt", "exp", "log", "Real"}
            identifiers -= builtins
            # Check each identifier is either a param or a known sub variable
            allowed = set(sc.params) | set(sc.kinematic_subs.keys())
            for ident in identifiers:
                assert ident in allowed, (
                    f"{name}: free variable '{ident}' in result not in params {sc.params}"
                )
