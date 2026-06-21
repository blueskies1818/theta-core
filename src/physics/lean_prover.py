"""Lean theorem generation for energy conservation proofs.

Given a discovered energy expression E (numerically constant across
observations), this module generates a Lean 4 theorem proving that E is
mathematically conserved under the given kinematic equations.

Architecture:
  1. PhysicsScenario defines the kinematics (substitution rules)
  2. _substitute_vars() replaces variables with kinematic expressions
  3. generate_theorem() produces a LeanTheorem with ring/nlinarith proofs
  4. verify_theorem()/verify_scenario() run Lean 4 to validate proofs

Scenarios:
  - free_fall:        Ball dropped from rest (v = g*t, h = h0 - ½*g*t²)
  - free_fall_v0:     Ball thrown with initial velocity v0
  - projectile:       Projectile with vx, vy decomposition + sin²+cos² identity
  - pendulum:         Simple pendulum energy (trigonometric)
  - spring:           Simple harmonic oscillator with ω² = k/m
  - spring_trig:      Spring energy via trig identity (sin² + cos² = 1)
  - work_energy:      Work-energy theorem: ΔKE = W
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from src.proof_checker.lean_interface import LeanProofChecker
from src.proof_checker.formats import ProofResult, LEAN_PREAMBLE_MATHLIB


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PhysicsScenario:
    """Definition of a physical scenario with kinematic substitution rules.

    Attributes:
        name: Short identifier (e.g. 'free_fall')
        description: Human-readable description
        params: Ordered list of parameter names for the theorem signature
        conserved_expr: The expression to prove is conserved (with variable names)
        invariant_rhs: The right-hand side (conserved quantity at t=0)
        kinematic_subs: Dict mapping variable → substitution expression
        proof_tactic: Tactic to apply after substitution (default: 'ring')
        extra_lemmas: Additional lemmas to prove alongside the main theorem
        notes: Extended documentation
    """

    name: str
    description: str
    params: list[str]
    conserved_expr: str
    invariant_rhs: str
    kinematic_subs: dict[str, str] = field(default_factory=dict)
    proof_tactic: str = "ring"
    extra_lemmas: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class LeanTheorem:
    """A generated Lean theorem with its proof.

    Attributes:
        name: Lean identifier (e.g. 'energy_conservation_free_fall')
        scenario: Which scenario this theorem proves
        statement: Type signature (parameters + goal)
        proof_block: The tactic proof block (body only)
        extra_code: Additional lemma statements before the theorem
    """

    name: str
    scenario: str
    statement: str
    proof_block: str
    extra_code: str = ""

    @property
    def lean_code(self) -> str:
        """Full Lean 4 code ready for verification."""
        parts = []
        if self.extra_code:
            parts.append(self.extra_code.strip())
        parts.append(
            f"theorem {self.name} {self.statement} := by\n"
            f"  {self.proof_block}"
        )
        return "\n\n".join(parts)


# ── Variable substitution ──────────────────────────────────────────────────────


def _substitute_vars(expr: str, subs: dict[str, str]) -> str:
    """Substitute variables in an expression using regex word boundaries.

    Variables are substituted longest-first to avoid partial matches
    (e.g. substituting 'vx' before 'v' so 'v' doesn't match inside 'vx').

    Args:
        expr: The expression containing variable names.
        subs: Dict mapping variable name → replacement expression.

    Returns:
        Expression with variables replaced, each substitution wrapped in parens.
    """
    result = expr
    # Sort by length descending so longer names are substituted first
    for var in sorted(subs, key=len, reverse=True):
        pattern = re.compile(r'\b' + re.escape(var) + r'\b')
        replacement = f"({subs[var]})"
        result = pattern.sub(replacement, result)
    return result


# ── Scenario definitions ───────────────────────────────────────────────────────


SCENARIOS: dict[str, PhysicsScenario] = {
    "free_fall": PhysicsScenario(
        name="free_fall",
        description="Ball dropped from rest in uniform gravity",
        params=["m", "g", "h0", "t"],
        conserved_expr="m * g * h + (1/2) * m * v ^ 2",
        invariant_rhs="m * g * h0",
        kinematic_subs={
            "v": "g * t",
            "h": "h0 - (1/2) * g * t ^ 2",
        },
        proof_tactic="ring",
        notes="v0 = 0. Energy at t=0 is all potential: m·g·h0.",
    ),
    "free_fall_v0": PhysicsScenario(
        name="free_fall_v0",
        description="Ball thrown with initial velocity v0",
        params=["m", "g", "h0", "v0", "t"],
        conserved_expr="m * g * h + (1/2) * m * v ^ 2",
        invariant_rhs="m * g * h0 + (1/2) * m * v0 ^ 2",
        kinematic_subs={
            "v": "v0 - g * t",
            "h": "h0 + v0 * t - (1/2) * g * t ^ 2",
        },
        proof_tactic="ring",
        notes="General free-fall with initial velocity (upward positive convention).",
    ),
    "projectile": PhysicsScenario(
        name="projectile",
        description="Projectile launched at angle θ — total mechanical energy",
        params=["m", "g", "h0", "v0", "theta", "t"],
        conserved_expr=(
            "(1/2) * m * (vx ^ 2 + vy ^ 2) + m * g * y"
        ),
        invariant_rhs="(1/2) * m * v0 ^ 2 + m * g * h0",
        kinematic_subs={
            "vx": "v0 * cos theta",
            "vy": "v0 * sin theta - g * t",
            "y": "h0 + v0 * sin theta * t - (1/2) * g * t ^ 2",
        },
        proof_tactic="special_projectile",
        extra_lemmas=[],
        notes=(
            "Total mechanical energy = KE_x + KE_y + PE. "
            "vx constant (no horizontal force); vy and y follow free-fall. "
            "Proof: ring_nf + factor sin²θ+cos²θ=1 + linarith."
        ),
    ),
    "pendulum": PhysicsScenario(
        name="pendulum",
        description="Simple pendulum — energy conservation",
        params=["m", "g", "L", "theta", "theta0"],
        conserved_expr=(
            "m * g * L * (1 - cos theta) + "
            "m * g * L * (cos theta - cos theta0)"
        ),
        invariant_rhs="m * g * L * (1 - cos theta0)",
        kinematic_subs={},  # Already expresses v² term directly
        proof_tactic="ring",
        notes=(
            "h = L·(1-cos θ), ½·m·v² = m·g·L·(cos θ - cos θ₀). "
            "v² = 2·g·L·(cos θ - cos θ₀) from energy conservation itself; "
            "this proof verifies self-consistency without sqrt."
        ),
    ),
    "spring": PhysicsScenario(
        name="spring",
        description="Simple harmonic oscillator — energy expression (placeholder)",
        params=["m", "k", "x", "v"],
        conserved_expr="(1/2) * m * v ^ 2 + (1/2) * k * x ^ 2",
        invariant_rhs="(1/2) * m * v ^ 2 + (1/2) * k * x ^ 2",
        kinematic_subs={},  # No time dependence — identity by rfl
        proof_tactic="rfl",
        notes=(
            "Trivially true: expression equals itself. "
            "A full conservation proof requires d/dt analysis (m·v·dv/dt + k·x·dx/dt = 0 "
            "given m·dv/dt = -k·x), which is beyond ring. "
            "See spring_trig for a trigonometric proof with ω² = k/m."
        ),
    ),
    "spring_trig": PhysicsScenario(
        name="spring_trig",
        description="Spring energy conservation via trigonometric solution",
        params=["m", "k", "A", "omega", "t"],
        conserved_expr=(
            "(1/2) * m * (A * omega * cos (omega * t)) ^ 2 + "
            "(1/2) * k * (A * sin (omega * t)) ^ 2"
        ),
        invariant_rhs="(1/2) * k * A ^ 2",
        kinematic_subs={},  # Already substituted in the expression
        proof_tactic="special_spring",  # Handled by generator
        extra_lemmas=["Real.cos_sq_add_sin_sq"],
        notes=(
            "x = A·sin(ωt), v = A·ω·cos(ωt). "
            "Requires hypothesis ω² = k/m. Uses sin² + cos² = 1."
        ),
    ),
    "work_energy": PhysicsScenario(
        name="work_energy",
        description="Work-energy theorem: ΔKE = W",
        params=["m", "g", "v0", "v", "delta_h"],
        conserved_expr="(1/2) * m * (v ^ 2 - v0 ^ 2)",
        invariant_rhs="m * g * delta_h",
        kinematic_subs={},  # Uses hypothesis v² = v0² + 2·g·Δh
        proof_tactic="special_work_energy",
        notes="Given kinematic relation v² = v0² + 2·g·Δh, prove ΔKE = m·g·Δh.",
    ),
}


# ── Theorem generation ─────────────────────────────────────────────────────────


def _params_to_lean(params: list[str]) -> str:
    """Convert param names to Lean binder syntax.

    >>> _params_to_lean(["m", "g", "h0", "t"])
    '(m : ℝ) (g : ℝ) (h0 : ℝ) (t : ℝ)'
    """
    return " ".join(f"({p} : ℝ)" for p in params)


def _sanitize_scenario_name(name: str) -> str:
    """Convert scenario name to valid Lean identifier."""
    return "".join(c if c.isalnum() else "_" for c in name)


def generate_theorem(scenario_name: str) -> LeanTheorem:
    """Generate a Lean theorem for a named scenario.

    Args:
        scenario_name: Key into SCENARIOS dict.

    Returns:
        LeanTheorem ready for verification or file writing.
    """
    sc = SCENARIOS[scenario_name]
    thm_name = f"energy_conservation_{_sanitize_scenario_name(scenario_name)}"

    # Build the theorem statement
    params_str = _params_to_lean(sc.params)

    # Substitute variables
    lhs = _substitute_vars(sc.conserved_expr, sc.kinematic_subs)
    rhs = sc.invariant_rhs

    statement = f"{params_str} :\n    {lhs} = {rhs}"

    # Build the proof block based on the tactic
    tactic = sc.proof_tactic
    extra = ""

    if tactic == "special_spring":
        # Spring: needs hypothesis omega^2 = k/m
        # We embed this as a hypothesis in the theorem
        statement = (
            f"{params_str}\n"
            f"    (h_omega_sq : k = m * omega ^ 2) :\n"
            f"    {lhs} = {rhs}"
        )
        proof_block = (
            "rw [h_omega_sq]\n"
            "  have hid : (cos (omega * t)) ^ 2 + (sin (omega * t)) ^ 2 = 1 := by\n"
            "    rw [Real.cos_sq_add_sin_sq]\n"
            "  calc\n"
            f"    {lhs.replace('k', '(m * omega ^ 2)')}\n"
            "        = (1/2 : ℝ) * m * omega ^ 2 * A ^ 2 * "
            "((cos (omega * t)) ^ 2 + (sin (omega * t)) ^ 2) := by ring\n"
            "    _ = (1/2 : ℝ) * m * omega ^ 2 * A ^ 2 * 1 := by rw [hid]\n"
            "    _ = (1/2 : ℝ) * (m * omega ^ 2) * A ^ 2 := by ring"
        )
    elif tactic == "special_work_energy":
        statement = (
            f"{params_str}\n"
            f"    (h_v_sq : v ^ 2 = v0 ^ 2 + 2 * g * delta_h) :\n"
            f"    {lhs} = {rhs}"
        )
        proof_block = "rw [h_v_sq]\n  ring"
    elif tactic == "special_projectile":
        proof_block = (
            "ring_nf\n"
            "  have hsq := Real.sin_sq_add_cos_sq theta\n"
            "  have h_factor : m * v0 ^ 2 * sin theta ^ 2 * (1/2 : ℝ) + m * v0 ^ 2 * cos theta ^ 2 * (1/2 : ℝ) = m * v0 ^ 2 * (1/2 : ℝ) := by\n"
            "    calc\n"
            "      m * v0 ^ 2 * sin theta ^ 2 * (1/2 : ℝ) + m * v0 ^ 2 * cos theta ^ 2 * (1/2 : ℝ)\n"
            "          = m * v0 ^ 2 * (1/2 : ℝ) * (sin theta ^ 2 + cos theta ^ 2) := by ring_nf\n"
            "      _ = m * v0 ^ 2 * (1/2 : ℝ) * 1 := by rw [hsq]\n"
            "      _ = m * v0 ^ 2 * (1/2 : ℝ) := by ring\n"
            "  linarith"
        )
    elif tactic == "field_simp; ring":
        # Pendulum with square root: need to square both sides or use sqrt elimination
        # Since sqrt appears in the expression, we use a different approach
        proof_block = (
            "have h_sq : (sqrt (2 * g / L * (cos theta - cos theta0))) ^ 2 = "
            "2 * g / L * (cos theta - cos theta0) := by\n"
            "    rw [Real.sq_sqrt (by\n"
            "      have : 0 ≤ 2 * g / L * (cos theta - cos theta0) := by\n"
            "        -- For a pendulum, cos θ ≤ cos θ₀ when |θ| ≥ |θ₀| (speed is real)\n"
            "        -- We assume the physically valid regime\n"
            "        positivity\n"
            "      exact this)]\n"
            "  -- This requires positivity which may not always hold\n"
            "  sorry"
        )
        # Fallback: use a simpler approach — just state the identity without sqrt
        # Actually, the pendulum scenario with sqrt is complex; skip the sqrt version
        # and use the direct substitution from the scenario definition
        proof_block = "ring"
    elif tactic == "rfl":
        # Trivial conservation (same expression on both sides)
        proof_block = "rfl"
    else:
        proof_block = tactic

    # Extra lemmas — only for non-special tactics (special_* handle their own)
    if sc.extra_lemmas and not tactic.startswith("special_"):
        extra = "\n".join(
            f"  have hid := {lemma}" for lemma in sc.extra_lemmas
        )

    return LeanTheorem(
        name=thm_name,
        scenario=scenario_name,
        statement=statement,
        proof_block=proof_block,
        extra_code=extra,
    )


def generate_all_theorems() -> list[LeanTheorem]:
    """Generate Lean theorems for all registered scenarios.

    Returns:
        List of LeanTheorem, one per scenario.
    """
    return [generate_theorem(name) for name in SCENARIOS]


# ── File I/O ───────────────────────────────────────────────────────────────────


def write_lean_file(theorems: list[LeanTheorem], path: Path) -> Path:
    """Write theorems to a .lean file with Mathlib preamble.

    Args:
        theorems: List of LeanTheorem to write.
        path: Output file path (will be overwritten).

    Returns:
        The output path.
    """
    parts = [LEAN_PREAMBLE_MATHLIB.strip()]
    for thm in theorems:
        parts.append(thm.lean_code)
    content = "\n\n".join(parts) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _find_lake_project() -> Path | None:
    """Locate the proof_checker_env Lake project directory."""
    from src.proof_checker.lean_interface import _find_project_dir
    return _find_project_dir()


def verified_theorems_dir() -> Path:
    """Return default output directory for saved verified theorems."""
    candidate = Path(__file__).resolve().parent.parent.parent / "data" / "verified_theorems"
    return candidate


def save_verified_theorem(thm: LeanTheorem, output_dir: Path | None = None) -> Path:
    """Save a single verified theorem to a .lean file.

    Args:
        thm: The verified LeanTheorem.
        output_dir: Directory to save to. Defaults to verified_theorems_dir().

    Returns:
        Path to the saved file.
    """
    out = output_dir or verified_theorems_dir()
    out.mkdir(parents=True, exist_ok=True)
    fname = f"{thm.name}.lean"
    path = out / fname
    return write_lean_file([thm], path)


# ── Proof verification ─────────────────────────────────────────────────────────


def verify_theorem(thm: LeanTheorem, timeout: float = 15.0) -> tuple[bool, str]:
    """Verify a LeanTheorem against the Lean 4 proof checker.

    Args:
        thm: The theorem to verify.
        timeout: Seconds before timeout.

    Returns:
        Tuple of (success: bool, output: str).
        output contains the proof result details or error text.
    """
    try:
        checker = LeanProofChecker(timeout=timeout)
        result = checker.check(thm.lean_code)
        if result.success:
            return True, f"Verified in {result.check_time_ms:.0f}ms"
        else:
            error_text = "; ".join(result.errors) if result.errors else "unknown error"
            return False, error_text
    except Exception as exc:
        return False, str(exc)


def verify_scenario(scenario_name: str, timeout: float = 15.0) -> tuple[bool, str, LeanTheorem]:
    """Generate and verify a theorem for a single scenario.

    Args:
        scenario_name: Key into SCENARIOS dict.
        timeout: Seconds before Lean timeout.

    Returns:
        Tuple of (success: bool, output: str, theorem: LeanTheorem).
    """
    thm = generate_theorem(scenario_name)
    success, output = verify_theorem(thm, timeout)
    return success, output, thm


def verify_all(
    scenario_names: list[str] | None = None,
    timeout: float = 15.0,
) -> dict[str, tuple[bool, str, LeanTheorem]]:
    """Verify multiple scenarios.

    Args:
        scenario_names: List of scenario names to verify. None = all.
        timeout: Seconds per scenario.

    Returns:
        Dict mapping scenario_name → (success, output, theorem).
    """
    if scenario_names is None:
        scenario_names = list(SCENARIOS)
    results: dict[str, tuple[bool, str, LeanTheorem]] = {}
    for name in scenario_names:
        results[name] = verify_scenario(name, timeout)
    return results
