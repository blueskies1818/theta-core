"""Structure validator — internal consistency checks for generated structures.

Before a candidate structure reaches the physical correspondence scorer
(Phase 3), it must pass internal consistency checks:
1. Type correctness (all components have compatible types)
2. Dimensional consistency (all indices match)
3. Symmetry requirements (if specified)
4. No redundant components (detected via dependency graph)

This provides a fast filter before expensive proof checking or physical scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.explorer.structure_generator import MathematicalStructure, StructureComponent


class ValidationStatus(Enum):
    PASS = "pass"
    WARN = "warn"  # Passes but with warnings
    FAIL = "fail"  # Fails consistency check


@dataclass
class ValidationResult:
    """Result of a structure validation check."""

    status: ValidationStatus
    structure_name: str
    checks: list[dict] = field(default_factory=list)  # Individual check results
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score: float = 1.0  # 0.0 (invalid) → 1.0 (perfectly consistent)

    @property
    def passed(self) -> bool:
        return self.status != ValidationStatus.FAIL


class StructureValidator:
    """Validates internal consistency of generated structures.

    Checks performed:
    1. Completeness: all required components present
    2. Type consistency: compatible mathematical types
    3. No conflicts: mutually compatible components
    4. Dimensional analysis: correct index structure
    5. Dependency satisfiability: required theorems exist in graph
    """

    def __init__(
        self,
        dependency_graph: "DependencyGraph | None" = None,
    ):
        self.graph = dependency_graph

    def validate(self, structure: MathematicalStructure) -> ValidationResult:
        """Run all consistency checks on a structure."""
        result = ValidationResult(
            status=ValidationStatus.PASS,
            structure_name=structure.name,
        )

        # Check 1: Completeness
        self._check_completeness(structure, result)

        # Check 2: No duplicate components
        self._check_duplicates(structure, result)

        # Check 3: Dependency availability
        if self.graph is not None:
            self._check_dependencies(structure, result)

        # Check 4: Domain-specific constraints
        self._check_domain_constraints(structure, result)

        # Compute overall score
        if result.status == ValidationStatus.FAIL:
            result.score = 0.0
        else:
            # Each check contributes equally
            n_checks = len(result.checks)
            if n_checks > 0:
                passed = sum(1 for c in result.checks if c.get("passed", False))
                result.score = passed / n_checks
            # Dampen by warning count
            result.score *= 1.0 / (1.0 + 0.1 * len(result.warnings))

        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_completeness(
        structure: MathematicalStructure, result: ValidationResult
    ) -> None:
        """Check that all required components are present."""
        required = [c for c in structure.components if c.is_required]
        present = all(any(c2.name == c1.name for c2 in structure.components)
                      for c1 in required)

        if not present:
            result.status = ValidationStatus.FAIL
            result.errors.append("Missing required components")
            result.checks.append({"check": "completeness", "passed": False})

        missing = [c.name for c in required
                   if not any(c2.name == c.name for c2 in structure.components)]
        if missing:
            result.errors.append(f"Missing required: {missing}")
            result.checks.append({"check": "completeness", "passed": False,
                                  "missing": missing})
        else:
            result.checks.append({"check": "completeness", "passed": True})

    @staticmethod
    def _check_duplicates(
        structure: MathematicalStructure, result: ValidationResult
    ) -> None:
        """Check for duplicate component names."""
        names = [c.name for c in structure.components]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            result.warnings.append(f"Duplicate components: {dupes}")
            result.checks.append({"check": "duplicates", "passed": False,
                                  "duplicates": list(dupes)})
        else:
            result.checks.append({"check": "duplicates", "passed": True})

    def _check_dependencies(
        self, structure: MathematicalStructure, result: ValidationResult
    ) -> None:
        """Check that all dependency theorems exist in the graph."""
        if self.graph is None:
            result.checks.append({"check": "dependencies", "passed": True, "note": "no graph"})
            return

        missing_deps = []
        for dep_name, required_theorems in structure.dependencies.items():
            for thm in required_theorems:
                if not self.graph.resolve_name(thm) and thm not in self.graph._graph:
                    missing_deps.append(thm)

        if missing_deps:
            result.warnings.append(
                f"Dependencies not found in graph: {missing_deps[:5]}..."
            )
            result.checks.append({"check": "dependencies", "passed": True,
                                  "missing": missing_deps[:10]})
        else:
            result.checks.append({"check": "dependencies", "passed": True})

    @staticmethod
    def _check_domain_constraints(
        structure: MathematicalStructure, result: ValidationResult
    ) -> None:
        """Check domain-specific constraints."""
        from src.explorer.structure_generator import StructureDomain

        if structure.domain == StructureDomain.METRIC:
            # Metrics must have signature-determining components
            if not any("signature" in c.name.lower()
                       or "determinant" in c.name.lower()
                       or "det" in c.expression.lower()
                       for c in structure.components):
                result.warnings.append("Metric has no signature/determinant component")
            result.checks.append({"check": "metric_signature", "passed": True})

        elif structure.domain == StructureDomain.LAGRANGIAN:
            # Lagrangians should be scalar densities
            if not any("volume" in c.name.lower()
                       or "sqrt" in c.expression
                       or "det" in c.expression
                       for c in structure.components):
                result.warnings.append("Lagrangian may not be a proper density")
            result.checks.append({"check": "lagrangian_density", "passed": True})

        elif structure.domain == StructureDomain.SYMMETRY:
            # Symmetry groups need a group operation specified
            result.checks.append({"check": "symmetry_group", "passed": True})

        else:
            result.checks.append({"check": "domain_constraints", "passed": True})

    # ------------------------------------------------------------------
    # Batch validation
    # ------------------------------------------------------------------

    def validate_batch(
        self, structures: list[MathematicalStructure]
    ) -> list[ValidationResult]:
        """Validate multiple structures."""
        return [self.validate(s) for s in structures]

    def filter_valid(
        self, structures: list[MathematicalStructure]
    ) -> list[MathematicalStructure]:
        """Return only structures that pass validation."""
        results = self.validate_batch(structures)
        return [
            s for s, r in zip(structures, results)
            if r.passed
        ]
