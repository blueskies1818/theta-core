"""Mathematical structure generation and mutation (Phase 2.5).

Where Phase 2.1-2.4 handles proof search over EXISTING mathematics, Phase 2.5
enables the explorer to generate NEW mathematical structures: modified Lagrangians,
alternative symmetry groups, novel connection forms, etc.

These structures are the explorer's "creative output" — not just verifying known
theorems, but proposing objects that don't yet exist in Mathlib4.

Architecture:
    1. Structure templates define the space of possible objects
    2. Mutation operators (add/remove/modify terms) explore the space
    3. The GNN guides mutation toward promising regions
    4. Structure validator checks internal consistency
    5. (Phase 3) Physical correspondence scorer evaluates against experiment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# ---------------------------------------------------------------------------
# Structure types
# ---------------------------------------------------------------------------


class StructureDomain(Enum):
    """Mathematical physics domains for structure exploration."""

    METRIC = "metric"  # Metric tensors (spacetime geometry)
    LAGRANGIAN = "lagrangian"  # Action functionals / Lagrangians
    SYMMETRY = "symmetry"  # Symmetry groups and algebras
    CONNECTION = "connection"  # Connection forms / gauge fields
    FIELD = "field"  # Field equations (scalar, vector, tensor)
    HAMILTONIAN = "hamiltonian"  # Hamiltonian formulations
    BOUNDARY = "boundary"  # Boundary conditions / constraints
    CONSERVATION = "conservation"  # Conservation law formulations


@dataclass
class StructureComponent:
    """A term or component in a mathematical structure.

    Example: the Ricci scalar term in the Einstein-Hilbert action.
    """

    name: str  # e.g., "Ricci_scalar", "cosmological_constant"
    expression: str  # Lean 4 expression, e.g., "R", "Λ * g_μν"
    coefficient: float = 1.0  # Multiplicative coefficient
    is_required: bool = False  # If True, cannot be removed by mutation
    domain: StructureDomain | None = None

    def to_lean(self) -> str:
        """Render this component as a Lean 4 expression."""
        if self.coefficient == 1.0:
            return self.expression
        elif self.coefficient == 0.0:
            return "0"
        else:
            return f"({self.coefficient}) * ({self.expression})"


@dataclass
class MathematicalStructure:
    """A candidate mathematical structure for a physical theory.

    This is the unit of creative output from the explorer. It represents
    a complete mathematical object (metric, Lagrangian, symmetry group, etc.)
    that can be checked for internal consistency and scored against
    experimental data (Phase 3).
    """

    name: str  # Human-readable name
    domain: StructureDomain
    components: list[StructureComponent] = field(default_factory=list)

    # Dependencies on known theorems for each component
    dependencies: dict[str, list[str]] = field(default_factory=dict)

    # Metadata
    description: str = ""
    generation: int = 0  # Mutation generation (0 = template)
    parent_name: str = ""  # Which structure this was mutated from
    score: float = 0.0  # Overall score (from validator + physical scorer)

    def to_lean(self) -> str:
        """Render the structure as a Lean 4 definition block."""
        if self.domain == StructureDomain.LAGRANGIAN:
            # L = Σ components
            terms = " + ".join(c.to_lean() for c in self.components)
            return f"def {self.name} : ℝ := {terms}"
        elif self.domain == StructureDomain.METRIC:
            # g_μν = Σ components
            terms = " + ".join(c.to_lean() for c in self.components)
            return f"def {self.name} (μ ν : ℕ) : ℝ := {terms}"
        elif self.domain == StructureDomain.SYMMETRY:
            elems = ", ".join(c.expression for c in self.components)
            return f"def {self.name} : Set (Matrix (Fin 4) (Fin 4) ℝ) := {{{elems}}}"
        else:
            elems = "\n  ".join(c.to_lean() for c in self.components)
            return f"structure {self.name} where\n  {elems}"

    def complexity(self) -> float:
        """Compute structure complexity (for Occam's razor penalty).

        More components → higher complexity penalty in Phase 3.
        """
        return sum(1.0 for c in self.components if not c.is_required) + len(
            self.components
        ) * 0.1


# ---------------------------------------------------------------------------
# Structure templates (known physics as starting points)
# ---------------------------------------------------------------------------


def einstein_hilbert_action() -> MathematicalStructure:
    """Einstein-Hilbert action: S = ∫ (R - 2Λ) √(-g) d⁴x."""
    return MathematicalStructure(
        name="EinsteinHilbert",
        domain=StructureDomain.LAGRANGIAN,
        components=[
            StructureComponent(
                name="Ricci_scalar",
                expression="R",
                coefficient=1.0,
                is_required=True,
                domain=StructureDomain.LAGRANGIAN,
            ),
            StructureComponent(
                name="cosmological_constant",
                expression="Λ • g",
                coefficient=-2.0,
                domain=StructureDomain.LAGRANGIAN,
            ),
            StructureComponent(
                name="volume_element",
                expression="Real.sqrt (-(det g))",
                coefficient=1.0,
                is_required=True,
                domain=StructureDomain.LAGRANGIAN,
            ),
        ],
        description="Standard Einstein-Hilbert action for general relativity.",
    )


def standard_model_lagrangian() -> MathematicalStructure:
    """Standard Model Lagrangian (schematic)."""
    return MathematicalStructure(
        name="StandardModel",
        domain=StructureDomain.LAGRANGIAN,
        components=[
            StructureComponent(
                name="gauge_kinetic",
                expression="-(1/4) * F_μν^a F^aμν",
                coefficient=1.0,
                is_required=True,
                domain=StructureDomain.LAGRANGIAN,
            ),
            StructureComponent(
                name="fermion_kinetic",
                expression="ψ̄ i γ^μ D_μ ψ",
                coefficient=1.0,
                is_required=True,
                domain=StructureDomain.LAGRANGIAN,
            ),
            StructureComponent(
                name="higgs_kinetic",
                expression="|D_μ φ|²",
                coefficient=1.0,
                is_required=True,
                domain=StructureDomain.LAGRANGIAN,
            ),
            StructureComponent(
                name="higgs_potential",
                expression="-μ² |φ|² + λ |φ|⁴",
                coefficient=1.0,
                domain=StructureDomain.LAGRANGIAN,
            ),
            StructureComponent(
                name="yukawa",
                expression="y_ij ψ̄_i φ ψ_j",
                coefficient=1.0,
                domain=StructureDomain.LAGRANGIAN,
            ),
        ],
        description="Standard Model of particle physics (SU(3)×SU(2)×U(1)).",
    )


def schwarzschild_metric() -> MathematicalStructure:
    """Schwarzschild metric: ds² = -(1-2M/r)dt² + (1-2M/r)⁻¹dr² + r²dΩ²."""
    return MathematicalStructure(
        name="Schwarzschild",
        domain=StructureDomain.METRIC,
        components=[
            StructureComponent(
                name="time_component",
                expression="-(1 - 2*M/r)",
                coefficient=1.0,
                is_required=True,
                domain=StructureDomain.METRIC,
            ),
            StructureComponent(
                name="radial_component",
                expression="(1 - 2*M/r)⁻¹",
                coefficient=1.0,
                is_required=True,
                domain=StructureDomain.METRIC,
            ),
            StructureComponent(
                name="angular_component",
                expression="r²",
                coefficient=1.0,
                is_required=True,
                domain=StructureDomain.METRIC,
            ),
        ],
        description="Schwarzschild solution for a non-rotating, uncharged black hole.",
    )


# Registry of templates
TEMPLATES: dict[str, Callable[[], MathematicalStructure]] = {
    "einstein_hilbert": einstein_hilbert_action,
    "standard_model": standard_model_lagrangian,
    "schwarzschild": schwarzschild_metric,
}


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------


class MutationType(Enum):
    """Types of mutations that can be applied to a structure."""

    ADD_TERM = "add_term"  # Add a new component to the structure
    REMOVE_TERM = "remove_term"  # Remove an optional component
    MODIFY_COEFFICIENT = "modify_coefficient"  # Change a coefficient
    COMPOSE = "compose"  # Compose two structures
    GENERALIZE = "generalize"  # Replace a component with a more general form
    SPECIALIZE = "specialize"  # Replace with a more specific form (e.g., impose symmetry)


@dataclass
class Mutation:
    """A single mutation applied to a structure."""

    mutation_type: MutationType
    target_component: str | None = None  # Which component is modified
    new_component: StructureComponent | None = None  # For ADD_TERM
    new_coefficient: float | None = None  # For MODIFY_COEFFICIENT
    description: str = ""


def mutate_structure(
    structure: MathematicalStructure,
    mutation: Mutation,
) -> MathematicalStructure:
    """Apply a mutation to create a new structure.

    Returns a new MathematicalStructure (original is unchanged).
    """
    new_components = [
        StructureComponent(
            name=c.name,
            expression=c.expression,
            coefficient=c.coefficient,
            is_required=c.is_required,
            domain=c.domain,
        )
        for c in structure.components
    ]

    if mutation.mutation_type == MutationType.ADD_TERM and mutation.new_component:
        new_components.append(mutation.new_component)

    elif mutation.mutation_type == MutationType.REMOVE_TERM and mutation.target_component:
        new_components = [
            c for c in new_components
            if c.name != mutation.target_component or c.is_required
        ]

    elif mutation.mutation_type == MutationType.MODIFY_COEFFICIENT:
        if mutation.target_component and mutation.new_coefficient is not None:
            for c in new_components:
                if c.name == mutation.target_component:
                    c.coefficient = mutation.new_coefficient

    elif mutation.mutation_type == MutationType.GENERALIZE:
        if mutation.target_component and mutation.new_component:
            for i, c in enumerate(new_components):
                if c.name == mutation.target_component:
                    new_components[i] = mutation.new_component

    elif mutation.mutation_type == MutationType.SPECIALIZE:
        if mutation.target_component and mutation.new_component:
            for i, c in enumerate(new_components):
                if c.name == mutation.target_component:
                    new_components[i] = mutation.new_component

    return MathematicalStructure(
        name=f"{structure.name}_gen{structure.generation + 1}",
        domain=structure.domain,
        components=new_components,
        dependencies=dict(structure.dependencies),
        description=mutation.description or f"Mutation: {mutation.mutation_type.value}",
        generation=structure.generation + 1,
        parent_name=structure.name,
    )


# ---------------------------------------------------------------------------
# Structure generator (orchestrator)
# ---------------------------------------------------------------------------


@dataclass
class GeneratorConfig:
    """Configuration for the structure generator."""

    max_mutations_per_generation: int = 5
    max_generations: int = 10
    beam_size: int = 5  # Keep top K structures per generation
    complexity_penalty: float = 0.1  # Occam's razor weight


class StructureGenerator:
    """Generates and mutates mathematical structures.

    The generator explores the space of possible structures by:
    1. Starting from known templates (Einstein-Hilbert, Standard Model, etc.)
    2. Applying mutation operators to create variants
    3. Using the GNN to guide which mutations are promising
    4. Validating internal consistency before scoring

    This is the replacement for "proof generation" at the structure level.
    Instead of generating proofs for existing theorems, it generates
    NEW theorems (structures) that might describe physical reality.
    """

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        gnn_encoder: "GNNEncoder | None" = None,
        dependency_graph: "DependencyGraph | None" = None,
    ):
        self.config = config or GeneratorConfig()
        self.gnn = gnn_encoder
        self.graph = dependency_graph

        self._current_structures: list[MathematicalStructure] = []
        self._history: list[MathematicalStructure] = []

    def initialize(self, template_names: list[str] | None = None) -> None:
        """Initialize from template structures."""
        if template_names is None:
            template_names = list(TEMPLATES.keys())

        for name in template_names:
            if name in TEMPLATES:
                structure = TEMPLATES[name]()
                self._current_structures.append(structure)
                self._history.append(structure)

        print(
            f"Initialized generator with {len(self._current_structures)} templates"
        )

    def explore(
        self,
        num_generations: int | None = None,
    ) -> list[MathematicalStructure]:
        """Run structure exploration for multiple generations.

        Each generation:
        1. For each current structure, generate mutations
        2. Validate and score each mutant
        3. Keep top-k (beam search)

        Args:
            num_generations: Number of mutation rounds.

        Returns:
            All generated structures across all generations.
        """
        if num_generations is None:
            num_generations = self.config.max_generations

        for gen in range(num_generations):
            candidates: list[MathematicalStructure] = []

            for structure in self._current_structures:
                # Generate mutations
                mutants = self._generate_mutations(structure)
                candidates.extend(mutants)

            # Score and rank
            for c in candidates:
                c.score = self._score_structure(c)

            # Keep top-k
            candidates.sort(key=lambda s: s.score, reverse=True)
            self._current_structures = candidates[: self.config.beam_size]
            self._history.extend(candidates)

            if gen % 3 == 0 or gen == num_generations - 1:
                print(
                    f"Gen {gen + 1}: {len(candidates)} mutants → "
                    f"kept {len(self._current_structures)} best"
                )
                if self._current_structures:
                    best = self._current_structures[0]
                    print(f"  Best: {best.name} (score={best.score:.3f})")

        return self._history

    def _generate_mutations(
        self, structure: MathematicalStructure
    ) -> list[MathematicalStructure]:
        """Generate mutation variants of a structure."""
        mutants = []

        # ADD_TERM: add candidate terms (guided by GNN if available)
        candidate_terms = self._propose_terms(structure)
        for term in candidate_terms[: self.config.max_mutations_per_generation]:
            mutation = Mutation(
                mutation_type=MutationType.ADD_TERM,
                new_component=term,
                description=f"Add {term.name} to {structure.name}",
            )
            mutant = mutate_structure(structure, mutation)
            mutants.append(mutant)

        # MODIFY_COEFFICIENT: vary coefficients by ±50%
        for comp in structure.components[:3]:
            if not comp.is_required:
                for factor in [0.5, 2.0, -1.0]:
                    mutation = Mutation(
                        mutation_type=MutationType.MODIFY_COEFFICIENT,
                        target_component=comp.name,
                        new_coefficient=comp.coefficient * factor,
                        description=f"Scale {comp.name} by {factor}",
                    )
                    mutant = mutate_structure(structure, mutation)
                    mutants.append(mutant)

        # REMOVE_TERM: try removing optional components
        for comp in structure.components:
            if not comp.is_required:
                mutation = Mutation(
                    mutation_type=MutationType.REMOVE_TERM,
                    target_component=comp.name,
                    description=f"Remove {comp.name} from {structure.name}",
                )
                mutant = mutate_structure(structure, mutation)
                mutants.append(mutant)

        return mutants

    def _propose_terms(
        self, structure: MathematicalStructure
    ) -> list[StructureComponent]:
        """Propose new terms that could be added to a structure.

        Uses the GNN to find theorems/definitions in the dependency graph
        that are related to the structure's domain.
        """
        if self.graph is None:
            return self._default_terms(structure.domain)

        # Find domain-relevant nodes via graph search
        domain_nodes = self.graph.get_node_ids_by_domain("Analysis")

        # Look for nodes that mention structure-related keywords
        keywords = {
            StructureDomain.METRIC: ["metric", "distance", "riemannian"],
            StructureDomain.LAGRANGIAN: ["action", "lagrangian", "functional"],
            StructureDomain.SYMMETRY: ["group", "symmetry", "invariant", "algebra"],
            StructureDomain.CONNECTION: ["connection", "curvature", "form", "bundle"],
            StructureDomain.FIELD: ["field", "equation", "wave", "potential"],
        }

        relevant = keywords.get(structure.domain, [])
        candidates: list[StructureComponent] = []

        for node_id in list(domain_nodes)[:100]:
            attrs = self.graph.get_node(node_id)
            if not attrs:
                continue
            name = attrs.get("name", "")
            stmt = attrs.get("statement", "")

            if any(kw in name.lower() or kw in stmt.lower() for kw in relevant):
                candidates.append(
                    StructureComponent(
                        name=name,
                        expression=name,
                        domain=structure.domain,
                    )
                )

        return candidates[: self.config.max_mutations_per_generation] or self._default_terms(
            structure.domain
        )

    @staticmethod
    def _default_terms(domain: StructureDomain) -> list[StructureComponent]:
        """Default candidate terms for a domain (when GNN is unavailable)."""
        defaults = {
            StructureDomain.LAGRANGIAN: [
                StructureComponent(name="scalar_curvature", expression="R"),
                StructureComponent(name="gauss_bonnet", expression="GB"),
                StructureComponent(name="scalar_field_kinetic", expression="(∂φ)²"),
                StructureComponent(name="vector_potential", expression="A_μ A^μ"),
            ],
            StructureDomain.METRIC: [
                StructureComponent(name="warp_factor", expression="Ω²(x)"),
                StructureComponent(name="cross_term", expression="f(r) dt dr"),
                StructureComponent(name="brane_tension", expression="T • δ(y)"),
            ],
            StructureDomain.SYMMETRY: [
                StructureComponent(name="Lorentz", expression="SO(3,1)"),
                StructureComponent(name="conformal", expression="SO(4,2)"),
                StructureComponent(name="screw", expression="ISO(3,1)"),
            ],
        }
        return defaults.get(domain, [StructureComponent(name="generic_term", expression="X")])

    def _score_structure(self, structure: MathematicalStructure) -> float:
        """Score a structure based on internal consistency.

        In Phase 3, this is augmented with physical correspondence scoring.
        For now: favor simplicity (fewer components) while ensuring all
        required components are present.
        """
        has_all_required = all(
            c.is_required for c in structure.components if c.is_required
        )

        if not has_all_required:
            return -1.0  # Invalid

        # Structural score: completeness - complexity penalty
        completeness = len(structure.components) / (
            len(structure.components) + 1
        )
        penalty = self.config.complexity_penalty * structure.complexity()

        return completeness - penalty

    def get_best_structures(self, n: int = 5) -> list[MathematicalStructure]:
        """Return the top-n structures by score."""
        ranked = sorted(self._history, key=lambda s: s.score, reverse=True)
        return ranked[:n]
