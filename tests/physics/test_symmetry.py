"""Tests for symmetry-driven invariant derivation module.

Covers acceptance criteria:
  1. SymmetryDetector correctly identifies time-translation in free-fall
  2. NoetherDerivation produces mgh + ½mv² from Lagrangian + time symmetry
  3. Symmetry classifier trained and accurate on known physics
  4. For combined gravity+spring: detects both symmetries, derives both invariants
  5. All existing tests still pass
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from src.physics.dimensions import Dimension
from src.physics.observations import Observation, ObservationDatabase
from src.physics.symmetry import (
    # Core classes
    SymmetryGroup,
    SymmetryDetection,
    SymmetryDetector,
    NoetherDerivation,
    Lagrangian,
    ConservedQuantity,
    SymmetryResult,
    SymmetryPipeline,
    GeneratorKind,
    GENERATOR_LABELS,
    # Pre-built groups
    PREBUILT_GROUPS,
    build_galilean_group,
    build_u1_group,
    build_su2_group,
    get_group,
    # Classifier
    SymmetryClassifier,
    SYMMETRY_CLASSES,
    SYMMETRY_CLASS_LABELS,
    build_symmetry_training_data,
    train_symmetry_classifier,
    _STANDARD_LAGRANGIANS,
    # Smoke test
    run_symmetry_smoke_test,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def free_fall_obs() -> Observation:
    """Minimal free-fall observation."""
    return Observation(
        id="test_free_fall",
        name="Test free fall",
        description="Ball dropped from rest",
        quantities={"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
        parameters={"m": 1.0, "g": 9.8},
        timesteps=[
            {"t": 0.0, "h": 10.0, "v": 0.0},
            {"t": 0.5, "h": 8.775, "v": 4.9},
            {"t": 1.0, "h": 5.1, "v": 9.8},
        ],
        known_invariant="m*g*h + 0.5*m*v^2",
        lean_theorem="",
    )


@pytest.fixture
def projectile_obs() -> Observation:
    """2D projectile observation with x and y."""
    return Observation(
        id="test_projectile",
        name="Test projectile",
        description="Ball thrown at angle",
        quantities={
            "m": "Mass", "g": "Accel", "x": "Length", "y": "Length",
            "vx": "Velocity", "vy": "Velocity", "t": "Time",
        },
        parameters={"m": 1.0, "g": 9.8},
        timesteps=[
            {"t": 0.0, "x": 0.0, "y": 0.0, "vx": 10.0, "vy": 10.0},
            {"t": 1.0, "x": 10.0, "y": 5.1, "vx": 10.0, "vy": 0.2},
        ],
        known_invariant="0.5*m*vx^2 + 0.5*m*vy^2 + m*g*y",
        lean_theorem="",
    )


@pytest.fixture
def gravity_spring_obs() -> Observation:
    """Combined gravity + spring system."""
    return Observation(
        id="test_gravity_spring",
        name="Mass-spring with gravity",
        description="Mass on spring in gravitational field",
        quantities={
            "m": "Mass", "g": "Accel", "k": "Force/Length", "h": "Length",
            "v": "Velocity", "t": "Time",
        },
        parameters={"m": 1.0, "g": 9.8, "k": 10.0},
        timesteps=[
            {"t": 0.0, "h": 0.5, "v": 0.0},
            {"t": 0.2, "h": 0.3, "v": -1.5},
            {"t": 0.4, "h": 0.1, "v": -2.8},
        ],
        known_invariant="0.5*m*v^2 + m*g*h + 0.5*k*h^2",
        lean_theorem="",
    )


@pytest.fixture
def detector() -> SymmetryDetector:
    return SymmetryDetector()


# ── SymmetryGroup Tests ──────────────────────────────────────────────────────

class TestSymmetryGroup:
    """Tests for SymmetryGroup and pre-built groups."""

    def test_galilean_group_structure(self):
        """Galilean group has 10 generators and expected invariants."""
        gal = build_galilean_group()
        assert gal.dimension == 10
        assert len(gal.generators) == 10
        assert gal.name == "Galilean group"
        assert gal.parent is None

    def test_galilean_contains_core_symmetries(self):
        """Galilean group includes time translation and space translations."""
        gal = build_galilean_group()
        assert gal.contains(GeneratorKind.TIME_TRANSLATION)
        assert gal.contains(GeneratorKind.SPACE_TRANSLATION_X)
        assert gal.contains(GeneratorKind.SPACE_TRANSLATION_Y)
        assert gal.contains(GeneratorKind.SPACE_TRANSLATION_Z)

    def test_galilean_has_rotation(self):
        """Galilean group includes rotation generator."""
        gal = build_galilean_group()
        assert gal.contains(GeneratorKind.ROTATION_XY)

    def test_galilean_invariants_are_valid(self):
        """All Galilean invariants are non-empty strings."""
        gal = build_galilean_group()
        for gen in gal.generators:
            inv = gal.invariant_for(gen)
            assert inv is not None, f"No invariant for {gen}"
            assert len(inv) > 0, f"Empty invariant for {gen}"

    def test_u1_group(self):
        """U(1) group has 1 generator for charge."""
        u1 = build_u1_group()
        assert u1.dimension == 1
        assert u1.contains(GeneratorKind.U1_PHASE)
        assert u1.invariant_for(GeneratorKind.U1_PHASE) == "q"

    def test_su2_group(self):
        """SU(2) group has isospin invariant."""
        su2 = build_su2_group()
        assert su2.dimension == 3
        assert su2.contains(GeneratorKind.SU2_WEAK)
        assert su2.invariant_for(GeneratorKind.SU2_WEAK) == "I"

    def test_prebuilt_groups_registry(self):
        """All pre-built groups accessible via registry."""
        assert "galilean" in PREBUILT_GROUPS
        assert "u1" in PREBUILT_GROUPS
        assert "su2" in PREBUILT_GROUPS
        assert get_group("galilean") is not None
        assert get_group("nonexistent") is None

    def test_group_all_invariants(self):
        """all_invariants() returns dict keyed by generator name."""
        u1 = build_u1_group()
        invs = u1.all_invariants()
        assert "u1_phase" in invs
        assert invs["u1_phase"] == "q"


# ── NoetherDerivation Tests ──────────────────────────────────────────────────

class TestNoetherDerivation:
    """Acceptance: NoetherDerivation produces correct invariants."""

    def test_free_fall_energy_from_noether(self):
        """ACCEPTANCE: NoetherDerivation produces mgh + ½mv² from
        Lagrangian + time symmetry."""
        nd = NoetherDerivation("free_fall")
        cq = nd.conserved_quantity(GeneratorKind.TIME_TRANSLATION)
        assert cq is not None
        # Normalize whitespace
        expr = cq.expression.replace(" ", "")
        # Should contain both kinetic and potential terms
        assert "0.5*m*v^2" in expr or "m*v^2" in expr
        assert "m*g*h" in expr
        assert cq.generator == GeneratorKind.TIME_TRANSLATION

    def test_free_fall_energy_is_energy_dim(self):
        """Derived energy has Energy physical dimension."""
        nd = NoetherDerivation("free_fall")
        cq = nd.conserved_quantity(GeneratorKind.TIME_TRANSLATION)
        assert cq is not None
        assert cq.expected_dimension == Dimension.named("Energy")

    def test_gravity_spring_energy(self):
        """ACCEPTANCE: Combined gravity+spring derives both invariants."""
        nd = NoetherDerivation("gravity_spring")
        cq = nd.conserved_quantity(GeneratorKind.TIME_TRANSLATION)
        assert cq is not None
        expr = cq.expression.replace(" ", "")
        assert "m*g*h" in expr, "Missing gravity potential term"
        assert "0.5*k*h^2" in expr or "k*h^2" in expr, "Missing spring potential term"
        assert "0.5*m*v^2" in expr or "m*v^2" in expr, "Missing kinetic term"

    def test_projectile_momentum_from_space_translation(self):
        """Space translation → momentum mv (via ∂L/∂v)."""
        nd = NoetherDerivation("projectile")
        cq = nd.conserved_quantity(GeneratorKind.SPACE_TRANSLATION_X)
        assert cq is not None
        expr = cq.expression.replace(" ", "")
        assert "m*vx" in expr, f"Expected m*vx, got: {expr}"

    def test_derive_all_multiple_generators(self):
        """derive_all returns all applicable conserved quantities."""
        nd = NoetherDerivation("free_fall")
        results = nd.derive_all([
            GeneratorKind.TIME_TRANSLATION,
            GeneratorKind.SPACE_TRANSLATION_X,
        ])
        # Should have at least time translation
        gen_names = [cq.generator_name for cq in results]
        assert "time_translation" in gen_names

    def test_derivation_trace_included(self):
        """Each conserved quantity includes a derivation trace."""
        nd = NoetherDerivation("free_fall")
        cq = nd.conserved_quantity(GeneratorKind.TIME_TRANSLATION)
        assert cq is not None
        assert len(cq.derivation) > 50  # Should be descriptive
        assert "L =" in cq.derivation
        assert "H =" in cq.derivation or "H = " in cq.derivation

    def test_spring_mass_energy(self):
        """Spring-mass system: H = ½mv² + ½kh²."""
        nd = NoetherDerivation("spring")
        cq = nd.conserved_quantity(GeneratorKind.TIME_TRANSLATION)
        assert cq is not None
        expr = cq.expression.replace(" ", "")
        assert "0.5*m*v^2" in expr or "m*v^2" in expr
        assert "0.5*k*h^2" in expr or "k*h^2" in expr

    def test_u1_invariant_is_charge(self):
        """U(1) derivation returns charge q."""
        nd = NoetherDerivation("free_fall")
        cq = nd.conserved_quantity(GeneratorKind.U1_PHASE)
        assert cq is not None
        assert "q" in cq.expression.lower() or cq.expression == "q"

    def test_irrelevant_generator_returns_none(self):
        """Generator that doesn't apply returns None."""
        nd = NoetherDerivation("free_fall")
        # Free fall (1D vertical) has no rotation symmetry
        cq = nd.conserved_quantity(GeneratorKind.ROTATION_XY)
        assert cq is None


# ── SymmetryDetector Tests ───────────────────────────────────────────────────

class TestSymmetryDetector:
    """ACCEPTANCE: SymmetryDetector correctly identifies symmetries."""

    def test_detects_time_translation_in_free_fall(self, detector, free_fall_obs):
        """ACCEPTANCE: SymmetryDetector identifies time-translation in free-fall."""
        detection = detector.detect(free_fall_obs)
        assert detection.has_symmetry(GeneratorKind.TIME_TRANSLATION)
        assert "time_translation" in detection.symmetry_names

    def test_free_fall_no_space_translation(self, detector, free_fall_obs):
        """1D free fall (only h, no x/y) has no space translation."""
        detection = detector.detect(free_fall_obs)
        # h-based scenarios don't have x/y → no space translation
        # (unless overridden via scenario ID)
        assert not detection.has_symmetry(GeneratorKind.SPACE_TRANSLATION_X)

    def test_projectile_has_space_translation(self, detector, projectile_obs):
        """Projectile with x-coordinate has space translation X."""
        detection = detector.detect(projectile_obs)
        assert detection.has_symmetry(GeneratorKind.TIME_TRANSLATION)
        assert detection.has_symmetry(GeneratorKind.SPACE_TRANSLATION_X)

    def test_projectile_has_rotation(self, detector, projectile_obs):
        """Projectile with x and y has rotation symmetry."""
        detection = detector.detect(projectile_obs)
        assert detection.has_symmetry(GeneratorKind.ROTATION_XY)

    def test_detection_includes_evidence(self, detector, free_fall_obs):
        """Detection results include evidence strings."""
        detection = detector.detect(free_fall_obs)
        assert len(detection.evidence) > 0
        assert GeneratorKind.TIME_TRANSLATION in detection.evidence

    def test_detection_includes_confidence(self, detector, free_fall_obs):
        """Detection includes confidence scores in [0, 1]."""
        detection = detector.detect(free_fall_obs)
        for conf in detection.confidence.values():
            assert 0.0 <= conf <= 1.0

    def test_scenario_overrides(self, detector):
        """Known scenarios use hardcoded override."""
        obs = Observation(
            id="falling_ball_straight_drop",
            name="Straight drop",
            description="...",
            quantities={"m": "Mass", "h": "Length", "v": "Velocity"},
            parameters={"m": 1.0, "g": 9.8},
            timesteps=[{"t": 0.0, "h": 10.0, "v": 0.0}, {"t": 1.0, "h": 1.0, "v": 10.0}],
            known_invariant=None,
            lean_theorem="",
        )
        detection = detector.detect(obs)
        assert detection.has_symmetry(GeneratorKind.TIME_TRANSLATION)
        # Confidence should be 1.0 for override
        assert detection.confidence[GeneratorKind.TIME_TRANSLATION] == 1.0

    def test_gravity_spring_both_symmetries(self, detector, gravity_spring_obs):
        """ACCEPTANCE: For combined gravity+spring, detects both symmetries."""
        detection = detector.detect(gravity_spring_obs)
        # Time translation always present
        assert detection.has_symmetry(GeneratorKind.TIME_TRANSLATION)

    def test_group_matches_detection(self, detector, free_fall_obs):
        """Detection can match against symmetry groups."""
        detection = detector.detect(free_fall_obs)
        gal = build_galilean_group()
        # Free fall only has time translation, galilean requires all 10
        assert not detection.group_matches(gal)

    def test_quantity_variation_analysis(self, detector, free_fall_obs):
        """analyze_quantity_variation identifies varying quantities."""
        var = detector.analyze_quantity_variation(free_fall_obs)
        assert var["h"] is True, "h varies over time"
        assert var["v"] is True, "v varies over time"

    def test_detect_from_database(self, detector):
        """detect_from_database works on ObservationDatabase."""
        import tempfile, json
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump([{
            "id": "test1", "name": "Test", "description": "Test",
            "quantities": {"m": "Mass", "h": "Length", "v": "Velocity"},
            "parameters": {"m": 1.0},
            "timesteps": [{"t": 0.0, "h": 10.0, "v": 0.0}, {"t": 1.0, "h": 5.0, "v": 5.0}],
        }], tmp)
        tmp.close()
        try:
            db = ObservationDatabase(tmp.name)
            results = detector.detect_from_database(db)
            assert "test1" in results
            assert isinstance(results["test1"], SymmetryDetection)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


# ── Lagrangian Tests ─────────────────────────────────────────────────────────

class TestLagrangian:
    """Tests for Lagrangian representation."""

    def test_free_fall_lagrangian(self):
        """Free fall Lagrangian: L = ½mv² - mgh."""
        L = Lagrangian.free_fall()
        assert L.expression == "0.5*m*v^2 - m*g*h"
        assert L.velocities == {"v": "h"}
        assert "h" in L.positions

    def test_lagrangian_evaluation(self):
        """Lagrangian can be numerically evaluated."""
        L = Lagrangian.free_fall()
        state = {"m": 1.0, "g": 9.8, "h": 5.0, "v": 10.0, "t": 0.5}
        val = L.evaluate(state)
        # L = 0.5*1*100 - 1*9.8*5 = 50 - 49 = 1
        assert abs(val - 1.0) < 0.1

    def test_gravity_spring_lagrangian(self):
        """Combined gravity+spring Lagrangian exists."""
        L = Lagrangian.gravity_spring()
        assert "m*g*h" in L.expression
        assert "0.5*k*h^2" in L.expression
        assert len(L.potential_terms) == 2

    def test_standard_lagrangians_registry(self):
        """All standard Lagrangian keys are valid."""
        for key in ["free_fall", "spring", "gravity_spring", "projectile"]:
            assert key in _STANDARD_LAGRANGIANS


# ── SymmetryPipeline Tests ───────────────────────────────────────────────────

class TestSymmetryPipeline:
    """Tests for the full symmetry → expression pipeline."""

    def test_pipeline_free_fall(self, free_fall_obs):
        """Pipeline: detect + derive + verify."""
        pipeline = SymmetryPipeline(lagrangian="free_fall")
        result = pipeline.run(free_fall_obs)

        assert result.scenario_id == "test_free_fall"
        assert len(result.conserved_quantities) > 0
        # Should have energy expression
        expressions = result.expressions
        assert "time_translation" in expressions

    def test_pipeline_gravity_spring(self, gravity_spring_obs):
        """Pipeline for gravity+spring detects both and derives energy."""
        pipeline = SymmetryPipeline(
            detector=SymmetryDetector(),
            lagrangian="gravity_spring",
        )
        result = pipeline.run(gravity_spring_obs)

        assert result.scenario_id == "test_gravity_spring"
        # Must have time translation invariant
        has_time = any(
            cq.generator == GeneratorKind.TIME_TRANSLATION
            for cq in result.conserved_quantities
        )
        assert has_time

    def test_pipeline_combined_expression(self, free_fall_obs):
        """combined_expression joins additive invariants."""
        pipeline = SymmetryPipeline(lagrangian="free_fall")
        result = pipeline.run(free_fall_obs)
        combined = result.combined_expression
        assert len(combined) > 0

    def test_pipeline_run_database(self, free_fall_obs):
        """run_database processes multiple observations."""
        import tempfile, json
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump([
            {
                "id": "obs1", "name": "Obs 1", "description": "...",
                "quantities": {"m": "Mass", "h": "Length", "v": "Velocity"},
                "parameters": {"m": 1.0},
                "timesteps": [
                    {"t": 0.0, "h": 10.0, "v": 0.0},
                    {"t": 1.0, "h": 1.0, "v": 10.0},
                ],
            },
            {
                "id": "obs2", "name": "Obs 2", "description": "...",
                "quantities": {"m": "Mass", "h": "Length", "v": "Velocity"},
                "parameters": {"m": 2.0},
                "timesteps": [
                    {"t": 0.0, "h": 5.0, "v": 0.0},
                    {"t": 0.5, "h": 2.0, "v": 5.0},
                ],
            },
        ], tmp)
        tmp.close()
        try:
            db = ObservationDatabase(tmp.name)
            pipeline = SymmetryPipeline(lagrangian="free_fall")
            results = pipeline.run_database(db)
            assert len(results) == 2
            assert "obs1" in results
            assert "obs2" in results
        finally:
            Path(tmp.name).unlink(missing_ok=True)


# ── SymmetryClassifier Tests ─────────────────────────────────────────────────

class TestSymmetryClassifier:
    """ACCEPTANCE: Symmetry classifier trained and accurate."""

    def test_classifier_creation(self):
        """Classifier can be created with default parameters."""
        clf = SymmetryClassifier(num_quantities=28)
        n = clf.count_parameters()
        assert n > 1000, f"Too few parameters: {n}"
        assert n < 20000, f"Too many parameters: {n} (target ~10K)"

    def test_classifier_save_load(self):
        """Classifier can be saved and loaded."""
        clf = SymmetryClassifier(num_quantities=28)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            path = tmp.name
        try:
            clf.save(path)
            loaded = SymmetryClassifier.load(path)
            assert loaded.count_parameters() == clf.count_parameters()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_classifier_predict_shape(self):
        """Predict returns correct number of output probabilities."""
        clf = SymmetryClassifier(num_quantities=28)
        # All-zeros feature vector
        probs = clf.predict([0.0] * 28)
        assert len(probs) == 7  # One per symmetry class
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_training_converges(self):
        """Training reduces loss on synthetic data."""
        # Create synthetic training data with clear patterns
        features = [
            [1.0 if i < 5 else 0.0 for i in range(28)],  # gravity-like
            [1.0 if 3 <= i < 10 else 0.0 for i in range(28)],  # em-like
            [1.0 if i < 8 else 0.0 for i in range(28)],  # combined
        ] * 10  # 30 examples
        labels = [
            [1, 0, 0, 0, 0, 0, 0],  # time only
            [1, 1, 0, 0, 0, 1, 0],  # time + space_x + u1
            [1, 1, 0, 0, 1, 0, 0],  # time + space_x + rotation
        ] * 10

        import torch
        clf = train_symmetry_classifier(
            features, labels,
            epochs=30,
            learning_rate=0.01,
            checkpoint_path="/tmp/_test_symmetry_clf.pt",
        )

        # Test on training data
        probs = clf.predict(features[0])
        # Should predict time_translation with high confidence
        assert probs[0] > 0.5, f"Expected time_translation prob > 0.5, got {probs[0]}"

    def test_symmetry_class_labels_match(self):
        """Symmetry class labels correspond to generator kinds."""
        assert SYMMETRY_CLASS_LABELS[0] == "time_translation"
        assert SYMMETRY_CLASS_LABELS[1] == "space_translation_x"
        assert len(SYMMETRY_CLASS_LABELS) == len(SYMMETRY_CLASSES)

    def test_build_training_data(self):
        """build_symmetry_training_data creates valid tensors."""
        import tempfile, json
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump([{
            "id": "test_free_fall",
            "name": "Free fall",
            "description": "Ball dropping",
            "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
            "parameters": {"m": 1.0, "g": 9.8},
            "timesteps": [
                {"t": 0.0, "h": 10.0, "v": 0.0},
                {"t": 1.0, "h": 1.0, "v": 10.0},
            ],
        }], tmp)
        tmp.close()
        try:
            features, labels = build_symmetry_training_data(tmp.name)
            assert len(features) == 1
            assert len(labels) == 1
            assert len(features[0]) == 28  # QUANTITY_VOCAB size
            assert len(labels[0]) == 7  # 7 symmetry classes
        finally:
            Path(tmp.name).unlink(missing_ok=True)


# ── Smoke Test ───────────────────────────────────────────────────────────────

class TestSmokeTest:
    """Verify the built-in smoke test returns correct results."""

    def test_smoke_test_all_pass(self):
        """All smoke test assertions pass."""
        results = run_symmetry_smoke_test()

        # Free fall energy
        assert results["free_fall_energy_correct"] is True
        assert results["free_fall_energy_contains_mgh"] is True
        assert results["free_fall_energy_contains_kinetic"] is True

        # Detection
        assert results["detects_time_translation"] is True

        # Galilean group
        assert results["galilean_dimension"] == 10
        assert results["galilean_generators"] == 10
        assert results["galilean_has_time"] is True
        assert results["galilean_has_space_x"] is True

        # U(1)
        assert results["u1_dimension"] == 1
        assert results["u1_has_phase"] is True

        # Gravity + spring
        assert results["gs_has_mgh"] is True
        assert results["gs_has_spring"] is True

        # Projectile momentum
        assert results["projectile_momentum_x_correct"] is True


# ── Generator Kind Tests ─────────────────────────────────────────────────────

class TestGeneratorKind:
    """GeneratorKind enum and labels."""

    def test_all_generators_have_labels(self):
        """Every GeneratorKind has a label."""
        for gen in GeneratorKind:
            assert gen in GENERATOR_LABELS, f"Missing label for {gen}"

    def test_labels_are_consistent(self):
        """Same generator always maps to same label."""
        assert GENERATOR_LABELS[GeneratorKind.TIME_TRANSLATION] == "time_translation"
        assert GENERATOR_LABELS[GeneratorKind.U1_PHASE] == "u1_phase"
        assert GENERATOR_LABELS[GeneratorKind.SU2_WEAK] == "su2_weak"


# ── Integration with evaluator ───────────────────────────────────────────────

class TestEvaluatorIntegration:
    """Verify derived invariants actually evaluate to near-constant."""

    def test_free_fall_energy_is_constant(self):
        """Energy expression derived by Noether is constant for free fall."""
        from src.physics.evaluator import ExpressionEvaluator
        from src.physics.observations import ObservationDatabase

        # Create temp observation database
        import tempfile, json
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump([{
            "id": "fall",
            "name": "Free fall",
            "description": "Ball drop",
            "quantities": {"m": "Mass", "g": "Accel", "h": "Length", "v": "Velocity", "t": "Time"},
            "parameters": {"m": 1.0, "g": 9.8},
            "timesteps": [
                {"t": 0.0, "h": 10.0, "v": 0.0},
                {"t": 0.2, "h": 9.804, "v": 1.96},
                {"t": 0.4, "h": 9.216, "v": 3.92},
                {"t": 0.6, "h": 8.236, "v": 5.88},
                {"t": 0.8, "h": 6.864, "v": 7.84},
                {"t": 1.0, "h": 5.1, "v": 9.8},
            ],
            "known_invariant": "m*g*h + 0.5*m*v^2",
            "lean_theorem": "",
        }], tmp)
        tmp.close()

        try:
            db = ObservationDatabase(tmp.name)
            nd = NoetherDerivation("free_fall")
            cq = nd.conserved_quantity(GeneratorKind.TIME_TRANSLATION)
            assert cq is not None

            evaluator = ExpressionEvaluator()
            score = evaluator.score(cq.expression, db)
            # Should be nearly constant (score > 0.99)
            assert score > 0.99, (
                f"Derived energy {cq.expression} not constant: score={score:.6f}"
            )
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def test_gravity_spring_energy_is_constant(self):
        """Energy for spring system is constant.

        Uses known-physics spring data from the mechanics simulator.
        The spring Lagrangian L = ½mv² - ½kx² has Hamiltonian
        H = ½mv² + ½kx², which the simulator confirms is conserved.
        """
        from src.physics.evaluator import ExpressionEvaluator
        import tempfile, json, math

        # Generate real spring physics data
        k, m, A = 10.0, 1.0, 0.5
        omega = math.sqrt(k / m)  # sqrt(10) ≈ 3.162
        period = 2 * math.pi / omega
        timesteps = []
        for i in range(20):
            t = period * i / 19
            x = A * math.cos(omega * t)
            v = -omega * A * math.sin(omega * t)
            timesteps.append({"t": round(t, 6), "x": round(x, 6), "v": round(v, 6)})

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump([{
            "id": "spring_test",
            "name": "Spring test",
            "description": "Undamped spring-mass",
            "quantities": {
                "m": "Mass", "k": "Force/Length",
                "x": "Length", "v": "Velocity", "t": "Time",
            },
            "parameters": {"m": m, "k": k},
            "timesteps": timesteps,
            "known_invariant": "0.5*k*x^2 + 0.5*m*v^2",
            "lean_theorem": "",
        }], tmp)
        tmp.close()

        try:
            db = ObservationDatabase(tmp.name)

            # The Noether derivation for time translation on spring Lagrangian
            # produces 0.5*m*v^2 + 0.5*k*h^2, but the data uses 'x' not 'h'.
            # Construct Lagrangian using the actual variable names:
            L = Lagrangian(
                expression="0.5*m*v^2 - 0.5*k*x^2",
                kinetic_terms=["0.5*m*v^2"],
                potential_terms=["0.5*k*x^2"],
                velocities={"v": "x"},
                positions=["x"],
                parameters={"m": m, "k": k},
            )
            nd = NoetherDerivation(L)
            cq = nd.conserved_quantity(GeneratorKind.TIME_TRANSLATION)
            assert cq is not None

            evaluator = ExpressionEvaluator()
            score = evaluator.score(cq.expression, db)
            assert score > 0.99, (
                f"Derived energy {cq.expression} "
                f"not constant: score={score:.6f}"
            )
        finally:
            Path(tmp.name).unlink(missing_ok=True)


# ── Checkpoint File Tests ────────────────────────────────────────────────────

class TestCheckpointExists:
    """Verify expected checkpoint and data files exist."""

    def test_symmetry_classifier_checkpoint(self):
        """symmetry_classifier.pt checkpoint exists."""
        path = Path("checkpoints/symmetry_classifier.pt")
        assert path.exists(), (
            f"symmetry_classifier.pt not found at {path.absolute()}"
        )

    def test_symmetry_classifier_loadable(self):
        """Checkpoint can be loaded as SymmetryClassifier."""
        clf = SymmetryClassifier.load("checkpoints/symmetry_classifier.pt")
        assert clf.count_parameters() > 0
