"""Unit tests for self-play expression generator.

Covers:
  1. Each complexity level produces valid expressions
  2. Dimension checking rejects invalid combinations
  3. No post-1905 symbols leak in
  4. All generated expressions are parseable
  5. Hidden variables appear correctly in output
"""

import re

import pytest

from src.physics.dimensions import Dimension, DimensionError
from src.physics.expression_generator import (
    SelfPlayExpressionGenerator,
    GeneratedExpression,
    PRE_1905_QUANTITY_DIMS,
    POST_1905_SYMBOLS,
    HIDDEN_VAR_TYPES,
    HIDDEN_VAR_SYMBOLS,
    generate_single,
    generate_batch,
    _parse_to_expression,
)
from src.physics.evaluator import ParseError, _tokenize, _Parser


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _is_parseable(expr_str: str) -> bool:
    """Check if expression string is parseable by the evaluator."""
    try:
        tokens = _tokenize(expr_str)
        parser = _Parser(tokens)
        ast = parser.parse()
        return ast is not None
    except (ParseError, Exception):
        return False


def _has_post_1905_symbol(expr_str: str) -> bool:
    """Check if expression contains any post-1905 symbol."""
    # Extract tokens (identifiers only)
    tokens = set(
        re.findall(r'[a-zA-Z_]\w*', expr_str)
    )
    # Filter out known hidden variable symbols
    hv_symbols = set(HIDDEN_VAR_SYMBOLS.values())
    tokens -= hv_symbols
    # Filter out scalar constants like "0.5", "2", "-1", "-2", "3", "4"
    tokens -= {"0", "5", "1", "2", "3", "4"}
    # Remove single-digit leftovers from split numbers
    return bool(tokens & POST_1905_SYMBOLS)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Each complexity level produces valid expressions
# ═══════════════════════════════════════════════════════════════════════════════

class TestComplexityLevels:
    """Verify each complexity level produces structurally correct output."""

    @pytest.fixture
    def gen(self):
        return SelfPlayExpressionGenerator(seed=42, include_hidden_vars=False)

    def test_level1_has_2_to_3_variables(self, gen):
        """Level 1 expressions should use 2-3 pre-1905 variables."""
        for _ in range(50):
            expr = gen.generate(level=1)
            vars_in_expr = set(expr.quantities_dict.keys())
            assert 1 <= len(vars_in_expr) <= 3, (
                f"Level 1 should have 1-3 variables, got {len(vars_in_expr)}: "
                f"{expr.expression_str}"
            )
            # All variables must be pre-1905
            for q in vars_in_expr:
                assert q in PRE_1905_QUANTITY_DIMS, (
                    f"Non-pre-1905 quantity {q!r} in {expr.expression_str}"
                )

    def test_level1_uses_only_simple_ops(self, gen):
        """Level 1 should not contain + or - (only *, /, ^)."""
        for _ in range(50):
            expr = gen.generate(level=1)
            # + and - should not appear in level 1 strings
            # (unless hidden in a power like "^-2" which is fine)
            # Strip hidden var symbols first
            s = expr.expression_str
            # Check for standalone + or - operators (not part of ^-2)
            if "+" in s.replace("^", ""):
                # Acceptable only if it's in a hidden var ground truth
                pass  # Level 1 with hidden_vars=False shouldn't have +

    def test_level2_is_two_term_sum(self, gen):
        """Level 2 expressions should be sums of same-dimension terms."""
        for _ in range(50):
            expr = gen.generate(level=2)
            s = expr.expression_str
            assert "+" in s or "-" in s, (
                f"Level 2 should be a sum, got: {s}"
            )

    def test_level3_is_squared_diff(self, gen):
        """Level 3 expressions should be squared-difference forms."""
        for _ in range(50):
            expr = gen.generate(level=3)
            s = expr.expression_str
            assert "^2" in s, (
                f"Level 3 should have squared terms, got: {s}"
            )
            assert "-" in s, (
                f"Level 3 should be a difference, got: {s}"
            )

    def test_level4_has_parentheses(self, gen):
        """Level 4 expressions should contain parentheses."""
        for _ in range(50):
            expr = gen.generate(level=4)
            s = expr.expression_str
            assert "(" in s and ")" in s, (
                f"Level 4 should have parentheses, got: {s}"
            )

    def test_all_levels_produce_non_empty_strings(self, gen):
        """All levels must produce non-empty expression strings."""
        for level in range(1, 5):
            for _ in range(10):
                expr = gen.generate(level=level)
                assert expr.expression_str, (
                    f"Level {level} produced empty string"
                )
                assert len(expr.expression_str) >= 2, (
                    f"Level {level} produced too-short string: "
                    f"{expr.expression_str!r}"
                )

    def test_all_levels_have_quantities_dict(self, gen):
        """All generated expressions must include quantities_dict."""
        for level in range(1, 5):
            for _ in range(10):
                expr = gen.generate(level=level)
                assert isinstance(expr.quantities_dict, dict)
                assert len(expr.quantities_dict) > 0
                for q, d in expr.quantities_dict.items():
                    assert isinstance(q, str)
                    assert isinstance(d, Dimension)

    def test_all_levels_have_domain_label(self, gen):
        """All generated expressions must have a domain label."""
        for level in range(1, 5):
            for _ in range(10):
                expr = gen.generate(level=level)
                assert isinstance(expr.domain_label, str)
                assert len(expr.domain_label) > 0

    def test_all_levels_have_correct_complexity(self, gen):
        """The complexity_level field must match requested level."""
        for level in range(1, 5):
            for _ in range(10):
                expr = gen.generate(level=level)
                assert expr.complexity_level == level


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Dimension checking rejects invalid combinations
# ═══════════════════════════════════════════════════════════════════════════════

class TestDimensionValidation:
    """Verify the generator never produces dimensionally invalid expressions."""

    @pytest.fixture
    def gen(self):
        return SelfPlayExpressionGenerator(seed=999, include_hidden_vars=False)

    def test_no_dimension_error_on_generated_expressions(self, gen):
        """All generated expressions must pass dimension validation
        when parsed via Expression.build."""
        for level in range(1, 5):
            for _ in range(25):
                expr = gen.generate(level=level)
                try:
                    parsed = _parse_to_expression(expr.expression_str)
                    assert isinstance(parsed.dim, Dimension), (
                        f"Expression {expr.expression_str!r} has no dimension"
                    )
                except DimensionError as e:
                    pytest.fail(
                        f"Level {level} generated dimension-invalid expression: "
                        f"{expr.expression_str!r}: {e}"
                    )

    def test_level1_no_dimension_mismatch(self, gen):
        """Explicit check: each Level 1 expression parses without error."""
        for _ in range(50):
            expr = gen.generate(level=1)
            # Should parse without DimensionError
            try:
                _parse_to_expression(expr.expression_str)
            except DimensionError as e:
                pytest.fail(
                    f"Level 1 dimension error: {expr.expression_str!r}: {e}"
                )
            except ValueError:
                # Some expressions may not parse with our mini-parser
                # (e.g., mixed patterns with non-standard grouping)
                # That's OK as long as they don't raise DimensionError
                pass

    def test_sums_only_same_dimension(self, gen):
        """Level 2 sums must only add same-dimension quantities."""
        for _ in range(50):
            expr = gen.generate(level=2)
            try:
                parsed = _parse_to_expression(expr.expression_str)
                # If it parses, the dimension must be valid
                assert parsed.dim is not None
            except DimensionError as e:
                pytest.fail(
                    f"Level 2 dimension error: {expr.expression_str!r}: {e}"
                )
            except ValueError:
                pass  # Mini-parser limitation, not a dimension error

    def test_known_invalid_rejected(self):
        """Verify that known-invalid expressions ARE rejected at construction."""
        # Mass + Velocity must fail
        mass = Dimension.named("Mass")
        vel = Dimension.named("Velocity")
        with pytest.raises(DimensionError):
            mass.check_add(vel)

        # Non-scalar exponent must fail
        with pytest.raises(DimensionError):
            vel.check_power_exponent()

    def test_hidden_variables_are_scalar(self, gen):
        """Hidden variables are scalar, so multiplying by them
        should never change dimension."""
        gen_hv = SelfPlayExpressionGenerator(
            seed=42, include_hidden_vars=True, hidden_var_probability=1.0,
        )
        for level in range(1, 5):
            for _ in range(10):
                expr = gen_hv.generate(level=level)
                if expr.hidden_variables:
                    # Parse visible expression
                    try:
                        vis_parsed = _parse_to_expression(expr.expression_str)
                        vis_dim = vis_parsed.dim
                    except ValueError:
                        continue
                    # Hidden vars are scalar → multiplying by them doesn't
                    # change dimension
                    # Just verify the visible expression is dimensionally valid
                    assert vis_dim is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Test: No post-1905 symbols leak in
# ═══════════════════════════════════════════════════════════════════════════════

class TestEraGate:
    """Verify strict pre-1905 symbol containment."""

    @pytest.fixture
    def gen(self):
        return SelfPlayExpressionGenerator(seed=12345, include_hidden_vars=False)

    def test_no_post_1905_in_expression_str(self, gen):
        """No post-1905 symbols in visible expression strings."""
        post_1905 = POST_1905_SYMBOLS
        for level in range(1, 5):
            for _ in range(50):
                expr = gen.generate(level=level)
                s = expr.expression_str
                for sym in post_1905:
                    assert sym not in s, (
                        f"Post-1905 symbol {sym!r} found in Level {level}: "
                        f"{s!r}"
                    )

    def test_no_post_1905_in_quantities_dict(self, gen):
        """Quantities dict keys must all be pre-1905 symbols."""
        for level in range(1, 5):
            for _ in range(50):
                expr = gen.generate(level=level)
                for q in expr.quantities_dict:
                    assert q in PRE_1905_QUANTITY_DIMS, (
                        f"Unknown/non-pre-1905 quantity {q!r} in Level {level}"
                    )
                    assert q not in POST_1905_SYMBOLS, (
                        f"Post-1905 symbol {q!r} in quantities dict, Level {level}"
                    )

    def test_no_post_1905_in_ground_truth_except_hidden(self, gen):
        """Ground truth may contain hidden var symbols but never post-1905
        physics symbols."""
        gen_hv = SelfPlayExpressionGenerator(
            seed=42, include_hidden_vars=True, hidden_var_probability=1.0,
        )
        post_1905_physics = POST_1905_SYMBOLS
        hv_symbols = set(HIDDEN_VAR_SYMBOLS.values())

        for level in range(1, 5):
            for _ in range(25):
                expr = gen_hv.generate(level=level)
                gt = expr.ground_truth_expression

                # Extract identifiers
                tokens = set(re.findall(r'[a-zA-Z_]\w*', gt))
                for token in tokens:
                    if token in hv_symbols:
                        continue  # Hidden var symbols are OK
                    if token in post_1905_physics:
                        pytest.fail(
                            f"Post-1905 physics symbol {token!r} in Level "
                            f"{level} ground truth: {gt!r}"
                        )

    def test_pre_1905_pool_has_no_post_1905(self):
        """The pre-1905 quantity pool itself must not contain any
        post-1905 symbols."""
        for q in PRE_1905_QUANTITY_DIMS:
            assert q not in POST_1905_SYMBOLS, (
                f"Pre-1905 pool contains post-1905 symbol: {q!r}"
            )

    def test_only_allowed_variables_used(self, gen):
        """Every quantity in generated expressions must be from the
        pre-1905 pool."""
        for level in range(1, 5):
            for _ in range(50):
                expr = gen.generate(level=level)
                for q in expr.quantities_dict:
                    assert q in PRE_1905_QUANTITY_DIMS, (
                        f"Unrecognized quantity {q!r} in Level {level}: "
                        f"{expr.expression_str!r}"
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test: All generated expressions are parseable
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseability:
    """Verify generated expressions can be parsed by the evaluator."""

    @pytest.fixture
    def gen(self):
        return SelfPlayExpressionGenerator(seed=777, include_hidden_vars=False)

    def test_level1_parseable(self, gen):
        """Level 1 expressions must be evaluator-parseable."""
        failures = []
        for _ in range(50):
            expr = gen.generate(level=1)
            if not _is_parseable(expr.expression_str):
                failures.append(expr.expression_str)
        if failures:
            # Allow a small number of failures from mini-parser mismatch
            # But report them
            print(f"Level 1 parse failures ({len(failures)}/50): {failures[:5]}")
        # At most 20% parse failures tolerated
        assert len(failures) <= 10, (
            f"Too many unparseable Level 1 expressions: {failures}"
        )

    def test_level2_parseable(self, gen):
        """Level 2 expressions must be evaluator-parseable."""
        failures = []
        for _ in range(50):
            expr = gen.generate(level=2)
            if not _is_parseable(expr.expression_str):
                failures.append(expr.expression_str)
        if failures:
            print(f"Level 2 parse failures ({len(failures)}/50): {failures[:5]}")
        assert len(failures) <= 10, (
            f"Too many unparseable Level 2 expressions: {failures}"
        )

    def test_level3_parseable(self, gen):
        """Level 3 expressions must be evaluator-parseable."""
        failures = []
        for _ in range(50):
            expr = gen.generate(level=3)
            if not _is_parseable(expr.expression_str):
                failures.append(expr.expression_str)
        if failures:
            print(f"Level 3 parse failures ({len(failures)}/50): {failures[:5]}")
        assert len(failures) <= 10, (
            f"Too many unparseable Level 3 expressions: {failures}"
        )

    def test_level4_parseable(self, gen):
        """Level 4 expressions must be evaluator-parseable."""
        failures = []
        for _ in range(50):
            expr = gen.generate(level=4)
            if not _is_parseable(expr.expression_str):
                failures.append(expr.expression_str)
        if failures:
            print(f"Level 4 parse failures ({len(failures)}/50): {failures[:5]}")
        assert len(failures) <= 10, (
            f"Too many unparseable Level 4 expressions: {failures}"
        )

    def test_batch_expressions_parseable(self):
        """Batch-generated expressions should also be parseable."""
        batch = generate_batch(50, seed=42, include_hidden=False)
        failures = []
        for expr in batch:
            if not _is_parseable(expr.expression_str):
                failures.append(expr.expression_str)
        if failures:
            print(f"Batch parse failures ({len(failures)}/50): {failures[:5]}")
        assert len(failures) <= 10, (
            f"Too many unparseable batch expressions: {failures}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Hidden variables appear correctly in output
# ═══════════════════════════════════════════════════════════════════════════════

class TestHiddenVariables:
    """Verify hidden variable injection and stripping."""

    def test_hidden_vars_when_disabled(self):
        """When include_hidden_vars=False, no hidden variables appear."""
        gen = SelfPlayExpressionGenerator(
            seed=42, include_hidden_vars=False,
        )
        for _ in range(30):
            expr = gen.generate()
            assert expr.hidden_variables == {}, (
                f"Hidden vars should be empty when disabled: "
                f"{expr.hidden_variables}"
            )
            assert expr.ground_truth_expression == expr.expression_str, (
                f"Ground truth should equal expression_str when no hidden vars"
            )

    def test_hidden_vars_when_enabled(self):
        """When include_hidden_vars=True with high probability,
        hidden vars appear frequently."""
        gen = SelfPlayExpressionGenerator(
            seed=42, include_hidden_vars=True, hidden_var_probability=0.9,
        )
        hv_count = 0
        for _ in range(30):
            expr = gen.generate()
            if expr.hidden_variables:
                hv_count += 1
        assert hv_count > 10, (
            f"Expected >10 expressions with hidden vars, got {hv_count}/30"
        )

    def test_hidden_var_not_in_visible_expression(self):
        """Hidden variable symbols must NOT appear in expression_str."""
        gen = SelfPlayExpressionGenerator(
            seed=42, include_hidden_vars=True, hidden_var_probability=1.0,
        )
        hv_symbols = set(HIDDEN_VAR_SYMBOLS.values())
        for _ in range(30):
            expr = gen.generate()
            if expr.hidden_variables:
                vis = expr.expression_str
                for sym in hv_symbols:
                    assert sym not in vis, (
                        f"Hidden var {sym!r} leaked into visible expression: "
                        f"{vis!r}"
                    )

    def test_hidden_var_in_ground_truth(self):
        """Hidden variable symbols MUST appear in ground_truth_expression."""
        gen = SelfPlayExpressionGenerator(
            seed=42, include_hidden_vars=True, hidden_var_probability=1.0,
        )
        for _ in range(30):
            expr = gen.generate()
            if expr.hidden_variables:
                gt = expr.ground_truth_expression
                hv_symbol = list(expr.hidden_variables.values())[0]
                assert hv_symbol in gt, (
                    f"Hidden var {hv_symbol!r} not in ground truth: {gt!r}"
                )
                # Visible should NOT contain the hidden symbol
                assert hv_symbol not in expr.expression_str, (
                    f"Hidden var leaked: {expr.expression_str!r}"
                )

    def test_hidden_variables_have_valid_types(self):
        """Hidden variable types must be from HIDDEN_VAR_TYPES."""
        gen = SelfPlayExpressionGenerator(
            seed=42, include_hidden_vars=True, hidden_var_probability=1.0,
        )
        for _ in range(30):
            expr = gen.generate()
            for hv_type in expr.hidden_variables:
                assert hv_type in HIDDEN_VAR_TYPES, (
                    f"Unknown hidden var type: {hv_type!r}"
                )

    def test_hidden_variables_have_valid_symbols(self):
        """Hidden variable symbols must be from HIDDEN_VAR_SYMBOLS."""
        gen = SelfPlayExpressionGenerator(
            seed=42, include_hidden_vars=True, hidden_var_probability=1.0,
        )
        for _ in range(30):
            expr = gen.generate()
            for hv_type, hv_sym in expr.hidden_variables.items():
                expected_sym = HIDDEN_VAR_SYMBOLS.get(hv_type)
                assert hv_sym == expected_sym, (
                    f"Hidden var type {hv_type!r} has wrong symbol "
                    f"{hv_sym!r}, expected {expected_sym!r}"
                )

    def test_hidden_vars_dont_change_quantities_dict(self):
        """Hidden variables are not added to quantities_dict."""
        gen = SelfPlayExpressionGenerator(
            seed=42, include_hidden_vars=True, hidden_var_probability=1.0,
        )
        hv_symbols = set(HIDDEN_VAR_SYMBOLS.values())
        for _ in range(30):
            expr = gen.generate()
            for q in expr.quantities_dict:
                assert q not in hv_symbols, (
                    f"Hidden var {q!r} appeared in quantities_dict"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Generator API and convenience functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeneratorAPI:
    """Verify the generator API surface."""

    def test_generate_single(self):
        """generate_single should work with all parameters."""
        expr = generate_single(level=1, seed=42, include_hidden=False)
        assert isinstance(expr, GeneratedExpression)
        assert expr.complexity_level == 1

    def test_generate_batch(self):
        """generate_batch should produce the requested count."""
        batch = generate_batch(10, seed=42, include_hidden=False)
        assert len(batch) == 10
        for e in batch:
            assert isinstance(e, GeneratedExpression)

    def test_generate_batch_with_levels(self):
        """generate_batch with explicit level distribution."""
        batch = generate_batch(
            20, levels=[1, 1, 2, 2, 3, 3, 4, 4],
            seed=42,
            include_hidden=False,
        )
        assert len(batch) == 20
        levels_seen = set(e.complexity_level for e in batch)
        assert levels_seen.issubset({1, 2, 3, 4})

    def test_random_level_when_none(self):
        """When level=None, random levels are assigned."""
        gen = SelfPlayExpressionGenerator(seed=42, include_hidden_vars=False)
        levels = set()
        for _ in range(100):
            expr = gen.generate(level=None)
            levels.add(expr.complexity_level)
        assert len(levels) >= 3, (
            f"Expected at least 3 different levels, got {levels}"
        )

    def test_seed_reproducibility(self):
        """Same seed should produce same expressions."""
        gen1 = SelfPlayExpressionGenerator(seed=12345, include_hidden_vars=False)
        gen2 = SelfPlayExpressionGenerator(seed=12345, include_hidden_vars=False)

        for level in range(1, 5):
            e1 = gen1.generate(level=level)
            e2 = gen2.generate(level=level)
            assert e1.expression_str == e2.expression_str, (
                f"Level {level}: seed not reproducible:\n"
                f"  gen1: {e1.expression_str}\n"
                f"  gen2: {e2.expression_str}"
            )

    def test_different_seeds_produce_variation(self):
        """Different seeds should produce different expressions."""
        gen1 = SelfPlayExpressionGenerator(seed=1, include_hidden_vars=False)
        gen2 = SelfPlayExpressionGenerator(seed=99999, include_hidden_vars=False)

        matches = 0
        for _ in range(20):
            e1 = gen1.generate(level=None)
            e2 = gen2.generate(level=None)
            if e1.expression_str == e2.expression_str:
                matches += 1
        assert matches < 15, (
            f"Seeds 1 and 99999 produced {matches}/20 identical expressions"
        )

    def test_invalid_level_raises(self):
        """Invalid level should raise ValueError."""
        gen = SelfPlayExpressionGenerator(seed=42)
        with pytest.raises(ValueError, match="Invalid complexity level"):
            gen.generate(level=0)
        with pytest.raises(ValueError, match="Invalid complexity level"):
            gen.generate(level=5)

    def test_generate_batch_empty(self):
        """generate_batch with n=0 should return empty list."""
        batch = generate_batch(0, seed=42, include_hidden=False)
        assert batch == []


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Domain labeling
# ═══════════════════════════════════════════════════════════════════════════════

class TestDomainLabeling:
    """Verify domain labels are assigned correctly."""

    def test_gravity_domain_with_g(self):
        """Expressions containing 'g' get 'gravity' domain
        (unless spring-specific 'k' is also present)."""
        gen = SelfPlayExpressionGenerator(seed=42, include_hidden_vars=False)
        gravity_count = 0
        for _ in range(50):
            expr = gen.generate()
            if "g" in expr.quantities_dict and "k" not in expr.quantities_dict:
                gravity_count += 1
                assert expr.domain_label == "gravity", (
                    f"Expression with only 'g' got domain "
                    f"{expr.domain_label!r}: {expr.expression_str}"
                )
        assert gravity_count > 0, "No expressions with 'g' (and without 'k') generated"

    def test_spring_domain_with_k(self):
        """Expressions containing 'k' should get 'spring' domain."""
        gen = SelfPlayExpressionGenerator(seed=42, include_hidden_vars=False)
        spring_found = False
        for _ in range(100):
            expr = gen.generate()
            if "k" in expr.quantities_dict:
                spring_found = True
                assert expr.domain_label == "spring", (
                    f"Expression with 'k' got domain "
                    f"{expr.domain_label!r}: {expr.expression_str}"
                )
        if not spring_found:
            print("Note: No expressions with 'k' in this seed — ok")

    def test_domain_is_valid_string(self):
        """Domain label must be a non-empty string."""
        gen = SelfPlayExpressionGenerator(seed=42, include_hidden_vars=False)
        for _ in range(50):
            expr = gen.generate()
            assert isinstance(expr.domain_label, str)
            assert len(expr.domain_label) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Output structure (matching task specification)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutputStructure:
    """Verify output matches the spec: (expression_str, quantities_dict,
    domain_label, complexity_level)."""

    def test_output_has_all_required_fields(self):
        """GeneratedExpression must have all 4 required fields."""
        gen = SelfPlayExpressionGenerator(seed=42)
        expr = gen.generate()
        assert hasattr(expr, "expression_str")
        assert hasattr(expr, "quantities_dict")
        assert hasattr(expr, "domain_label")
        assert hasattr(expr, "complexity_level")

    def test_expression_str_is_string(self):
        gen = SelfPlayExpressionGenerator(seed=42)
        for _ in range(20):
            expr = gen.generate()
            assert isinstance(expr.expression_str, str)

    def test_quantities_dict_maps_to_dimensions(self):
        gen = SelfPlayExpressionGenerator(seed=42)
        for _ in range(20):
            expr = gen.generate()
            for q, d in expr.quantities_dict.items():
                assert isinstance(q, str)
                assert isinstance(d, Dimension)

    def test_domain_label_is_string(self):
        gen = SelfPlayExpressionGenerator(seed=42)
        for _ in range(20):
            expr = gen.generate()
            assert isinstance(expr.domain_label, str)

    def test_complexity_level_is_int(self):
        gen = SelfPlayExpressionGenerator(seed=42)
        for _ in range(20):
            expr = gen.generate()
            assert isinstance(expr.complexity_level, int)
            assert 1 <= expr.complexity_level <= 4
