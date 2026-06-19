"""Quick test of domain normalization and matching helpers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.explorer.gnn_best_first_search import GNNBestFirstSearch


def test_normalize_domain():
    bf = GNNBestFirstSearch
    assert bf._normalize_domain('Algebra') == 'algebra'
    assert bf._normalize_domain('NumberTheory') == 'numbertheory'
    assert bf._normalize_domain('Analysis/Calculus') == 'analysis/calculus'
    assert bf._normalize_domain('number_theory') == 'numbertheory'
    assert bf._normalize_domain('  Algebra  ') == 'algebra'


def test_domain_matches_exact():
    bf = GNNBestFirstSearch
    assert bf._domain_matches('Algebra', 'algebra') is True
    assert bf._domain_matches('algebra', 'Algebra') is True
    assert bf._domain_matches('Analysis', 'analysis') is True


def test_domain_matches_prefix():
    bf = GNNBestFirstSearch
    assert bf._domain_matches('Algebra/Polynomial', 'algebra') is True
    assert bf._domain_matches('Analysis/Normed', 'analysis') is True
    assert bf._domain_matches('Analysis/Calculus/Deriv', 'analysis') is True


def test_domain_matches_substring():
    bf = GNNBestFirstSearch
    assert bf._domain_matches('NumberTheory', 'number_theory') is True
    assert bf._domain_matches('LinearAlgebra', 'algebra') is True


def test_domain_matches_reject():
    bf = GNNBestFirstSearch
    assert bf._domain_matches('CategoryTheory', 'algebra') is False
    assert bf._domain_matches('Data', 'algebra') is False


def test_domain_matches_cross_bundle():
    bf = GNNBestFirstSearch
    assert bf._domain_matches('LinearAlgebra', 'algebra') is True
    assert bf._domain_matches('Algebra/Order', 'algebra') is True
    assert bf._domain_matches('Algebra/Polynomial', 'algebra') is True
    assert bf._domain_matches('MeasureTheory', 'analysis') is True
    assert bf._domain_matches('Analysis/Calculus', 'analysis') is True
    assert bf._domain_matches('SetTheory', 'logic') is True
    assert bf._domain_matches('Order', 'logic') is True
    # Tighter bundles: RingTheory/GroupTheory NOT bundled with algebra
    assert bf._domain_matches('RingTheory', 'algebra') is False
    assert bf._domain_matches('GroupTheory', 'algebra') is False


if __name__ == '__main__':
    tests = [
        ('normalize_domain', test_normalize_domain),
        ('domain_matches_exact', test_domain_matches_exact),
        ('domain_matches_prefix', test_domain_matches_prefix),
        ('domain_matches_substring', test_domain_matches_substring),
        ('domain_matches_reject', test_domain_matches_reject),
        ('domain_matches_cross_bundle', test_domain_matches_cross_bundle),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'  PASS: {name}')
            passed += 1
        except AssertionError as e:
            print(f'  FAIL: {name} - {e}')
        except Exception as e:
            print(f'  ERROR: {name} - {e}')
    print(f'\n{passed}/{len(tests)} passed')
