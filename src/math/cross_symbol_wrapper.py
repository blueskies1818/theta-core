"""Cross-symbol proposer + beam search composition.

Architecture:
  1. Cross-symbol model proposes sub-expressions (a*b, c², E/p, etc.)
  2. Beam search composes them with operators + parentheses
  3. Cross-validation selects the best composed invariant

The model learns a simpler task: map [symbols] → useful sub-expressions.
Beam search handles structural composition — it can express any tree.
"""

from __future__ import annotations

from pathlib import Path
import random

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TREE_DECODER_PATH = PROJECT_ROOT / "checkpoints" / "math_self_play" / "tree_decoder.pt"

_tree_decoder: object | None = None
_tree_symbol_map: dict[str, int] = {}
_device = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

_BINARY_OPS = ["+", "-", "*", "/", "^"]
_SCALAR_CONSTANTS = ["0", "0.5", "1", "2", "-1"]


def _try_parse(expr: str) -> bool:
    """Check if expression is parseable."""
    try:
        from src.physics.evaluator import ExpressionEvaluator
        ExpressionEvaluator().parse(expr)
        return True
    except Exception:
        return False



def _load_tree_decoder() -> bool:
    """Load the tree-based AST decoder (Phase C proper)."""
    global _tree_decoder, _tree_symbol_map, _device
    if _tree_decoder is not None:
        return True
    if not TREE_DECODER_PATH.exists():
        return False
    try:
        ckpt = torch.load(TREE_DECODER_PATH, map_location=_device, weights_only=False)
        from src.math.tree_decoder import TreeDecoder
        _tree_symbol_map = ckpt["symbol_map"]
        config = ckpt.get("config", {"d_model": 128, "nhead": 4,
                         "num_encoder_layers": 3, "num_decoder_layers": 3,
                         "max_seq_len": 20})
        _tree_decoder = TreeDecoder(**config)
        _tree_decoder.load_state_dict(ckpt["model_state_dict"])
        _tree_decoder.to(_device)
        _tree_decoder.eval()
        return True
    except Exception:
        return False


def propose_sub_expressions(
    symbols: list[str],
    num_samples: int = 32,
    temperature: float = 1.5,
) -> list[str]:
    """Generate sub-expression candidates.

    Uses tree-based AST decoder merged with deterministic enumeration.
    Tree decoder proposes novel forms; deterministic fills in basics.
    """
    # Try tree-based AST decoder (Phase C — primary generator)
    tree_proposals: list[str] = []
    if _load_tree_decoder() and _tree_decoder is not None:
        from src.math.tree_decoder import MAX_VARS
        proposals = []
        seen = set()
        sym_ids = [_tree_symbol_map.get(s, 0) for s in symbols]
        while len(sym_ids) < MAX_VARS:
            sym_ids.append(0)
        src = torch.tensor([sym_ids[:MAX_VARS]], device=_device)

        for i in range(8):
            torch.manual_seed(hash(f"tree_{i}_{hash(tuple(symbols))}") % (2**31))
            results = _tree_decoder.generate(
                src, len(symbols), var_names=symbols,
                temperature=0.8 + i * 0.15, num_samples=5)
            for r in results:
                if r and r not in seen:
                    # Clean up extra parens and test parseability
                    clean = r.strip()
                    if clean.startswith("(") and clean.endswith(")") and clean.count("(") == 1:
                        clean = clean[1:-1]  # strip outer single parens
                    if clean and clean not in seen:
                        try:
                            from src.physics.evaluator import ExpressionEvaluator
                            ExpressionEvaluator().parse(clean)
                            seen.add(clean)
                            proposals.append(clean)
                        except Exception:
                            # Still include if it looks valid
                            if any(op in clean for op in "+-*/^"):
                                seen.add(clean)
                                proposals.append(clean)

        # Store tree proposals — merge with deterministic, don't return early
        tree_proposals = proposals

    # Fallback: deterministic enumeration
    exprs: list[str] = list(tree_proposals) if tree_proposals else []
    seen: set[str] = set(exprs)
    for s in symbols:
        if s not in seen:
            seen.add(s); exprs.append(s)
    for s in symbols:
        sq = f"{s}^2"
        if sq not in seen:
            seen.add(sq); exprs.append(sq)
    for i, a in enumerate(symbols):
        for b in symbols[i+1:]:
            for seed in [f"{a}*{b}", f"{b}*{a}", f"{a}/{b}", f"{b}/{a}",
                         f"{a}+{b}", f"{a}-{b}", f"{b}-{a}"]:
                if seed not in seen:
                    seen.add(seed); exprs.append(seed)
    return exprs


def _is_composite(expr: str) -> bool:
    """Does the expression need parentheses when used as a child?"""
    return any(op in expr for op in "+-") or (
        any(op in expr for op in "*/^") and len(expr) > 3
    )


def _parenthesize(expr: str) -> str:
    """Wrap in parentheses if composite."""
    if _is_composite(expr):
        return f"({expr})"
    return expr


def compose_restricted(
    seeds: list[str],
    quantities: dict,
    observations: list,
    evaluator,
    *,
    discovery_threshold: float = 0.90,
) -> str | None:
    """Restricted composition: square sub-expressions, subtract like terms.

    The proposer generates products, ratios, and powers.  This composer
    only does structurally clean operations:
      1. Square each sub-expression: (X)^2
      2. For same-dimension pairs, try subtraction: A^2 - B^2
      3. For same-dimension pairs, try addition: A^2 + B^2

    No unconstrained beam search — just the patterns that produce
    meaningful physics invariants (energy-momentum, spacetime interval).
    """
    from src.physics.dimensions import Dimension

    scalar_dim = Dimension.scalar()
    dim_lookup: dict[str, Dimension] = {}
    scored: dict[str, float] = {}

    def _dim_of(expr: str) -> Dimension | None:
        if expr in dim_lookup:
            return dim_lookup[expr]
        try:
            ast = evaluator.parse(expr)
        except Exception:
            dim_lookup[expr] = None
            return None
        from src.physics.evaluator import NumberNode, VarNode, FuncNode, BinOpNode
        def _dim(node) -> Dimension | None:
            if isinstance(node, NumberNode): return scalar_dim
            if isinstance(node, VarNode):
                d = quantities.get(node.name)
                return d if d else scalar_dim
            if isinstance(node, FuncNode): return scalar_dim
            if isinstance(node, BinOpNode):
                ld, rd = _dim(node.left), _dim(node.right)
                if ld is None or rd is None: return None
                try:
                    if node.op in ("+", "-"):
                        return ld if (isinstance(node.left, NumberNode) or
                                      ld.compatible_with(rd)) else None
                    elif node.op == "*": return ld * rd
                    elif node.op == "/": return ld / rd
                    elif node.op == "^":
                        if isinstance(node.right, NumberNode):
                            return ld ** float(node.right.value)
                except Exception: pass
            return None
        d = _dim(ast)
        dim_lookup[expr] = d
        return d

    def _score(expr: str) -> float:
        if expr in scored: return scored[expr]
        try:
            s = sum(evaluator.score(expr, o) for o in observations) / len(observations)
        except Exception:
            s = 0.0
        scored[expr] = s
        return s

    best_expr = ""
    best_score = 0.0

    # Score all seeds
    good_seeds: list[tuple[str, float, Dimension | None]] = []
    for seed in seeds:
        s = _score(seed)
        if s >= 0.3:
            d = _dim_of(seed)
            good_seeds.append((seed, s, d))
            if s > best_score:
                best_score, best_expr = s, seed

    # Step 1: Square each seed (unless already squared).
    # Also include the seed itself for subtraction.
    squared: list[tuple[str, float, Dimension | None]] = []
    for seed, s, d in good_seeds:
        if d is None:
            continue
        # Include the seed as-is for potential subtraction
        squared.append((seed, s, d))

        # Square it (unless already ends with ^2)
        if seed.endswith("^2"):
            continue
        sq_expr = f"({seed})^2" if _is_composite(seed) else f"{seed}^2"
        sq = _score(sq_expr)
        try:
            sq_dim = d ** 2
        except Exception:
            sq_dim = None
        if sq >= 0.3:
            squared.append((sq_expr, sq, sq_dim))
            if sq > best_score:
                best_score, best_expr = sq, sq_expr

    # Step 2: Subtract/Add same-dimension squares
    for i, (a_expr, a_score, a_dim) in enumerate(squared):
        if a_dim is None:
            continue
        for j, (b_expr, b_score, b_dim) in enumerate(squared):
            if j <= i:
                continue
            if b_dim is None or not a_dim.compatible_with(b_dim):
                continue

            # Avoid same-variable-set: a^2 - a^2 = 0
            import re
            va = set(re.findall(r'\b[a-zA-Z_]\w*\b', a_expr))
            vb = set(re.findall(r'\b[a-zA-Z_]\w*\b', b_expr))
            va -= {"sin", "cos", "sqrt", "exp", "log", "abs", "tan"}
            vb -= {"sin", "cos", "sqrt", "exp", "log", "abs", "tan"}
            if va == vb:
                continue
            # Avoid subset: (c*t)^2 - c^2*t^2 is fine in physics but
            # (E+E)^2 - E^2 is degenerate
            if va.issubset(vb) or vb.issubset(va):
                continue

            for op in ["-", "+"]:
                child = f"{a_expr}{op}{b_expr}"
                s = _score(child)
                if s > best_score:
                    best_score, best_expr = s, child

    return best_expr if best_score >= discovery_threshold else None


def _complexity(expr: str) -> int:
    return (expr.count("+") + expr.count("-") + expr.count("*") +
            expr.count("/") + expr.count("^") + expr.count("("))


def _train_test_split(observations: list, test_frac: float = 0.3, seed: int = 42):
    if len(observations) < 4:
        return list(observations), []
    rng = random.Random(seed)
    shuffled = list(observations)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_frac))
    return shuffled[n_test:], shuffled[:n_test]


def cross_symbol_template_search(
    quantities: dict,
    observations: list,
    discovery_threshold: float = 0.90,
):
    """Proposer + beam search composition + cross-validation selection.

    1. Proposer generates sub-expression candidates.
    2. Beam search composes them with operators + parentheses.
    3. Cross-validation selects the best invariant.
    """
    from src.physics.evaluator import ExpressionEvaluator
    from src.physics.search import SearchResult

    symbols = sorted(quantities.keys())
    evaluator = ExpressionEvaluator()

    # ── Load semantic memory ──────────────────────────────────────
    try:
        from src.memory import load_memory, score_candidate as memory_score
        from src.memory import update_memory as mem_update, save_memory as mem_save
        _mem = load_memory()
        _has_memory = True
    except Exception:
        _mem = {}
        memory_score = lambda e, m: 0.0
        _has_memory = False

    # ── Step 1: Propose sub-expressions ────────────────────────────
    proposed = propose_sub_expressions(symbols, num_samples=32, temperature=1.5)

    # Filter: must use ONLY input symbols (no hallucinated symbols from training).
    # Accept operators, numbers, and the input symbols — reject everything else.
    # Also reject proposals that redundantly use the same variable (X+X, X*X/X)
    # — these lead to trivially constant degenerate forms in beam search.
    valid_proposals = []
    for p in proposed:
        import re
        all_tokens = re.findall(r'[a-zA-Z_]\w*', p)
        all_tokens = [t for t in all_tokens
                      if t not in {"sin", "cos", "sqrt", "exp", "log", "abs", "tan"}
                      and not t.replace('.', '').isdigit()]
        tokens = set(all_tokens)
        if not tokens:
            continue
        if not tokens.issubset(set(symbols)):
            continue
        # Reject if any variable appears redundantly (X+X, X*X, X-X, X/X)
        var_counts = {}
        for t in all_tokens:
            var_counts[t] = var_counts.get(t, 0) + 1
        if any(c > 1 for c in var_counts.values()):
            continue
        valid_proposals.append(p)

    # ── Step 2: Score proposals on train ────────────────────────────
    train_obs, test_obs = _train_test_split(observations, test_frac=0.3)
    has_test = len(test_obs) > 0

    def score_on(expr: str, obs_list: list) -> float:
        if not obs_list:
            return 0.0
        try:
            scores = [evaluator.score(expr, obs) for obs in obs_list]
            return sum(scores) / len(scores)
        except Exception:
            return 0.0

    # Score proposals, keep top-scoring ones
    scored_proposals = []
    for p in valid_proposals:
        s = score_on(p, train_obs)
        if s >= 0.3:
            scored_proposals.append((s, p))

    # Apply semantic memory scoring: known product pairs get bonus,
    # novel pairs get penalty.  This biases the search toward
    # relationships previously observed across discoveries.
    if _has_memory:
        scored_proposals = [
            (s + memory_score(p, _mem), p) for s, p in scored_proposals
        ]

    scored_proposals.sort(key=lambda x: -x[0])

    # ── Apply neural seed scorer (plastic) ──
    # Boost proposals that the model recognizes from prior experience.
    # Plastic adaptation accumulates over experiments.
    try:
        from src.math.seed_scorer import score_seeds as model_score_seeds
        from src.math.plastic_seed_scorer import score_seed as plastic_score_seed, update_plastic as plastic_update
        all_exprs = [p for _, p in scored_proposals]
        model_scored = dict(model_score_seeds(symbols, all_exprs))
        # Blend frozen + plastic scores
        scored_proposals = [
            (s + 0.3 * plastic_score_seed(symbols, p), p)
            for s, p in scored_proposals
        ]
        scored_proposals.sort(key=lambda x: -x[0])
    except Exception:
        pass  # model not available, use constancy-only scoring

    # Take top proposals as seeds for composition, filtered by memory.
    # Seeds that score well OR match known product pairs are kept.
    # Nuisance variable seeds (low score, no memory) are dropped.
    top_seeds = []
    for s, p in scored_proposals[:40]:  # consider more candidates
        keep = s >= 0.5  # strong constancy signal
        if not keep and _has_memory:
            # Check if this seed uses a known product pair
            mem_bonus = memory_score(p, _mem)
            keep = mem_bonus > 0  # memory recognizes this pattern
        if keep:
            top_seeds.append(p)
        if len(top_seeds) >= 8:
            break

    # Seed with combinatoric building blocks: products, ratios, squares.
    # Always include all product/ratio pairs as building blocks — they're
    # essential for composition (P*V/T requires P*V as seed even if P*V
    # varies).  Individual symbols and squares are filtered by score.
    from itertools import combinations
    for a, b in combinations(symbols, 2):
        for seed in [f"{a}*{b}", f"{b}*{a}", f"{a}/{b}", f"{b}/{a}"]:
            if seed not in top_seeds:
                top_seeds.append(seed)
    # Squares of product seeds: (a*b)^2 enables (c*t)^2 - x^2 composition.
    for a, b in combinations(symbols, 2):
        for base in [f"{a}*{b}", f"{b}*{a}"]:
            seed = f"({base})^2"
            if seed in top_seeds:
                continue
            s = score_on(seed, train_obs)
            keep = s >= 0.3  # lower bar for composed seeds
            if not keep and _has_memory:
                keep = memory_score(seed, _mem) > 0
            if keep:
                top_seeds.append(seed)
    for s in symbols:
        for seed in [s, f"{s}^2"]:
            if seed in top_seeds:
                continue
            sc = score_on(seed, train_obs)
            keep = sc >= 0.5
            if not keep and _has_memory:
                keep = memory_score(seed, _mem) > 0
            if keep:
                top_seeds.append(seed)

    # ── Step 3: Tree beam search composition ───────────────────────
    from src.math.tree_beam_search import tree_beam_search

    composed = tree_beam_search(
        top_seeds, quantities, train_obs, evaluator,
        discovery_threshold=discovery_threshold,
    )

    # Also check if any raw proposal is good enough
    best_expr = composed
    best_train = 0.0

    if composed:
        best_train = score_on(composed, train_obs)
    else:
        # Fall back to best raw proposal
        for s, p in scored_proposals:
            if s >= discovery_threshold:
                best_expr = p
                best_train = s
                break

    if not best_expr:
        return None

    # ── Step 4: Cross-validation ────────────────────────────────────
    if has_test:
        test_score = score_on(best_expr, test_obs)
        if test_score < discovery_threshold:
            return None
        if best_train - test_score > 0.10:
            return None
        final_score = test_score
    else:
        final_score = best_train

    if final_score < discovery_threshold:
        return None

    train_constancies = []
    for obs in observations:
        try:
            train_constancies.append(evaluator.score(best_expr, obs))
        except Exception:
            train_constancies.append(0.0)

    # ── Memory: learn only from winning expression ──────────────────
    # Only store pairs from expressions that crossed the discovery
    # threshold — not from all high-scoring candidates.  This prevents
    # memory pollution from nuisance variables that are constant in
    # their data but not part of genuine invariants.
    if _has_memory and final_score >= discovery_threshold:
        try:
            mem_update(_mem, best_expr, evaluator=evaluator,
                       observations=observations, domain='auto')
            mem_save(_mem)
        except Exception:
            pass

    # ── Plastic update: learn from this discovery ──
    try:
        from src.math.plastic_seed_scorer import update_plastic as plastic_update_fn
        outcome = 1.0 if final_score >= discovery_threshold else -0.1
        plastic_update_fn(outcome, symbols, best_expr)
    except Exception:
        pass

    return SearchResult(
        expression=best_expr, score=final_score, depth=1,
        expansions=len(valid_proposals), train_constancies=train_constancies,
    )
