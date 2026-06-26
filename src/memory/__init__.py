"""Structural memory — learns variable roles and relationships.

Tracks:
  1. Variable profiles: which operations, positions, co-occurrences
  2. Co-occurrence counts: which pairs appear together across discoveries  
  3. Structural templates: product, ratio, squared-diff, sum

Scoring rewards expressions that use historically-active variables
and penalizes those with no co-occurrence history.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MEMORY_PATH = PROJECT_ROOT / "data" / "semantic_memory.json"
EMPTY_PATH = PROJECT_ROOT / "data" / "semantic_memory_empty.json"


# ═══════════════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════════════

def empty_memory() -> dict[str, Any]:
    return {
        "variables": {},       # {name: {profile}}
        "co_occurrence": {},   # {"a*b": {count, confidence}}
        "templates": {},       # {"ratio": count, "product": count, ...}
        "constants": {},       # {name: {confidence, sources}}
        "discovery_count": 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Variable profile extraction
# ═══════════════════════════════════════════════════════════════════════════

def _classify_structure(expr: str) -> str:
    """Classify the structural pattern of an expression."""
    ops = set()
    for c in expr:
        if c in "+-": ops.add("add")
        if c in "*/": ops.add("mul_div")
        if c == "^": ops.add("pow")
        if c == "(": ops.add("nested")
    if "nested" in ops and "mul_div" in ops and "add" in ops:
        return "nested_fraction"
    if "pow" in ops and "add" in ops:
        return "squared_diff"
    if "/" in expr and "+" not in expr and "-" not in expr:
        return "ratio"
    if "*" in expr and "+" not in expr and "-" not in expr:
        return "product"
    if "add" in ops and "pow" not in ops:
        return "sum"
    if "pow" in ops:
        return "power"
    return "atomic"


def extract_variable_profile(expr: str) -> dict[str, dict]:
    """Extract per-variable roles from an expression.

    Returns: {var_name: {operations, positions, co_occurs_with}}
    """
    vars_in = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
    vars_in -= {"sin", "cos", "sqrt", "exp", "log", "abs", "tan"}
    vars_in = {v for v in vars_in if not v.replace('.', '').isdigit()}

    profiles = {}
    for var in vars_in:
        ops = set()
        positions = set()

        # Check what operations this variable participates in
        # Find var in context: what's around it?
        idx = 0
        while True:
            idx = expr.find(var, idx)
            if idx == -1:
                break
            # Look at surrounding characters
            before = expr[idx-1] if idx > 0 else ""
            after = expr[idx+len(var):idx+len(var)+1] if idx+len(var) < len(expr) else ""

            if before == "/" or (before == "(" and idx > 1 and expr[idx-2] == "/"):
                positions.add("denominator")
            elif before == "*":
                positions.add("factor")
            elif before in "+-":
                positions.add("term")
            if after == "^":
                ops.add("power_base")
            if before == "^":
                ops.add("power_exponent")
            if before in "+-":
                ops.add("additive")
            if before in "*/" or after in "*/" or (before == "(" and expr[idx-2:idx] in ("*(", "/(")):
                ops.add("multiplicative")

            idx += len(var)

        # Co-occurring variables
        co_occurs = vars_in - {var}

        profiles[var] = {
            "operations": sorted(ops) if ops else ["unknown"],
            "positions": sorted(positions) if positions else ["unknown"],
            "co_occurs_with": sorted(co_occurs),
        }

    return profiles


# ═══════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════

def structural_score(expr: str, memory: dict[str, Any]) -> float:
    """Score an expression based on structural memory.

    Rewards: variables with discovery history, known co-occurring pairs,
             familiar structural templates.

    Penalizes: variables never seen in any discovery, pairs that never
               co-occurred, novel structural patterns.
    """
    if not memory or memory.get("discovery_count", 0) == 0:
        return 0.0  # cold start — no bias

    score = 0.0
    var_profiles = extract_variable_profile(expr)
    var_mem = memory.get("variables", {})
    co_occ = memory.get("co_occurrence", {})
    templates = memory.get("templates", {})

    all_vars = set(var_profiles.keys())

    # ── Per-variable: activity bonus ──────────────────────────────
    for var in all_vars:
        profile = var_mem.get(var, {})
        appearances = profile.get("appearances", 0)
        if appearances == 0:
            score -= 0.10  # never seen this variable in any discovery
        elif appearances >= 3:
            score += 0.03  # well-established variable
        elif appearances >= 1:
            score += 0.01  # seen before

    # ── Pair co-occurrence: known pairs get bonus ─────────────────
    for var in all_vars:
        for other in all_vars:
            if var >= other:
                continue
            pair_key = f"{var}*{other}" if var < other else f"{other}*{var}"
            pair_data = co_occ.get(pair_key, {})
            count = pair_data.get("count", 0)
            if count >= 1:
                score += 0.05  # any co-occurrence is informative
            elif appearances_check(var, other, var_mem):
                score -= 0.08  # both seen, never together — likely noise

    # ── Structural template: familiar patterns get bonus ──────────
    struct = _classify_structure(expr)
    template_count = templates.get(struct, 0)
    if template_count >= 2:
        score += 0.04  # familiar pattern
    elif template_count == 0 and struct not in ("atomic", "unknown"):
        score -= 0.02  # novel pattern

    return max(-0.30, min(0.30, score))  # clamp


def appearances_check(var_a: str, var_b: str, var_mem: dict) -> bool:
    """Check if both variables have been seen in at least one discovery."""
    return (var_mem.get(var_a, {}).get("appearances", 0) > 0 and
            var_mem.get(var_b, {}).get("appearances", 0) > 0)


# ═══════════════════════════════════════════════════════════════════════════
# Update
# ═══════════════════════════════════════════════════════════════════════════

def update_structural_memory(
    memory: dict[str, Any],
    expr: str,
    evaluator=None,
    observations=None,
    domain: str = "auto",
    score: float = 0.0,
) -> dict[str, Any]:
    """Update structural memory with a verified discovery."""

    # ── Variable profiles ────────────────────────────────────────
    profiles = extract_variable_profile(expr)
    var_mem = memory.setdefault("variables", {})

    for var, profile in profiles.items():
        v = var_mem.setdefault(var, {
            "appearances": 0,
            "operations": defaultdict(int),
            "positions": defaultdict(int),
            "co_occurs_with": defaultdict(int),
        })
        v["appearances"] += 1
        for op in profile["operations"]:
            v["operations"][op] += 1
        for pos in profile["positions"]:
            v["positions"][pos] += 1
        for co in profile.get("co_occurs_with", []):
            v["co_occurs_with"][co] += 1

    # ── Co-occurrence pairs ──────────────────────────────────────
    co_occ = memory.setdefault("co_occurrence", {})
    all_vars = sorted(set(profiles.keys()))
    for i in range(len(all_vars)):
        for j in range(i + 1, len(all_vars)):
            a, b = all_vars[i], all_vars[j]
            key = f"{a}*{b}" if a < b else f"{b}*{a}"
            existing = co_occ.setdefault(key, {"count": 0, "confidence": 0.0})
            existing["count"] += 1
            existing["confidence"] = min(1.0, existing["confidence"] + 0.2)

    # ── Structural template ──────────────────────────────────────
    templates = memory.setdefault("templates", {})
    struct = _classify_structure(expr)
    templates[struct] = templates.get(struct, 0) + 1

    # ── Constants ────────────────────────────────────────────────
    constants = memory.setdefault("constants", {})
    if evaluator and observations:
        for var in all_vars:
            if _is_constant(var, evaluator, observations):
                c = constants.setdefault(var, {"confidence": 0.0, "sources": []})
                c["confidence"] = min(1.0, c["confidence"] + 0.2)
                if domain not in c["sources"]:
                    c["sources"].append(domain)

    memory["discovery_count"] = memory.get("discovery_count", 0) + 1
    return memory


def _is_constant(var_name: str, evaluator, observations) -> bool:
    try:
        values = []
        for obs in observations:
            for ts in obs.timesteps:
                if var_name in ts:
                    values.append(ts[var_name])
                elif var_name in obs.parameters:
                    values.append(obs.parameters[var_name])
        if len(values) < 2:
            return False
        mean = sum(values) / len(values)
        if abs(mean) < 1e-30:
            return True
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        rel_std = (variance ** 0.5) / abs(mean)
        return rel_std < 0.05
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Persistence (compatible with existing API)
# ═══════════════════════════════════════════════════════════════════════════

def load_memory(path: Path | None = None) -> dict[str, Any]:
    p = path or MEMORY_PATH
    if p.exists():
        try:
            with open(p) as f:
                mem = json.load(f)
                # Ensure new fields exist
                for key in ["variables", "co_occurrence", "templates", "constants"]:
                    mem.setdefault(key, {} if key != "discovery_count" else 0)
                return mem
        except (json.JSONDecodeError, OSError):
            pass
    return empty_memory()


def save_memory(memory: dict[str, Any], path: Path | None = None) -> None:
    p = path or MEMORY_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(memory, f, indent=2)


def reset_memory() -> None:
    save_memory(empty_memory())


def memory_summary(memory: dict[str, Any]) -> str:
    lines = [f"Discoveries: {memory.get('discovery_count', 0)}"]

    var_mem = memory.get("variables", {})
    if var_mem:
        lines.append("\nVariables (>0 appearances):")
        for var, profile in sorted(var_mem.items()):
            apps = profile.get("appearances", 0)
            if apps > 0:
                ops = dict(profile.get("operations", {}))
                top_ops = sorted(ops.keys(), key=lambda k: ops[k], reverse=True)[:3]
                lines.append(f"  {var}: seen={apps}  ops={top_ops}")

    co_occ = memory.get("co_occurrence", {})
    if co_occ:
        lines.append("\nCo-occurrence pairs (count ≥2):")
        for pair, data in sorted(co_occ.items()):
            if data.get("count", 0) >= 2:
                lines.append(f"  {pair}: count={data['count']} conf={data['confidence']:.2f}")

    templates = memory.get("templates", {})
    if templates:
        lines.append("\nTemplates:")
        for tmpl, count in sorted(templates.items(), key=lambda x: -x[1]):
            lines.append(f"  {tmpl}: {count}")

    constants = memory.get("constants", {})
    if constants:
        lines.append("\nConstants:")
        for var, info in sorted(constants.items()):
            lines.append(f"  {var}: conf={info['confidence']:.2f}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Legacy API compatibility
# ═══════════════════════════════════════════════════════════════════════════

def extract_relations(expr: str, evaluator=None, observations=None) -> dict:
    """Legacy: returns product_pairs for old code compatibility."""
    profiles = extract_variable_profile(expr)
    pairs = {}
    for var, profile in profiles.items():
        for co in profile.get("co_occurs_with", []):
            a, b = (var, co) if var < co else (co, var)
            pairs[f"{a}*{b}"] = max(pairs.get(f"{a}*{b}", 0), 0.7)
    return {"product_pairs": pairs, "isolated": set(profiles.keys()),
            "denominators": set(), "constants": {}}


def score_candidate(expr: str, memory: dict[str, Any]) -> float:
    """Legacy: returns structural score."""
    return structural_score(expr, memory)


def update_memory(memory: dict[str, Any], expr: str, evaluator=None,
                  observations=None, domain: str = "auto",
                  score: float = 0.0) -> dict[str, Any]:
    """Legacy: delegates to structural update."""
    return update_structural_memory(memory, expr, evaluator, observations, domain, score)
