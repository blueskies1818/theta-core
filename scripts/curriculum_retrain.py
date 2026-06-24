#!/usr/bin/env python3
"""Curriculum retraining: progressive learning from solved examples.

A physicist doesn't start with general relativity. They master classical
mechanics, then use that foundation to understand relativity. This script
implements the same progressive learning:

1. Phase 1: Train on pre-1905 self-play data. Run era gate at 1905.
   Collect all SUCCESSFULLY DISCOVERED invariants.

2. Phase 2: Add those discovered invariants to the training data as
   "solved examples." The system now knows: "these expressions ARE
   invariants — use them as building blocks for more complex ones."

3. Phase 3: Retrain template generator on expanded data (original +
   solved examples). The generator now has templates for both:
   - Simple structural patterns (from abstract self-play)
   - Proven invariants (from era gate verification)

4. Phase 4: Run era gate again. The system should discover MORE because
   it can build on previously-verified invariants.

5. Repeat: each generation's discoveries feed the next generation's
   training data.

METRICS TRACKED:
- Discovery rate per generation
- Whether the system discovers invariants at generation N that it
  couldn't at generation N-1
- Whether the system synthesizes compound invariants from simpler ones

USAGE:
  python scripts/curriculum_retrain.py --generations 3
  python scripts/curriculum_retrain.py --generations 5 --era-cutoff 1905
  python scripts/curriculum_retrain.py --generations 3 --output data/curriculum_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch

# ── Imports from project modules ───────────────────────────────────────────
from scripts.training.train_self_play_template import (
    load_self_play_data,
    train_domain_generator_from_self_play,
    save_combined_checkpoint,
)
from scripts.training.collect_self_play_data import generate_training_data


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

PRE_1905_DOMAINS = ["gravity", "spring", "em", "thermal"]
DEFAULT_SEED = 42
DEFAULT_DATA = "data/self_play_training.jsonl"
CURRICULUM_DATA = "data/curriculum_training.jsonl"
ERA_GATE_SCRIPT = "scripts/spacetime_era_gate.py"
TEMPLATE_TRAINING_SCRIPT = "scripts/training/train_self_play_template.py"

# Map scenario IDs to their closest pre-1905 domain for curriculum learning
SCENARIO_TO_CLASSICAL_DOMAIN: dict[str, str] = {
    # Relativistic scenarios → gravity (mechanics/energy concepts)
    "muon_time_dilation": "gravity",
    "velocity_addition": "gravity",
    "relativistic_momentum": "gravity",
    "length_contraction": "gravity",
    "doppler_shift": "gravity",
    "twin_paradox": "gravity",
    "mass_energy": "gravity",
    "spacetime_interval": "gravity",
    # Quantum scenarios → em (field/wave concepts)
    "qed_fine_structure": "em",
    "compton_scattering": "em",
    "dirac_spinor_norm": "em",
    "qcd_asymptotic_freedom": "em",
    "electroweak_mixing": "em",
    "higgs_mechanism": "em",
    "neutrino_oscillation": "em",
}

# Quantity mapping for each scenario (needed for training examples)
SCENARIO_QUANTITIES: dict[str, list[str]] = {
    "muon_time_dilation": ["c", "t", "x", "v", "gamma", "tau"],
    "velocity_addition": ["c", "t", "x", "v1", "v2"],
    "relativistic_momentum": ["c", "E", "p", "m", "v", "gamma"],
    "length_contraction": ["c", "t", "x", "v", "gamma"],
    "doppler_shift": ["c", "t", "x", "v", "gamma"],
    "twin_paradox": ["c", "t", "x", "v", "gamma"],
    "mass_energy": ["c", "E", "m", "t", "x", "v", "gamma"],
    "spacetime_interval": ["c", "t", "x"],
    "qed_fine_structure": ["e", "hbar", "c", "alpha"],
    "compton_scattering": ["dlambda", "theta", "lambda_c"],
    "dirac_spinor_norm": ["psi_re", "psi_im"],
    "qcd_asymptotic_freedom": ["alpha_s", "Q", "log_Q"],
    "electroweak_mixing": ["g", "gp"],
    "higgs_mechanism": ["phi", "v"],
    "neutrino_oscillation": ["phase", "E", "L"],
}

# Post-1905 scenario IDs — used to identify "new knowledge" discoveries
POST_1905_SCENARIO_IDS: set[str] = {
    "muon_time_dilation", "velocity_addition", "relativistic_momentum",
    "length_contraction", "doppler_shift", "twin_paradox",
    "mass_energy", "spacetime_interval",
    "qed_fine_structure", "compton_scattering", "dirac_spinor_norm",
    "qcd_asymptotic_freedom", "electroweak_mixing",
    "higgs_mechanism", "neutrino_oscillation",
}


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DiscoveredInvariant:
    """A single successfully discovered invariant with metadata."""
    expression: str
    quantities: list[str]
    scenario_id: str
    scenario_name: str
    constancy: float
    generation: int
    domain: str

    def expression_key(self) -> str:
        """Normalized key for deduplication."""
        return self.expression.replace(" ", "").lower()


@dataclass
class GenerationResult:
    """Results from one generation of curriculum retraining."""
    generation: int
    training_examples: int
    discoveries: list[DiscoveredInvariant]
    total_scenarios: int
    spacetime_verified: int
    any_discovery: int
    new_discoveries: int = 0
    compound_discoveries: int = 0
    timing_seconds: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Core curriculum functions
# ═══════════════════════════════════════════════════════════════════════════

def prepare_initial_data(
    n_examples: int = 12000,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> Path:
    """Ensure initial pre-1905 self-play training data exists.

    Returns path to the base training data JSONL file.
    """
    data_path = _PROJECT_ROOT / DEFAULT_DATA
    if data_path.exists() and not force:
        print(f"  Using existing training data: {data_path}")
        return data_path

    print(f"  Generating {n_examples} pre-1905 self-play training examples...")
    examples = generate_training_data(
        n_examples=n_examples,
        levels=[1, 2, 3],
        seed=seed,
        output_path=str(data_path),
    )
    print(f"  Generated {len(examples)} examples → {data_path}")
    return data_path


def prepare_curriculum_data(
    base_data_path: Path,
    solved_examples: list[dict],
    generation: int,
) -> Path:
    """Create curriculum training data by merging base data with solved examples.

    Args:
        base_data_path: Path to the original self-play training data.
        solved_examples: List of training example dicts from discovered invariants.
        generation: Current generation number (for output path naming).

    Returns:
        Path to the merged curriculum training data JSONL.
    """
    # Load base data
    base_examples: list[dict] = []
    if base_data_path.exists():
        with open(base_data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    base_examples.append(json.loads(line))

    # Deduplicate: skip solved examples whose expression is already in base
    base_expressions: set[str] = {
        ex.get("expression", "").replace(" ", "").lower()
        for ex in base_examples
    }

    new_examples: list[dict] = []
    for ex in solved_examples:
        expr_key = ex.get("expression", "").replace(" ", "").lower()
        if expr_key and expr_key not in base_expressions:
            new_examples.append(ex)
            base_expressions.add(expr_key)

    merged = base_examples + new_examples

    # Write merged data
    output_path = _PROJECT_ROOT / CURRICULUM_DATA
    with open(output_path, "w") as f:
        for ex in merged:
            f.write(json.dumps(ex) + "\n")

    print(f"  Curriculum data: {len(base_examples)} base + "
          f"{len(new_examples)} new = {len(merged)} total → {output_path}")
    return output_path


def train_template_generators_on(data_path: Path) -> dict[str, Any]:
    """Train template generators on the given training data.

    Runs train_self_play_template.py as a subprocess to ensure
    consistent training with the rest of the pipeline.

    Returns training stats dict.
    """
    print(f"\n  Training template generators on: {data_path.name}")

    start = time.time()
    # Use the training script directly via import for programmatic control
    examples = load_self_play_data(str(data_path), max_per_domain=2000)

    if not examples:
        print("  WARNING: No training examples loaded!")
        return {"error": "no_examples", "domains": {}}

    domain_counts: dict[str, int] = {}
    for ex in examples:
        from scripts.training.train_self_play_template import _normalize_domain
        d = _normalize_domain(ex.get("domain", ""))
        domain_counts[d] = domain_counts.get(d, 0) + 1
    print(f"  Domain distribution: {dict(sorted(domain_counts.items()))}")

    checkpoint_dir = _PROJECT_ROOT / "checkpoints"
    device = torch.device("cpu")
    torch.set_num_threads(4)
    torch.manual_seed(DEFAULT_SEED)

    models: dict = {}
    all_stats: dict = {}

    for domain in PRE_1905_DOMAINS:
        try:
            model, stats = train_domain_generator_from_self_play(
                examples=examples,
                domain=domain,
                checkpoint_dir=checkpoint_dir,
                epochs=50,
                lr=1e-3,
                batch_size=8,
                device=device,
                d_model=48,
                nhead=2,
            )
            models[domain] = model
            all_stats[domain] = stats
        except Exception as exc:
            print(f"  ERROR training {domain}: {exc}")
            all_stats[domain] = {"error": str(exc)}

    if models:
        save_combined_checkpoint(models, checkpoint_dir)

    elapsed = time.time() - start
    total_params = sum(
        m.count_parameters() for m in models.values() if hasattr(m, "count_parameters")
    )
    print(f"  Training complete in {elapsed:.0f}s ({total_params:,} params)")

    return {
        "domains": list(models.keys()),
        "total_params": total_params,
        "training_time_s": elapsed,
        "stats": all_stats,
    }


def run_era_gate(cutoff: int) -> dict:
    """Run the era gate as a subprocess and return parsed results."""
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / ERA_GATE_SCRIPT),
        "--era-cutoff", str(cutoff),
    ]

    print(f"  Running era gate (cutoff={cutoff})...")
    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=1200,  # 20 minutes
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  Era gate FAILED (exit {result.returncode}) after {elapsed:.0f}s")
        # Show last portion of stderr
        stderr_tail = result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr
        print(f"  STDERR tail:\n{stderr_tail}")
        return {"error": f"exit_code={result.returncode}", "scenarios": [], "spacetime_verified": 0}

    print(f"  Era gate completed in {elapsed:.0f}s")

    # Parse the results JSON
    results_path = (
        _PROJECT_ROOT / "data" / "spacetime_era_gate_results.json"
        if cutoff == 1905 else
        _PROJECT_ROOT / "data" / f"era_gate_{cutoff}_results.json"
    )

    if not results_path.exists():
        print(f"  WARNING: Results file not found: {results_path}")
        return {"error": "results_not_found", "scenarios": [], "spacetime_verified": 0}

    with open(results_path) as f:
        return json.load(f)


def _is_noise_expression(expression: str) -> bool:
    """Check if an expression looks like search noise rather than a real invariant.

    Filters out patterns like '-1-qty+qty+2' and excessively long expressions
    that are search artifacts rather than meaningful physics invariants.
    """
    expr = expression.strip()

    # Excessively long expressions are likely noise
    if len(expr) > 80:
        return True

    # Pattern: repeating "+-1-qty+qty+2" or "-1-qty+qty+2" structures
    # These are algebraic identity noise from search
    if re.search(r'[+-]-1-', expr) and expr.count('+') > 5:
        return True

    # Pattern: a quantity immediately negated: "-qty+qty" (identity noise)
    # Example: "-alpha_s+alpha_s", "-psi_im+psi_im"
    if re.search(r'-(\w+)\+\1\+', expr):
        return True

    # Pattern: expressions that are just algebraic identities with numbers
    # Check for quantity names that appear both positively and negatively
    parts = [p for p in re.split(r'[+\-*/()^]', expr) if p.strip()]
    quantity_names = [
        p for p in parts
        if p and not p.replace('.', '').replace('e', '').lstrip('-').isdigit()
    ]
    if not quantity_names:
        return True

    # Check if all quantities appear in self-canceling pairs
    # (e.g., "phi" appears multiple times in "-phi+phi-phi+phi")
    qty_counts = Counter(quantity_names)
    if len(quantity_names) >= 4 and all(c >= 2 for c in qty_counts.values()):
        return True

    return False


def extract_discoveries(
    era_results: dict,
    generation: int,
) -> list[DiscoveredInvariant]:
    """Extract successfully discovered invariants from era gate results.

    An invariant is "successfully discovered" if:
    - spacetime_verified=True (relativistic scenarios passed gate)
    - OR (discovered=True AND constancy >= 0.95 AND not noise)
    """
    discoveries: list[DiscoveredInvariant] = []

    for scenario in era_results.get("scenarios", []):
        is_discovered = scenario.get("discovered", False)
        is_verified = scenario.get("spacetime_verified", False)

        if not is_discovered and not is_verified:
            continue

        expression = scenario.get("best_expression")
        if not expression or expression == "N/A":
            continue

        constancy = scenario.get("best_constancy", 0.0)

        # Quality filters:
        # 1. Noise expressions are always excluded
        if _is_noise_expression(expression):
            continue
        # 2. Not spacetime-verified → require high constancy
        if not is_verified and constancy < 0.95:
            continue

        scenario_id = scenario.get("scenario_id", "unknown")
        domain = SCENARIO_TO_CLASSICAL_DOMAIN.get(scenario_id, "gravity")
        quantities = SCENARIO_QUANTITIES.get(scenario_id, [])

        discoveries.append(DiscoveredInvariant(
            expression=expression,
            quantities=quantities,
            scenario_id=scenario_id,
            scenario_name=scenario.get("scenario_name", scenario_id),
            constancy=constancy,
            generation=generation,
            domain=domain,
        ))

    return discoveries


def discovery_to_training_example(
    d: DiscoveredInvariant,
) -> dict:
    """Convert a discovered invariant to a self-play training example."""
    return {
        "quantities": sorted(set(d.quantities)),
        "domain": d.domain,
        "expression": d.expression,
        "complexity_level": 3,
        "source": f"curriculum_gen{d.generation}",
        "scenario": d.scenario_id,
    }


def detect_new_discoveries(
    current: list[DiscoveredInvariant],
    previous: list[DiscoveredInvariant],
) -> int:
    """Count discoveries in current that weren't in previous generation.

    A discovery is "new" if its expression wasn't discovered before
    AND its scenario wasn't solved before.
    """
    prev_exprs: set[str] = {d.expression_key() for d in previous}
    prev_scenarios: set[str] = {d.scenario_id for d in previous}

    new_count = 0
    for d in current:
        if d.expression_key() not in prev_exprs and d.scenario_id not in prev_scenarios:
            new_count += 1
    return new_count


def detect_compound_discoveries(
    current: list[DiscoveredInvariant],
    all_previous: list[DiscoveredInvariant],
) -> int:
    """Count discoveries that appear to be compound — built from previously
    discovered simpler invariants.

    A discovery is "compound" if its expression contains pieces of known
    invariants from previous generations.

    Example: "E^2 - (p*c)^2" is compound if "E" and "p*c" were discovered
    in earlier generations.
    """
    if not all_previous:
        return 0

    # Build a set of known sub-expressions from previous discoveries
    known_pieces: set[str] = set()
    for prev in all_previous:
        expr = prev.expression
        # Extract meaningful sub-expressions (skip pure constants)
        # Split on operators and collect pieces
        clean = expr.replace("(", " ").replace(")", " ").replace("^2", "")
        tokens = clean.replace("+", " ").replace("-", " ").replace("*", " ").replace("/", " ")
        for token in tokens.split():
            token = token.strip()
            if token and not token.replace(".", "").replace("e", "").replace("-", "").isdigit():
                known_pieces.add(token)

    compound_count = 0
    for d in current:
        # Skip if this exact expression was already discovered before
        expr_key = d.expression.replace(" ", "")
        already_known = any(
            p.expression.replace(" ", "") == expr_key
            for p in all_previous
        )
        if already_known:
            continue

        expr_norm = d.expression.replace(" ", "")
        pieces_found = 0
        for piece in known_pieces:
            if piece in expr_norm and piece != d.expression:
                pieces_found += 1
        if pieces_found >= 2:  # Need at least 2 known pieces to be "compound"
            compound_count += 1

    return compound_count


# ═══════════════════════════════════════════════════════════════════════════
# Main curriculum loop
# ═══════════════════════════════════════════════════════════════════════════

def run_curriculum(
    generations: int = 3,
    era_cutoff: int = 1905,
    base_data_examples: int = 12000,
    seed: int = DEFAULT_SEED,
    output_path: str | Path = "data/curriculum_results.json",
    skip_initial_training: bool = False,
) -> dict:
    """Run the full curriculum retraining pipeline.

    Args:
        generations: Number of curriculum generations.
        era_cutoff: Physics era cutoff year.
        base_data_examples: Number of initial self-play examples.
        seed: Random seed.
        output_path: Path for final results JSON.
        skip_initial_training: Skip Gen 0 template training (use existing
            checkpoints).

    Returns:
        Full results dict suitable for JSON export.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CURRICULUM RETRAINING — Progressive Learning Pipeline")
    print(f"Generations: {generations}")
    print(f"Era cutoff: {era_cutoff}")
    print(f"Output: {output_path}")
    print("=" * 70)

    # ── Prepare initial training data ─────────────────────────────────────
    print("\n── Preparing initial training data ──")
    base_data = prepare_initial_data(n_examples=base_data_examples, seed=seed)

    # ── Track all discoveries across generations ──────────────────────────
    all_discoveries: list[DiscoveredInvariant] = []
    gen_results: list[GenerationResult] = []
    current_data_path = base_data
    prev_discoveries: list[DiscoveredInvariant] = []

    for gen in range(generations):
        gen_start = time.time()
        print(f"\n{'#' * 70}")
        print(f"# GENERATION {gen + 1}/{generations}")
        print(f"{'#' * 70}")

        # ── Phase 1: Train template generators ────────────────────────────
        if gen == 0 and skip_initial_training:
            print("\n  Skipping Gen 0 template training (using existing checkpoints)")
        else:
            train_template_generators_on(current_data_path)

        # ── Phase 2: Run era gate ─────────────────────────────────────────
        era_results = run_era_gate(era_cutoff)

        # ── Phase 3: Extract discoveries ──────────────────────────────────
        discoveries = extract_discoveries(era_results, gen)
        all_discoveries.extend(discoveries)

        # Compute metrics
        new_count = detect_new_discoveries(discoveries, prev_discoveries)
        compound_count = detect_compound_discoveries(discoveries, prev_discoveries)

        # Show discoveries
        print(f"\n  Discoveries this generation: {len(discoveries)}")
        for d in discoveries:
            marker = ""
            if d.expression_key() not in {p.expression_key() for p in prev_discoveries}:
                if d.scenario_id not in {p.scenario_id for p in prev_discoveries}:
                    marker = " ★ NEW"
            print(f"    {marker} [{d.scenario_id}] {d.expression} "
                  f"(const={d.constancy:.4f}, domain={d.domain})")

        if compound_count > 0:
            print(f"  Compound discoveries: {compound_count} "
                  f"(building on previous invariants)")

        # ── Phase 4: Generate solved examples for next generation ─────────
        solved_examples = [
            discovery_to_training_example(d) for d in discoveries
        ]

        # ── Record generation result ──────────────────────────────────────
        gen_elapsed = time.time() - gen_start
        gr = GenerationResult(
            generation=gen,
            training_examples=len(solved_examples),
            discoveries=discoveries,
            total_scenarios=era_results.get("total_scenarios", 0),
            spacetime_verified=era_results.get("spacetime_verified", 0),
            any_discovery=len(discoveries),
            new_discoveries=new_count,
            compound_discoveries=compound_count,
            timing_seconds=gen_elapsed,
        )
        gen_results.append(gr)

        # ── Prepare for next generation ───────────────────────────────────
        if gen < generations - 1 and discoveries:
            current_data_path = prepare_curriculum_data(
                base_data_path=current_data_path,
                solved_examples=solved_examples,
                generation=gen,
            )
        elif not discoveries:
            print("\n  No discoveries in this generation — stopping early.")
            break

        prev_discoveries = discoveries[:]

    # ═══════════════════════════════════════════════════════════════════════
    # Final summary
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("CURRICULUM RESULTS SUMMARY")
    print(f"{'=' * 70}")

    total_discoveries = len(all_discoveries)
    unique_exprs = len({d.expression_key() for d in all_discoveries})
    unique_scenarios = len({d.scenario_id for d in all_discoveries})

    print(f"  Total discoveries: {total_discoveries}")
    print(f"  Unique expressions: {unique_exprs}")
    print(f"  Unique scenarios: {unique_scenarios}")

    for gr in gen_results:
        status = "✓" if gr.any_discovery > 0 else "✗"
        compound_info = ""
        if gr.compound_discoveries > 0:
            compound_info = f", compound={gr.compound_discoveries}"
        print(f"  Gen {gr.generation}: {status} "
              f"verified={gr.spacetime_verified}/{gr.total_scenarios}, "
              f"discovered={gr.any_discovery}, "
              f"new={gr.new_discoveries}{compound_info}, "
              f"{gr.timing_seconds:.0f}s")

    # Check for progressive improvement
    if len(gen_results) >= 2:
        first_gen = gen_results[0]
        best_gen = max(gen_results, key=lambda g: g.spacetime_verified)
        if best_gen.spacetime_verified > first_gen.spacetime_verified:
            print(f"\n  📈 PROGRESSIVE IMPROVEMENT DETECTED:")
            print(f"     Gen 0: {first_gen.spacetime_verified} verified")
            print(f"     Best (Gen {best_gen.generation}): "
                  f"{best_gen.spacetime_verified} verified")
            improvement = best_gen.spacetime_verified - first_gen.spacetime_verified
            print(f"     +{improvement} new scenarios discovered through curriculum")

    # Check for compound synthesis
    total_compound = sum(gr.compound_discoveries for gr in gen_results)
    if total_compound > 0:
        print(f"\n  🧩 COMPOUND SYNTHESIS: {total_compound} compound invariants")
        print(f"     System built complex invariants from simpler building blocks")

    # Build output dict
    output = {
        "experiment": "CURRICULUM_RETRAINING",
        "description": "Progressive learning: discoveries → training data → better discoveries",
        "parameters": {
            "generations": generations,
            "era_cutoff": era_cutoff,
            "base_data_examples": base_data_examples,
            "seed": seed,
            "skip_initial_training": skip_initial_training,
        },
        "generations": [
            {
                "generation": gr.generation,
                "training_examples": gr.training_examples,
                "total_scenarios": gr.total_scenarios,
                "spacetime_verified": gr.spacetime_verified,
                "any_discovery": gr.any_discovery,
                "new_discoveries": gr.new_discoveries,
                "compound_discoveries": gr.compound_discoveries,
                "timing_seconds": gr.timing_seconds,
                "discoveries": [
                    {
                        "expression": d.expression,
                        "scenario_id": d.scenario_id,
                        "scenario_name": d.scenario_name,
                        "constancy": d.constancy,
                        "domain": d.domain,
                        "quantities": d.quantities,
                    }
                    for d in gr.discoveries
                ],
            }
            for gr in gen_results
        ],
        "summary": {
            "total_discoveries": total_discoveries,
            "unique_expressions": unique_exprs,
            "unique_scenarios": unique_scenarios,
            "compound_invariants": total_compound,
            "progressive_improvement": (
                gen_results[-1].spacetime_verified > gen_results[0].spacetime_verified
                if len(gen_results) >= 2 else False
            ),
            "generations_completed": len(gen_results),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return output


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curriculum Retraining — Progressive learning from solved examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/curriculum_retrain.py --generations 3\n"
            "  python scripts/curriculum_retrain.py --generations 5 --era-cutoff 1905\n"
            "  python scripts/curriculum_retrain.py --generations 3 --output data/my_results.json\n"
            "  python scripts/curriculum_retrain.py --generations 3 --skip-initial-training\n"
        ),
    )
    parser.add_argument(
        "--generations", type=int, default=3,
        help="Number of curriculum generations (default: 3)",
    )
    parser.add_argument(
        "--era-cutoff", type=int, default=1905,
        help="Physics era cutoff year (default: 1905)",
    )
    parser.add_argument(
        "--output", type=str, default="data/curriculum_results.json",
        help="Output path for curriculum results JSON",
    )
    parser.add_argument(
        "--base-examples", type=int, default=12000,
        help="Number of initial self-play examples (default: 12000)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-initial-training", action="store_true",
        help="Skip Gen 0 template training (use existing checkpoints)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without running the era gate",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — would execute:")
        print(f"  Generations: {args.generations}")
        print(f"  Era cutoff: {args.era_cutoff}")
        print(f"  Output: {args.output}")
        print(f"  Base examples: {args.base_examples}")
        print(f"  Skip initial training: {args.skip_initial_training}")
        print()
        for gen in range(args.generations):
            print(f"  Gen {gen}:")
            print(f"    1. Train template generators on curriculum data")
            print(f"    2. Run: python {ERA_GATE_SCRIPT} --era-cutoff {args.era_cutoff}")
            print(f"    3. Extract discovered invariants")
            print(f"    4. Add to training data for next generation")
        return

    run_curriculum(
        generations=args.generations,
        era_cutoff=args.era_cutoff,
        base_data_examples=args.base_examples,
        seed=args.seed,
        output_path=args.output,
        skip_initial_training=args.skip_initial_training,
    )


if __name__ == "__main__":
    main()
