"""Phase E self-play: discover work-energy theorem with conditional conservation."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.self_play_loop import SelfPlayLoop
from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import ObservationDatabase


def main():
    db_path = "data/observations/phase2_extended.json"
    db = ObservationDatabase(db_path)
    print(f"Database: {len(db)} scenarios")

    # Split: all scenarios with known_invariant as potential train
    # We want the system to discover from the training set
    # Use explicit train/test to ensure coverage
    conservative_ids = [o.id for o in db if o.is_conservative]
    nonconservative_ids = [o.id for o in db if not o.is_conservative]

    # Train on 30 conservative + 10 non-conservative
    train_ids = conservative_ids[:30] + nonconservative_ids[:10]
    test_ids = conservative_ids[30:] + nonconservative_ids[10:]

    print(f"Train: {len(train_ids)} ({len([x for x in train_ids if x in conservative_ids])} cons, {len([x for x in train_ids if x in nonconservative_ids])} noncons)")
    print(f"Test:  {len(test_ids)} ({len([x for x in test_ids if x in conservative_ids])} cons, {len([x for x in test_ids if x in nonconservative_ids])} noncons)")

    # Run self-play loop
    loop = SelfPlayLoop(
        db_path=db_path,
        train_ids=train_ids,
        test_ids=test_ids,
        max_expansions=20_000,
        max_depth=10,
        discovery_threshold=0.90,
        top_k=50,
        seed=42,
    )

    print("\nRunning self-play discovery loop...")
    discoveries = loop.run_with_progress()

    print(f"\n{'='*60}")
    print(f"Total expansions: {loop.total_expansions}")
    print(f"Discoveries: {len(discoveries)}")
    print()

    # Phase E: conditional analysis for each discovery
    ev = ExpressionEvaluator()
    all_discoveries = []

    for i, d in enumerate(discoveries):
        print(f"Discovery {i+1}: {d.expression}")
        print(f"  Train score: {d.train_score:.4f}, Test score: {d.test_score:.4f}")

        # Conditional analysis
        cond = ev.score_conditional(d.expression, db)
        print(f"  Conservative score: {cond['conservative_score']:.4f} ({cond['conservative_count']} scenarios)")
        print(f"  Nonconservative score: {cond['nonconservative_score']:.4f} ({cond['nonconservative_count']} scenarios)")
        print(f"  Pattern: {cond['conditional_pattern']}")

        # Piecewise analysis on collision scenarios
        pw_summary = {}
        for obs in db:
            if obs.phase_regions:
                pw = ev.score_piecewise(d.expression, obs)
                pw_summary[obs.id] = {
                    "overall": pw.get("overall", 0),
                    "piecewise_mean": pw.get("piecewise_mean", 0),
                }
        if pw_summary:
            print(f"  Piecewise (collisions):")
            for sid, scores in pw_summary.items():
                print(f"    {sid}: overall={scores['overall']:.4f}, piecewise_mean={scores['piecewise_mean']:.4f}")

        all_discoveries.append({
            "expression": d.expression,
            "train_score": d.train_score,
            "test_score": d.test_score,
            "depth": d.depth,
            "expansions_needed": d.expansions_needed,
            "conditional": cond,
            "piecewise": pw_summary,
            "train_constancies": d.train_constancies,
            "test_constancies": d.test_constancies,
        })
        print()

    # Export discoveries for Phase F
    output = {
        "database": str(loop.db_path),
        "train_ids": train_ids,
        "test_ids": test_ids,
        "total_scenarios": len(db),
        "total_expansions": loop.total_expansions,
        "discoveries": all_discoveries,
        "summary": loop.summary(),
    }

    output_path = Path("data/phase_e_discoveries.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nExported discoveries to {output_path}")

    # Final acceptance checks
    print(f"\n{'='*60}")
    print("ACCEPTANCE CHECKS:")
    print(f"  1. Discovery count > 0: {'PASS' if discoveries else 'FAIL'}")
    has_conservation = any(
        d.expression for d in discoveries if d.test_score >= 0.80
    )
    print(f"  2. Discovered conserved expression: {'PASS' if has_conservation else 'FAIL'}")
    has_conditional = any(
        d["conditional"]["conditional_pattern"] == "conservative_only"
        for d in all_discoveries
    )
    print(f"  3. Conditional pattern detected: {'PASS' if has_conditional else 'NOT YET'}")

    return all_discoveries


if __name__ == "__main__":
    main()
