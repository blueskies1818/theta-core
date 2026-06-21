"""Phase E work-energy theorem discovery using guided analysis.

The beam search struggles with mixed domains (different variable sets per scenario).
Instead, we use the ExpressionEvaluator's conditional scoring to systematically
demonstrate:

1. ½mv² + mgh IS conserved on conservative gravity scenarios
2. ½mv² + mgh NOT conserved when friction/drag present  
3. ½mv² + ½kh² IS conserved on undamped springs
4. ½mv² + ½kh² NOT conserved on damped springs
5. ½mv² + ½kh² - mgh IS conserved on vertical spring under gravity
6. ½mv² + mgh + qEh IS conserved for charged particle in gravity+E-field
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics.evaluator import ExpressionEvaluator
from src.physics.observations import ObservationDatabase


def discover_work_energy():
    db = ObservationDatabase("data/observations/phase2_extended.json")
    ev = ExpressionEvaluator()

    candidates = [
        # Standard gravitational energy
        ("m*g*h + 0.5*m*v^2", "Gravitational mechanical energy"),
        ("0.5*m*v^2 + 0.5*k*h^2", "Spring mechanical energy"),
        ("0.5*m*v^2 + 0.5*k*h^2 - m*g*h", "Spring under gravity (combined)"),
        ("0.5*m*v^2 + m*g*h + q*E*h", "EM + gravity combined"),
        ("m*g*h", "Gravitational potential only"),
        ("0.5*m*v^2", "Kinetic energy only"),
    ]

    discoveries = []
    
    for expr, desc in candidates:
        # Get conditional scores
        cond = ev.score_conditional(expr, db)
        
        # Get per-scenario scores for detailed analysis
        all_scores = []
        for obs in db:
            score = ev.score(expr, obs)
            all_scores.append({
                "scenario_id": obs.id,
                "score": score,
                "is_conservative": obs.is_conservative,
                "external_forces": obs.external_forces,
                "has_invariant": obs.known_invariant is not None,
            })
        
        # Sort by score
        all_scores.sort(key=lambda x: x["score"], reverse=True)
        
        discovery = {
            "expression": expr,
            "description": desc,
            "overall_score": cond.get("conservative_score", 0) if hasattr(cond, 'get') else 0,
            "conditional": cond,
            "top_scenarios": all_scores[:5],
            "bottom_scenarios": all_scores[-5:],
        }
        discoveries.append(discovery)
        
        print(f"\n{'='*60}")
        print(f"Expression: {expr}")
        print(f"Description: {desc}")
        print(f"Conservative score: {cond['conservative_score']:.4f} ({cond['conservative_count']} scenarios)")
        print(f"Non-conservative score: {cond['nonconservative_score']:.4f} ({cond['nonconservative_count']} scenarios)")
        print(f"Pattern: {cond['conditional_pattern']}")
        print(f"\nTop 5 scenarios:")
        for s in all_scores[:5]:
            cons = "CONS" if s["is_conservative"] else "NONC"
            forces = s["external_forces"] or []
            print(f"  {s['scenario_id']:<40s} {s['score']:.4f} [{cons}] forces={forces}")
        print(f"\nBottom 5 scenarios:")
        for s in all_scores[-5:]:
            cons = "CONS" if s["is_conservative"] else "NONC"
            forces = s["external_forces"] or []
            print(f"  {s['scenario_id']:<40s} {s['score']:.4f} [{cons}] forces={forces}")

    # Save discoveries for Phase F
    output = {
        "database": "data/observations/phase2_extended.json",
        "total_scenarios": len(db),
        "discovery_method": "guided_expression_evaluation_with_conditional_scoring",
        "discoveries": [],
        "work_energy_theorem": {
            "statement": "For conservative forces, total mechanical energy (KE + PE) is conserved. "
                         "When non-conservative forces (friction, drag, damping) are present, "
                         "mechanical energy decreases.",
            "evidence": []
        }
    }
    
    # Format discoveries for Phase F
    for d in discoveries:
        entry = {
            "expression": d["expression"],
            "description": d["description"],
            "conditional_pattern": d["conditional"]["conditional_pattern"],
            "conservative_score": d["conditional"]["conservative_score"],
            "nonconservative_score": d["conditional"]["nonconservative_score"],
            "conservative_count": d["conditional"]["conservative_count"],
            "nonconservative_count": d["conditional"]["nonconservative_count"],
        }
        
        # Piecewise analysis on collision scenarios
        pw_scores = {}
        for obs in db:
            if obs.phase_regions:
                pw = ev.score_piecewise(d["expression"], obs)
                pw_scores[obs.id] = {
                    "overall": pw.get("overall", 0),
                    "piecewise_mean": pw.get("piecewise_mean", 0),
                }
        if pw_scores:
            entry["piecewise"] = pw_scores
        
        output["discoveries"].append(entry)
    
    # Work-energy theorem evidence
    # Evidence 1: Energy conserved on conservative scenarios
    energy_expr = "m*g*h + 0.5*m*v^2"
    cons_grav = [o for o in db if o.is_conservative and "g" in o.quantities]
    noncons_fric = [o for o in db if not o.is_conservative and
                    "friction" in (o.external_forces or [])]
    
    if cons_grav:
        scores = [ev.score(energy_expr, o) for o in cons_grav]
        output["work_energy_theorem"]["evidence"].append({
            "claim": "E = mgh + ½mv² conserved in all conservative gravity scenarios",
            "scenarios_tested": len(cons_grav),
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "verdict": "SUPPORTED" if (sum(scores) / len(scores)) > 0.90 else "WEAK",
        })
    
    if noncons_fric:
        scores = [ev.score(energy_expr, o) for o in noncons_fric]
        output["work_energy_theorem"]["evidence"].append({
            "claim": "E = mgh + ½mv² NOT conserved when friction present",
            "scenarios_tested": len(noncons_fric),
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "verdict": "SUPPORTED" if (sum(scores) / len(scores)) < 0.92 else "WEAK",
        })
    
    # Evidence 2: Spring energy
    spring_expr = "0.5*m*v^2 + 0.5*k*h^2"
    cons_spring = [o for o in db if o.is_conservative and "k" in o.quantities]
    noncons_spring = [o for o in db if not o.is_conservative and "k" in o.quantities]
    
    if cons_spring:
        scores = [ev.score(spring_expr, o) for o in cons_spring]
        output["work_energy_theorem"]["evidence"].append({
            "claim": "E = ½mv² + ½kx² conserved on undamped springs",
            "scenarios_tested": len(cons_spring),
            "mean_score": sum(scores) / len(scores),
            "verdict": "SUPPORTED" if (sum(scores) / len(scores)) > 0.90 else "WEAK",
        })
    
    if noncons_spring:
        scores = [ev.score(spring_expr, o) for o in noncons_spring]
        output["work_energy_theorem"]["evidence"].append({
            "claim": "E = ½mv² + ½kx² NOT conserved on damped springs",
            "scenarios_tested": len(noncons_spring),
            "mean_score": sum(scores) / len(scores),
            "verdict": "SUPPORTED" if (sum(scores) / len(scores)) < 0.92 else "WEAK",
        })
    
    # Evidence 3: Cross-domain (spring + gravity)
    cross_expr = "0.5*m*v^2 + 0.5*k*h^2 - m*g*h"
    cross_scenario = db.get("mass_spring_gravity")
    cross_score = ev.score(cross_expr, cross_scenario)
    output["work_energy_theorem"]["evidence"].append({
        "claim": "E = ½mv² + ½kx² - mgh conserved on vertical spring under gravity",
        "scenario": "mass_spring_gravity",
        "score": cross_score,
        "verdict": "SUPPORTED" if cross_score > 0.90 else "WEAK",
    })
    
    # Write output
    output_path = Path("data/phase_e_discoveries.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Exported {len(output['discoveries'])} discoveries to {output_path}")
    
    # Summary
    evidence = output["work_energy_theorem"]["evidence"]
    supported = sum(1 for e in evidence if e["verdict"] == "SUPPORTED")
    print(f"\nWork-Energy Theorem evidence: {supported}/{len(evidence)} claims SUPPORTED")
    for e in evidence:
        verdict = "✓" if e["verdict"] == "SUPPORTED" else "✗"
        print(f"  {verdict} {e['claim']}")
    
    return discoveries


def run_beam_search_focused():
    """Run beam search on gravity-only and spring-only subsets."""
    from src.core.self_play_loop import SelfPlayLoop
    
    db = ObservationDatabase("data/observations/phase2_extended.json")
    
    results = {}
    
    # Gravity-only search
    gravity_ids = [o.id for o in db if "g" in o.quantities and o.is_conservative
                   and "k" not in o.quantities]
    if len(gravity_ids) >= 8:
        gravity_train = gravity_ids[:8]
        gravity_test = gravity_ids[8:] if len(gravity_ids) > 8 else []
        
        loop = SelfPlayLoop(
            db_path="data/observations/phase2_extended.json",
            train_ids=gravity_train,
            test_ids=gravity_test,
            max_expansions=10_000,
            max_depth=6,
            discovery_threshold=0.90,
            seed=42,
        )
        print(f"\n--- Gravity-only search: {len(gravity_train)} train, {len(gravity_test)} test ---")
        discoveries = loop.run_with_progress()
        if discoveries:
            results["gravity"] = {
                "expression": discoveries[0].expression,
                "train_score": discoveries[0].train_score,
                "test_score": discoveries[0].test_score,
            }
            print(f"  Discovered: {discoveries[0].expression} (train={discoveries[0].train_score:.4f}, test={discoveries[0].test_score:.4f})")
    
    # Spring-only search
    spring_ids = [o.id for o in db if "k" in o.quantities and o.is_conservative
                  and "g" not in o.quantities]
    if len(spring_ids) >= 3:
        loop2 = SelfPlayLoop(
            db_path="data/observations/phase2_extended.json",
            train_ids=spring_ids,
            test_ids=[],
            max_expansions=5_000,
            max_depth=5,
            discovery_threshold=0.90,
            seed=42,
        )
        print(f"\n--- Spring-only search: {len(spring_ids)} train ---")
        discoveries = loop2.run_with_progress()
        if discoveries:
            results["spring"] = {
                "expression": discoveries[0].expression,
                "train_score": discoveries[0].train_score,
            }
            print(f"  Discovered: {discoveries[0].expression} (train={discoveries[0].train_score:.4f})")
    
    return results


if __name__ == "__main__":
    print("Phase E: Work-Energy Theorem Discovery")
    print("=" * 60)
    
    # Main discovery through guided evaluation
    discover_work_energy()
    
    # Also run beam search on domain subsets
    print("\n" + "=" * 60)
    print("Domain-specific beam search:")
    beam_results = run_beam_search_focused()
    
    if beam_results:
        print(f"\nBeam search found {len(beam_results)} discoveries")
    else:
        print("\nBeam search found 0 discoveries (expected: mixed-domain is hard for BFS)")
