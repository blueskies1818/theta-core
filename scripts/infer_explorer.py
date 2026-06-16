#!/usr/bin/env python3
"""Run inference with a trained GNN+MCTS explorer on held-out physics theorems.

This is the moment-of-truth test:
  - Train on pre-1905 physics theorems (classical through pre-relativity)
  - Run inference on post-1905 theorems (old quantum through modern)
  - Measure: can the explorer find valid proofs for physics it wasn't trained on?

Usage:
    # Single-scale inference
    python scripts/infer_explorer.py --checkpoint checkpoints/explorer_run4/gnn_final.pt

    # Compare pure GNN vs heuristics (H=0.0 vs H=1.0)
    python scripts/infer_explorer.py --checkpoint checkpoints/gnn/proof_step_pretrained.pt --compare

    # Test on post-1905 file specifically
    python scripts/infer_explorer.py --checkpoint checkpoints/gnn/proof_step_pretrained.pt \\
        --theorems data/raw/physics_theorems_post1905.jsonl --no-era-filter
"""

import sys, json, argparse, time, re, csv, statistics
from pathlib import Path
from collections import Counter

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch
from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from src.explorer.mcts import MCTS, MCTSConfig
from src.explorer.proof_state import ProofState
from src.proof_checker.batch_checker import BatchChecker
from src.proof_checker.formats import wrap_theorem_with_proof
from src.correspondence.era_tracker import create_era_tracker


def classify_proof_pattern(proof_steps: list[str]) -> str:
    """Classify a proof into its primary pattern category.

    Categories:
      - rfl:         reflexive (exact rfl, Eq.refl)
      - add_comm:    commutative addition (rw [add_comm])
      - mul_comm:    commutative multiplication (rw [mul_comm])
      - ring:        ring tactic
      - field_simp:  field_simp tactic
      - linarith:    linarith tactic
      - simp:        simp tactic
      - hypothesis:  uses a hypothesis (h, h1, etc.)
      - intro:       uses intro
      - apply:       uses apply
      - multi:       multi-step proof (2+ distinct tactic types)
      - auto:        automation tactics (positivity, norm_num, nlinarith)
      - other:       unrecognized pattern
    """
    if not proof_steps:
        return "empty"

    steps_text = " ".join(proof_steps).lower()

    # Multi-step check first (multiple distinct tactics)
    tactic_types = set()
    for step in proof_steps:
        s = step.strip().lower()
        if s.startswith("rw"):
            tactic_types.add("rw")
        elif s.startswith("exact"):
            tactic_types.add("exact")
        elif s.startswith("apply"):
            tactic_types.add("apply")
        elif s.startswith("intro"):
            tactic_types.add("intro")
        elif s.startswith("have"):
            tactic_types.add("have")
        elif s in ("ring", "simp", "linarith", "field_simp", "positivity", "norm_num", "nlinarith"):
            tactic_types.add(s)
        elif s.startswith("calc"):
            tactic_types.add("calc")
        else:
            tactic_types.add("other")

    if len(tactic_types) >= 2:
        return "multi"

    # Single-step classification
    # Reflexive
    if any(tok in steps_text for tok in ("rfl", "eq.refl")):
        return "rfl"

    # Lemma-based
    if "add_comm" in steps_text:
        return "add_comm"
    if "mul_comm" in steps_text:
        return "mul_comm"
    if "add_assoc" in steps_text or "mul_assoc" in steps_text:
        return "assoc"
    if "add_zero" in steps_text or "zero_add" in steps_text:
        return "identity"
    if "mul_one" in steps_text or "one_mul" in steps_text:
        return "identity"

    # Tactics
    if "ring" in steps_text:
        return "ring"
    if "field_simp" in steps_text:
        return "field_simp"
    if "linarith" in steps_text:
        return "linarith"
    if "simp" in steps_text:
        return "simp"

    # Structural tactics
    if "intro" in steps_text:
        return "intro"
    if "apply" in steps_text:
        return "apply"
    if "have" in steps_text:
        return "hypothesis"

    # Automation
    if any(t in steps_text for t in ("positivity", "norm_num", "nlinarith")):
        return "auto"

    # Hypothesis usage: references like h, h1, h_ (not lemma names)
    if re.search(r'\b(h\b|h\d+|h_\w+)', steps_text):
        return "hypothesis"

    return "other"


def run_inference_multi(
    n_runs: int,
    checkpoint: str,
    graph_path: str,
    domain: str,
    theorems_path: str,
    max_theorems: int | None,
    mcts_sims: int,
    era: str,
    device: str,
    heuristic_scale: float,
    verbose: bool,
    no_era_filter: bool = False,
) -> list[list[dict]]:
    """Run inference N times, returning a list of result-lists."""
    all_runs = []
    label = f"H={heuristic_scale}"
    for run_i in range(n_runs):
        if n_runs > 1:
            print(f"\n--- {label} run {run_i+1}/{n_runs} ---")
        results = run_inference(
            checkpoint=checkpoint,
            graph_path=graph_path,
            domain=domain,
            theorems_path=theorems_path,
            max_theorems=max_theorems,
            mcts_sims=mcts_sims,
            era=era,
            device=device,
            heuristic_scale=heuristic_scale,
            verbose=verbose,
            no_era_filter=no_era_filter,
        )
        all_runs.append(results)
    return all_runs


def print_multi_run_stats(runs: list[list[dict]], label: str):
    """Print mean ± stdev statistics across multiple inference runs."""
    n_runs = len(runs)
    n_theorems = len(runs[0]) if runs else 0

    # Per-run success counts
    run_successes = [sum(1 for r in run if r["success"]) for run in runs]

    import statistics
    mean_success = statistics.mean(run_successes)
    stdev_success = statistics.stdev(run_successes) if n_runs > 1 else 0.0

    print(f"\n{label} ({n_runs} runs): "
          f"{mean_success:.1f} ± {stdev_success:.1f} / {n_theorems} "
          f"({mean_success/n_theorems*100:.1f}% ± {stdev_success/n_theorems*100:.1f}pp)")

    # Per-theorem consistency: how many times each theorem was proved
    if n_runs > 1:
        theorem_hits = {}
        for run in runs:
            for r in run:
                name = r["name"]
                theorem_hits.setdefault(name, 0)
                if r["success"]:
                    theorem_hits[name] += 1

        always = [n for n, h in theorem_hits.items() if h == n_runs]
        never = [n for n, h in theorem_hits.items() if h == 0]
        sometimes = [n for n, h in theorem_hits.items() if 0 < h < n_runs]

        print(f"  Always proved ({len(always)}): {always}")
        print(f"  Never proved  ({len(never)}): {never}")
        if sometimes:
            print(f"  Inconsistent   ({len(sometimes)}): {[(n, theorem_hits[n]) for n in sometimes]}")


def run_trivial_baselines(
    theorems: list[dict],
    checker: BatchChecker,
    verbose: bool = False,
) -> dict[str, dict]:
    """Test each trivial tactic on every theorem — no MCTS, no GNN.

    Returns dict mapping tactic name → {success_count, total, results}.
    This gives a lower bound: "what if we just throw simp at everything?"
    """
    tactics = ["rfl", "simp", "linarith", "ring", "field_simp", "positivity", "norm_num"]

    baseline_results = {t: {"success": 0, "failed": 0, "results": []} for t in tactics}

    for t in theorems:
        stmt = t["statement"]
        name = t.get("name", "?")
        for tactic in tactics:
            code = wrap_theorem_with_proof(stmt, tactic)
            check_results = checker.check_batch([code])
            ok = check_results[0].success
            if ok:
                baseline_results[tactic]["success"] += 1
                if verbose:
                    print(f"  ✓ {tactic:12s} → {name}")
            else:
                baseline_results[tactic]["failed"] += 1
            baseline_results[tactic]["results"].append(
                {"name": name, "success": ok}
            )

    return baseline_results


def run_inference(
    checkpoint: str,
    graph_path: str,
    domain: str,
    theorems_path: str,
    max_theorems: int | None,
    mcts_sims: int,
    era: str,
    device: str,
    heuristic_scale: float,
    verbose: bool,
    no_era_filter: bool = False,
) -> list[dict]:
    """Run inference and return list of result dicts."""

    device_t = torch.device(device)
    if not verbose:
        print(f"Device: {device_t}  H-scale: {heuristic_scale}  Sims: {mcts_sims}")

    # ---- Load graph ----
    gp = Path(graph_path)
    if not gp.is_absolute():
        gp = _project_root / gp
    if not gp.with_suffix(".nx.pkl").exists():
        print(f"Error: graph not found at {gp}.nx.pkl")
        return []

    graph = DependencyGraph.load(gp)
    if domain:
        available = graph.get_statistics().get("nodes_by_domain", {})
        if domain in available:
            graph = graph.domain_subgraph(domain)
            if not verbose:
                print(f"Graph: {graph.num_nodes} nodes ({domain})")

    # ---- Load GNN ----
    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = _project_root / ckpt_path
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        return []

    gnn = GNNEncoder.load(str(ckpt_path))
    gnn.eval()
    gnn = gnn.to(device_t)
    if not verbose:
        n_params = sum(p.numel() for p in gnn.parameters())
        print(f"GNN: {n_params:,} params, hidden={gnn.config.hidden_dim}")

    # ---- Compute embeddings ----
    features = extract_initial_features(graph, gnn.config)
    sources, targets, edge_types, num_nodes = prepare_graph_tensors(graph)
    features = features.to(device_t)
    sources = sources.to(device_t)
    targets = targets.to(device_t)
    edge_types = edge_types.to(device_t)

    with torch.no_grad():
        embeddings = gnn(features, sources, targets, edge_types, num_nodes)

    # ---- Load theorems ----
    tp = Path(theorems_path)
    if not tp.is_absolute():
        tp = _project_root / tp
    if not tp.exists():
        print(f"Error: theorem file not found: {tp}")
        return []

    with open(tp) as f:
        all_theorems = [json.loads(line) for line in f]

    # Era filtering
    if no_era_filter:
        test_theorems = all_theorems
        pre_theorems = []
    else:
        era_tracker = create_era_tracker(era) if era else None
        cutoff_year = era_tracker.cutoff_year if era_tracker else 1904
        era_order = ["classical", "classical_crisis", "pre_relativity",
                     "pre_gr", "old_quantum", "pre_qed", "pre_sm",
                     "sm_construction", "sm_confirmed", "precision_era", "modern"]
        cutoff_idx = era_order.index(era) if era in era_order else 2
        post_theorems = []
        pre_theorems = []
        for t in all_theorems:
            t_era = t.get("era", "modern")
            t_idx = era_order.index(t_era) if t_era in era_order else len(era_order)
            if t_idx > cutoff_idx:
                post_theorems.append(t)
            else:
                pre_theorems.append(t)
        test_theorems = post_theorems

    if max_theorems:
        test_theorems = test_theorems[:max_theorems]

    if not verbose:
        print(f"Theorems: {len(test_theorems)}")

    # ---- Proof checker ----
    checker = BatchChecker(timeout=30, max_workers=4, cache_size=128)

    # ---- MCTS ----
    mcts_config = MCTSConfig(
        num_simulations=mcts_sims,
        max_depth=10,
        top_k_lemmas=30,
        c_puct=1.4,
        heuristic_scale=heuristic_scale,
        use_proof_checker=True,
        verify_timeout=5.0,
    )
    mcts = MCTS(gnn_encoder=gnn, dependency_graph=graph, config=mcts_config,
                proof_checker=checker)
    mcts.set_embeddings(embeddings, sorted(graph.node_ids))

    # ---- Run inference ----
    results = []
    t_start = time.time()

    for i, t in enumerate(test_theorems):
        stmt = t['statement']
        name = t['name']
        zone = t.get('frontier_zone', 'unknown')
        era_name = t.get('era', 'unknown')
        ground_truth = t.get('proof', '?')

        t0 = time.time()
        best_steps, root = mcts.search(stmt, verbose=False)
        search_time = time.time() - t0

        proof_text = ProofState._render_proof(best_steps)
        full_code = wrap_theorem_with_proof(stmt, proof_text or 'sorry')

        # Truncation heuristic
        if len(best_steps) > 1:
            first_action = best_steps[0]
            if (first_action.tactic_type.value in ("rewrite", "apply")
                    and first_action.lemma in ("add_comm", "mul_comm", "rfl", "Eq.refl")):
                single_text = ProofState._render_proof(best_steps[:1])
                full_code = wrap_theorem_with_proof(stmt, single_text or 'sorry')

        check_results = checker.check_batch([full_code])
        ok = check_results[0].success
        err = check_results[0].errors[0][:120] if check_results[0].errors else ""
        mcts_steps = [s.to_lean() for s in best_steps[:5]]
        pattern = classify_proof_pattern(mcts_steps) if ok else "failed"

        result = {
            "name": name, "era": era_name, "zone": zone,
            "success": ok, "error": err,
            "mcts_steps": mcts_steps, "num_steps": len(best_steps),
            "ground_truth": ground_truth,
            "search_time_s": search_time,
            "pattern": pattern,
            "heuristic_scale": heuristic_scale,
        }
        results.append(result)

        if verbose or ok:
            status = "✓" if ok else "✗"
            eta_display = era_name.replace("_", " ")
            print(f"  [{i+1:2d}/{len(test_theorems)}] {status} {name:40s} {eta_display:20s} {zone:25s} [{pattern}] {search_time:.1f}s")
            if ok:
                print(f"         Proof: {mcts_steps}")
            elif verbose:
                print(f"         Steps: {mcts_steps}")
                print(f"         Error: {err}")

    elapsed = time.time() - t_start
    if not verbose:
        passed = sum(1 for r in results if r["success"])
        print(f"  Result: {passed}/{len(results)} ({passed/max(1,len(results))*100:.0f}%) in {elapsed:.0f}s")

    try:
        checker.shutdown()
    except:
        pass

    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained GNN+MCTS explorer")
    parser.add_argument("--checkpoint", required=True, help="Path to trained GNN checkpoint")
    parser.add_argument("--graph", default="data/graph/dependency_graph", help="Graph path prefix")
    parser.add_argument("--domain", default="Algebra", help="Graph domain filter")
    parser.add_argument("--theorems", default="data/raw/physics_theorems_post1905.jsonl",
                        help="Theorem file (default: post-1905 held-out)")
    parser.add_argument("--max-theorems", type=int, default=None, help="Max theorems to test")
    parser.add_argument("--mcts-sims", type=int, default=500, help="MCTS simulations per proof")
    parser.add_argument("--era", default="pre_relativity", help="Era cutoff for discovery tracking")
    parser.add_argument("--device", default="cpu", help="Device for GNN inference")
    parser.add_argument("--heuristic-scale", type=float, default=1.0,
                        help="Heuristic scale (0.0 = pure GNN, 1.0 = full heuristics)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed proof output")
    parser.add_argument("--no-era-filter", action="store_true",
                        help="Test all theorems in file (no era split)")
    parser.add_argument("--compare", action="store_true",
                        help="Run both H=0.0 and H=1.0 and compare results")
    parser.add_argument("--csv", default=None, help="Save results to CSV file")
    parser.add_argument("--baselines", action="store_true",
                        help="Run trivial baselines (simp/linarith/ring/etc on every theorem)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat inference N times and report mean ± stdev")
    args = parser.parse_args()

    print("=" * 70)
    print("GNN+MCTS Inference")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Theorems:   {args.theorems}")
    print(f"Graph:      {args.graph} ({args.domain})")
    print(f"MCTS sims:  {args.mcts_sims}")
    print()

    if args.compare:
        # ---- Comparison mode: H=0.0 vs H=1.0 ----
        print("-" * 70)
        print(f"Running H=0.0 (pure GNN) × {args.repeat}...")
        print("-" * 70)
        gnn_runs = run_inference_multi(
            n_runs=args.repeat,
            checkpoint=args.checkpoint,
            graph_path=args.graph,
            domain=args.domain,
            theorems_path=args.theorems,
            max_theorems=args.max_theorems,
            mcts_sims=args.mcts_sims,
            era=args.era,
            device=args.device,
            heuristic_scale=0.0,
            verbose=args.verbose,
            no_era_filter=args.no_era_filter,
        )
        results_gnn = gnn_runs[0]  # First run for detailed display

        print()
        print("-" * 70)
        print(f"Running H=1.0 (full heuristics) × {args.repeat}...")
        print("-" * 70)
        heur_runs = run_inference_multi(
            n_runs=args.repeat,
            checkpoint=args.checkpoint,
            graph_path=args.graph,
            domain=args.domain,
            theorems_path=args.theorems,
            max_theorems=args.max_theorems,
            mcts_sims=args.mcts_sims,
            era=args.era,
            device=args.device,
            heuristic_scale=1.0,
            verbose=args.verbose,
            no_era_filter=args.no_era_filter,
        )
        results_heur = heur_runs[0]  # First run for detailed display

        # ---- Trivial baselines ----
        if args.baselines:
            print()
            print("-" * 70)
            print("Running trivial baselines (simp/linarith/ring/field_simp/rfl on everything)...")
            print("-" * 70)
            # Load theorems for baseline (same theorem set)
            all_theorems_for_baseline = []
            tp = Path(args.theorems)
            if not tp.is_absolute():
                tp = _project_root / tp
            with open(tp) as f:
                all_theorems_for_baseline = [json.loads(line) for line in f]
            # Apply same era filter as H=0.0 run
            if args.no_era_filter:
                baseline_theorems = all_theorems_for_baseline
            else:
                era_tracker = create_era_tracker(args.era) if args.era else None
                cutoff_year = era_tracker.cutoff_year if era_tracker else 1904
                era_order = ["classical", "classical_crisis", "pre_relativity",
                             "pre_gr", "old_quantum", "pre_qed", "pre_sm",
                             "sm_construction", "sm_confirmed", "precision_era", "modern"]
                cutoff_idx = era_order.index(args.era) if args.era in era_order else 2
                baseline_theorems = []
                for t in all_theorems_for_baseline:
                    t_era = t.get("era", "modern")
                    t_idx = era_order.index(t_era) if t_era in era_order else len(era_order)
                    if t_idx > cutoff_idx:
                        baseline_theorems.append(t)
            if args.max_theorems:
                baseline_theorems = baseline_theorems[:args.max_theorems]

            checker2 = BatchChecker(timeout=30, max_workers=4, cache_size=128)
            try:
                baseline_results = run_trivial_baselines(baseline_theorems, checker2, verbose=args.verbose)
                print(f"\n  {'Tactic':<12} {'Success':>8} {'Failed':>8} {'Rate':>8}")
                print(f"  {'-'*40}")
                for tactic in ["rfl", "simp", "linarith", "ring", "field_simp", "positivity", "norm_num"]:
                    br = baseline_results[tactic]
                    total = br["success"] + br["failed"]
                    rate = br["success"] / max(1, total) * 100
                    print(f"  {tactic:<12} {br['success']:>8} {br['failed']:>8} {rate:>7.0f}%")
            finally:
                try:
                    checker2.shutdown()
                except:
                    pass

        # ---- Comparison summary ----
        gnn_passed = [r for r in results_gnn if r["success"]]
        heur_passed = [r for r in results_heur if r["success"]]
        n_total = len(results_gnn)

        print(f"\n{'='*70}")
        print(f"COMPARISON: H=0.0 (pure GNN) vs H=1.0 (heuristics)")
        print(f"{'='*70}")
        print(f"{'Metric':<30} {'H=0.0':>10} {'H=1.0':>10} {'Δ':>10}")
        print(f"{'-'*60}")
        print(f"{'Proved':<30} {len(gnn_passed):>10} {len(heur_passed):>10} {len(heur_passed)-len(gnn_passed):>+10}")
        print(f"{'Success rate':<30} {len(gnn_passed)/max(1,n_total)*100:>9.0f}% {len(heur_passed)/max(1,n_total)*100:>9.0f}% {(len(heur_passed)-len(gnn_passed))/max(1,n_total)*100:>+9.0f}pp")

        # Pattern comparison
        print(f"\nProof patterns (successful proofs only):")
        gnn_patterns = Counter(r["pattern"] for r in gnn_passed)
        heur_patterns = Counter(r["pattern"] for r in heur_passed)
        all_patterns = sorted(set(list(gnn_patterns.keys()) + list(heur_patterns.keys())))
        if all_patterns:
            print(f"  {'Pattern':<15} {'H=0.0':>8} {'H=1.0':>8}")
            for pat in all_patterns:
                print(f"  {pat:<15} {gnn_patterns.get(pat,0):>8} {heur_patterns.get(pat,0):>8}")

        # By zone
        print(f"\nBy frontier zone:")
        zone_order = sorted(set(r["zone"] for r in results_gnn))
        print(f"  {'Zone':<30} {'H=0.0':>10} {'H=1.0':>10}")
        for zone in zone_order:
            gnn_zone = sum(1 for r in gnn_passed if r["zone"] == zone)
            heur_zone = sum(1 for r in heur_passed if r["zone"] == zone)
            zone_total = sum(1 for r in results_gnn if r["zone"] == zone)
            if zone_total > 0:
                print(f"  {zone:<30} {gnn_zone}/{zone_total:>4} ({gnn_zone/zone_total*100:>4.0f}%) {heur_zone}/{zone_total:>4} ({heur_zone/zone_total*100:>4.0f}%)")

        # Theorems where GNN succeeded but heuristics failed (pure GNN wins)
        gnn_names = {r["name"] for r in gnn_passed}
        heur_names = {r["name"] for r in heur_passed}
        gnn_only = gnn_names - heur_names
        heur_only = heur_names - gnn_names

        if gnn_only:
            print(f"\nTheorems proved ONLY by GNN (H=0.0):")
            for r in gnn_passed:
                if r["name"] in gnn_only:
                    print(f"  ✓ {r['name']} [{r['era']}] [{r['pattern']}] → {r['mcts_steps']}")

        if heur_only:
            print(f"\nTheorems proved ONLY with heuristics:")
            for r in heur_passed:
                if r["name"] in heur_only:
                    print(f"  ✓ {r['name']} [{r['era']}] [{r['pattern']}] → {r['mcts_steps']}")

        # Multi-run statistics (if --repeat > 1)
        if args.repeat > 1:
            print(f"\n{'='*70}")
            print(f"MULTI-RUN STATISTICS ({args.repeat} runs)")
            print(f"{'='*70}")
            print_multi_run_stats(gnn_runs, "H=0.0 (pure GNN)")
            print_multi_run_stats(heur_runs, "H=1.0 (heuristics)")

        # Save CSV
        if args.csv:
            csv_path = _project_root / args.csv
            fieldnames = ["name", "era", "zone", "heuristic_scale", "success", "pattern",
                          "mcts_steps", "ground_truth", "error", "search_time_s"]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in results_gnn + results_heur:
                    r_copy = dict(r)
                    r_copy["mcts_steps"] = " | ".join(r_copy["mcts_steps"])
                    writer.writerow(r_copy)
            print(f"\nResults saved to {csv_path}")

        return 0 if len(gnn_passed) > 0 else 1

    else:
        # ---- Single-scale mode ----
        all_runs = run_inference_multi(
            n_runs=args.repeat,
            checkpoint=args.checkpoint,
            graph_path=args.graph,
            domain=args.domain,
            theorems_path=args.theorems,
            max_theorems=args.max_theorems,
            mcts_sims=args.mcts_sims,
            era=args.era,
            device=args.device,
            heuristic_scale=args.heuristic_scale,
            verbose=args.verbose,
            no_era_filter=args.no_era_filter,
        )
        results = all_runs[0]  # First run for detailed display

        if not results:
            return 1

        # ---- Summary ----
        passed = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        print(f"\n{'='*70}")
        print(f"INFERENCE RESULTS  (H={args.heuristic_scale})")
        print(f"{'='*70}")
        print(f"  Theorems tested:  {len(results)}")
        print(f"  Proved:           {len(passed)} ({len(passed)/max(1,len(results))*100:.0f}%)")
        print(f"  Failed:           {len(failed)}")

        # Proof patterns
        if passed:
            patterns = Counter(r["pattern"] for r in passed)
            print(f"\n  Proof patterns:")
            for pat, count in patterns.most_common():
                print(f"    {pat:<15}: {count}")

        # By era
        era_groups = {}
        for r in results:
            era_groups.setdefault(r["era"], []).append(r)
        print("\nBy era:")
        for era_name in sorted(era_groups):
            era_results = era_groups[era_name]
            era_passed = [r for r in era_results if r["success"]]
            print(f"  {era_name:25s}: {len(era_passed)}/{len(era_results)} ({len(era_passed)/max(1,len(era_results))*100:.0f}%)")

        # By zone
        print("\nBy frontier zone:")
        zone_counts = Counter(r["zone"] for r in results)
        zone_passed = Counter(r["zone"] for r in passed)
        for zone in sorted(zone_counts):
            p = zone_passed.get(zone, 0)
            t = zone_counts[zone]
            bar = "█" * int(p/t*20) + "░" * (20 - int(p/t*20))
            print(f"  {zone:30s}: {p}/{t} ({p/t*100:3.0f}%) {bar}")

        # Proof list
        if passed:
            print(f"\nProved theorems:")
            for r in passed:
                print(f"  ✓ [{r['zone']:25s}] [{r['era']:20s}] {r['name']}")
                print(f"    Pattern: {r['pattern']} | Proof: {r['mcts_steps']}")
                print(f"    Ground truth: {r['ground_truth']}")

        if failed and args.verbose:
            print(f"\nFailed theorems:")
            for r in failed:
                print(f"  ✗ [{r['zone']:25s}] [{r['era']:20s}] {r['name']}")
                print(f"    Steps: {r['mcts_steps'][:3]}")
                print(f"    Error: {r['error']}")

        # Multi-run statistics (if --repeat > 1)
        if args.repeat > 1:
            print(f"\n{'='*70}")
            print(f"MULTI-RUN STATISTICS ({args.repeat} runs)")
            print(f"{'='*70}")
            print_multi_run_stats(all_runs, f"H={args.heuristic_scale}")

        # Save CSV
        if args.csv:
            csv_path = _project_root / args.csv
            fieldnames = ["name", "era", "zone", "heuristic_scale", "success", "pattern",
                          "mcts_steps", "ground_truth", "error", "search_time_s"]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in results:
                    r_copy = dict(r)
                    r_copy["mcts_steps"] = " | ".join(r_copy["mcts_steps"])
                    writer.writerow(r_copy)
            print(f"\nResults saved to {csv_path}")

        return 0 if len(passed) > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
