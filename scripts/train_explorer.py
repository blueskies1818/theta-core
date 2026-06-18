#!/usr/bin/env python3
"""Train the GNN+MCTS explorer on the real mathlib4 dependency graph.

Wires the full Phase 2 pipeline:
    Dependency graph (58K nodes) → GNN encoder → MCTS proof search
    → Lean proof checker → Correspondence reward modifier → GRPO update

This is the AlphaGo Zero training loop applied to theorem proving with
physics-correspondence reward shaping.

Usage:
    # Train on GroupTheory (2,929 nodes — fastest)
    python scripts/train_explorer.py --domain GroupTheory --max-theorems 200 --steps 50

    # Train on Algebra (16,800 nodes)
    python scripts/train_explorer.py --domain Algebra --max-theorems 500 --steps 100

    # Train on full graph (58K nodes — heavy)
    python scripts/train_explorer.py --full-graph --max-theorems 500 --steps 20

    # Resume from checkpoint
    python scripts/train_explorer.py --domain Algebra --resume checkpoints/explorer/gnn_step_50.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.explorer.dependency_graph import DependencyGraph
from src.explorer.gnn_config import GNNConfig
from src.explorer.gnn_encoder import GNNEncoder, prepare_graph_tensors, extract_initial_features
from src.explorer.mcts import MCTSConfig
from src.explorer.explorer_trainer import ExplorerTrainer, ExplorerConfig
from src.proof_checker.batch_checker import BatchChecker
from src.correspondence.reward_integration import (
    CorrespondenceRewardModifier,
    EraGatedRewardModifier,
    create_default_modifier,
)
from src.correspondence.era_tracker import (
    EraTracker,
    ERA_CUTOFFS,
    create_era_tracker,
)
from src.reward.config import RewardConfig, load_reward_config
from src.utils.xpu_utils import get_device, print_device_info


def load_theorems_subset(
    jsonl_path: Path,
    max_theorems: int = 500,
    domain_filter: str | None = None,
    shuffle: bool = True,
) -> list[dict]:
    """Load a subset of theorems from the JSONL file.

    Args:
        jsonl_path: Path to mathlib4_theorems.jsonl.
        max_theorems: Maximum number of theorems to load.
        domain_filter: If set, filter theorems whose source_file mentions this domain.
        shuffle: Whether to shuffle before truncating.

    Returns:
        List of theorem dicts with 'statement' key.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Theorem file not found: {jsonl_path}")

    # Normalize domain filter: extract domain from paths like
    # ../mathlib4/Mathlib/<Domain>/... by checking for Mathlib/<Domain>/
    def _domain_matches(src: str, filt: str) -> bool:
        """Check if source_file path matches the domain filter."""
        if filt.lower() in src.lower():
            return True
        # Also check for Mathlib/<Domain>/ pattern
        for part in src.split("/"):
            if part.lower() == filt.lower():
                return True
        return False

    theorems = []
    # Check if entries have source_file fields (mathlib4 theorems do, bootstrap don't)
    _has_source_files = False
    with open(jsonl_path) as f:
        first_line = f.readline()
        if first_line:
            try:
                probe = json.loads(first_line)
                _has_source_files = "source_file" in probe
            except json.JSONDecodeError:
                pass

    if domain_filter and _has_source_files:
        # Domain-filtered: read the entire file and collect matches.
        # We need to scan all lines because rare domains may be deep in the file.
        print(f"Scanning theorems for domain '{domain_filter}'...")
        with open(jsonl_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    src = d.get("source_file", "")
                    if _domain_matches(src, domain_filter):
                        theorems.append(d)
                        if max_theorems and len(theorems) >= max_theorems:
                            break
                except json.JSONDecodeError:
                    continue
        print(f"  Found {len(theorems)} theorems matching '{domain_filter}'")
    elif domain_filter and not _has_source_files:
        print(f"Note: theorem file has no source_file fields — ignoring domain filter '{domain_filter}'")
        domain_filter = None  # Fall through to no-filter path below

    if not domain_filter:
        # No filter: read enough, then shuffle and truncate.
        read_limit = max_theorems * 5 if max_theorems else None
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if read_limit and i >= read_limit:
                    break
                try:
                    d = json.loads(line)
                    theorems.append(d)
                except json.JSONDecodeError:
                    continue
        if shuffle:
            random.shuffle(theorems)
        theorems = theorems[:max_theorems]

    # Normalize to the format ExplorerTrainer expects
    result = []
    for t in theorems:
        stmt = t.get("statement", "")
        if not stmt or len(stmt) < 10:
            continue
        # Clean: strip theorem/lemma keyword prefix, keep the statement
        # The proof checker wraps it in a theorem declaration anyway
        result.append({
            "statement": stmt,
            "name": t.get("name", ""),
            "proof": t.get("proof", ""),  # Ground truth (for evaluation)
        })

    print(f"Loaded {len(result)} training theorems")
    return result


def build_graph(
    graph_path: str = "data/graph/dependency_graph",
    domain: str | None = None,
) -> DependencyGraph:
    """Load the dependency graph, optionally filtered to a domain.

    Args:
        graph_path: Path prefix for the saved graph files.
        domain: If set, extract only this domain's subgraph.

    Returns:
        DependencyGraph ready for training.
    """
    full_path = _project_root / graph_path
    if not full_path.with_suffix(".nx.pkl").exists():
        raise FileNotFoundError(
            f"Graph not found at {full_path}.nx.pkl\n"
            f"Run: python scripts/build_dependency_graph.py"
        )

    graph = DependencyGraph.load(full_path)
    print(f"Loaded graph: {graph.summary()}")

    if domain:
        available = graph.get_statistics().get("nodes_by_domain", {})
        if domain not in available:
            print(f"Warning: domain '{domain}' not found. Available: {list(available.keys())}")
            print("Using full graph instead.")
        else:
            print(f"Extracting '{domain}' subgraph ({available[domain]} nodes)...")
            graph = graph.domain_subgraph(domain)
            print(f"  Subgraph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    return graph


def create_gnn(
    graph: DependencyGraph,
    pretrained_path: str | None = None,
    hidden_dim: int = 768,
    num_layers: int = 5,
    num_heads: int = 12,
    device: torch.device | None = None,
) -> GNNEncoder:
    """Create or load a GNN encoder.

    Args:
        graph: The dependency graph (provides node count for input_dim config).
        pretrained_path: Path to a pretrained GNN checkpoint (optional).
        hidden_dim: Hidden dimension (ignored if loading pretrained).
        num_layers: Number of GAT layers.
        num_heads: Number of attention heads.
        device: Target device.

    Returns:
        GNNEncoder ready for training.
    """
    if pretrained_path:
        path = Path(pretrained_path)
        if not path.is_absolute():
            path = _project_root / path
        if path.exists():
            print(f"Loading pretrained GNN from {path}")
            gnn = GNNEncoder.load(path)
            cfg = gnn.config
            print(f"  Config: hidden={cfg.hidden_dim}, layers={cfg.num_layers}, "
                  f"heads={cfg.num_heads}")
            if device:
                gnn = gnn.to(device)
            return gnn
        else:
            print(f"Warning: pretrained path not found: {path}, creating fresh GNN")

    # Fresh GNN — match input_dim to graph feature extraction
    config = GNNConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        input_dim=hidden_dim,  # Match hidden_dim for random init
        dropout=0.1,
        activation="gelu",
        use_edge_types=True,
        num_edge_types=4,
        bidirectional=True,
    )

    gnn = GNNEncoder(config)
    total_params = sum(p.numel() for p in gnn.parameters())
    print(f"Created fresh GNN: {total_params:,} params "
          f"(hidden={hidden_dim}, layers={num_layers}, heads={num_heads})")

    if device:
        gnn = gnn.to(device)

    return gnn


def main():
    parser = argparse.ArgumentParser(
        description="Train the GNN+MCTS explorer on mathlib4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --domain GroupTheory --max-theorems 200 --steps 50
  %(prog)s --domain Algebra --max-theorems 500 --steps 100 --pretrained checkpoints/gnn/gnn_best.pt
  %(prog)s --full-graph --max-theorems 1000 --steps 20 --device cuda:0
        """,
    )

    # -- Data ----------------------------------------------------------------
    parser.add_argument("--graph", default="data/graph/dependency_graph",
                        help="Path prefix for saved graph files")
    parser.add_argument("--theorems", default="data/raw/mathlib4_theorems.jsonl",
                        help="Path to theorems JSONL file")
    parser.add_argument("--domain", default=None,
                        help="Filter to a specific math domain (e.g., Algebra, GroupTheory)")
    parser.add_argument("--full-graph", action="store_true",
                        help="Use the full 58K-node graph (overrides --domain)")
    parser.add_argument("--max-theorems", type=int, default=500,
                        help="Maximum training theorems to load")

    # -- Model ---------------------------------------------------------------
    parser.add_argument("--pretrained", default=None,
                        help="Path to pretrained GNN checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=768,
                        help="GNN hidden dimension (for fresh init)")
    parser.add_argument("--num-layers", type=int, default=5,
                        help="GNN layers")
    parser.add_argument("--num-heads", type=int, default=12,
                        help="GNN attention heads")

    # -- Training ------------------------------------------------------------
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of training steps (epochs)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Theorems per training step")
    parser.add_argument("--group-size", type=int, default=2,
                        help="Proofs per theorem for GRPO advantages")
    parser.add_argument("--mcts-sims", type=int, default=500,
                        help="MCTS simulations per proof search (500 balances search depth vs gradient diversity)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--policy-weight", type=float, default=1.0,
                        help="Policy loss weight")
    parser.add_argument("--value-weight", type=float, default=0.5,
                        help="Value loss weight")
    parser.add_argument("--heuristic-anneal-epochs", type=int, default=2000,
                        help="Epochs to anneal heuristics from 1.0 to 0.0 "
                             "(0 = no annealing, keep heuristics at full)")
    parser.add_argument("--heuristic-scale-min", type=float, default=0.25,
                        help="Final heuristic scale after annealing "
                             "(0.25 prevents policy collapse, 0.0 = pure GNN)")
    parser.add_argument("--resume-epoch", type=int, default=0,
                        help="Resume from this epoch (sets starting heuristic scale)")

    # -- Correspondence ------------------------------------------------------
    parser.add_argument("--no-correspondence", action="store_true",
                        help="Disable correspondence-layer reward shaping")
    parser.add_argument("--energy-scale", type=float, default=None,
                        help="Energy scale hint for zone classification (GeV)")

    # -- Temporal gating (era-based discovery monitoring) --------------------
    parser.add_argument("--era", default=None,
                        choices=list(ERA_CUTOFFS.keys()),
                        help="Historical era cutoff for passive discovery monitoring. "
                             "Proofs are scanned for physics concepts from AFTER this era "
                             "but discoveries are OBSERVED, not rewarded. This is an "
                             "honest test: does the explorer spontaneously find post-era "
                             "physics without being told what to look for?\n"
                             "Example: --era pre_relativity (≤1904) monitors for "
                             "special relativity, QM, GR concepts.")

    # -- Output --------------------------------------------------------------
    parser.add_argument("--output", default="checkpoints/explorer",
                        help="Output directory for checkpoints")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=5,
                        help="Log metrics every N steps")
    parser.add_argument("--device", default=None,
                        help="Device (auto-detected if not set)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # -- Evaluation ----------------------------------------------------------
    parser.add_argument("--eval-theorems", type=int, default=20,
                        help="Number of held-out theorems for evaluation")
    parser.add_argument("--eval-every", type=int, default=25,
                        help="Evaluate every N steps")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation")
    parser.add_argument("--curriculum", action="store_true",
                        help="Sort theorems by complexity: single-tactic first, then multi-step. "
                             "Disables shuffle for ordered curriculum learning.")

    # -- Traversal bonus (H3 study) ---------------------------------
    parser.add_argument("--traversal", action="store_true",
                        help="Enable graph-traversal reward bonus (H3 study). "
                             "Rewards proofs that use lemmas 3+ hops from training distribution.")
    parser.add_argument("--traversal-weight", type=float, default=0.5,
                        help="Traversal bonus weight (default: 0.5)")
    parser.add_argument("--traversal-hop-threshold", type=int, default=3,
                        help="Minimum graph hops to count as 'far' (default: 3)")

    args = parser.parse_args()

    # ---- Setup ----
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else get_device()
    print_device_info()
    print(f"Using device: {device}")

    # ---- Load graph ----
    domain = None if args.full_graph else args.domain
    graph = build_graph(args.graph, domain=domain)

    # ---- Create GNN ----
    gnn = create_gnn(
        graph,
        pretrained_path=args.pretrained,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        device=device,
    )

    # ---- Load training data ----
    jsonl_path = _project_root / args.theorems
    if not jsonl_path.exists():
        print(f"Error: theorem file not found: {jsonl_path}")
        print("Run: python scripts/prepare_data.py")
        sys.exit(1)

    domain_for_filter = domain if domain else None
    train_theorems = load_theorems_subset(
        jsonl_path,
        max_theorems=args.max_theorems,
        domain_filter=domain_for_filter,
        shuffle=not args.curriculum,  # No shuffle for curriculum — ordered presentation
    )

    # Curriculum: sort by proof complexity (single-tactic → multi-step)
    if args.curriculum and train_theorems:
        def _proof_complexity(t: dict) -> int:
            proof = t.get("proof", "")
            # Score: count tactic separators (semicolons, newlines) + 1
            n_tactics = proof.count(";") + proof.count("\n") + 1
            # Also weigh by proof length for similar tactic counts
            return n_tactics * 1000 + len(proof)

        train_theorems.sort(key=_proof_complexity)
        print(f"Curriculum mode: theorems sorted by complexity "
              f"(1-tactic → multi-step)")
        print(f"  First: {train_theorems[0].get('name', '?')} "
              f"({train_theorems[0].get('proof', '').strip()[:50]}...)")
        print(f"  Last:  {train_theorems[-1].get('name', '?')} "
              f"({train_theorems[-1].get('proof', '').strip()[:50]}...)")

    if len(train_theorems) < args.batch_size:
        print(f"Error: only {len(train_theorems)} theorems available, "
              f"need at least {args.batch_size}")
        sys.exit(1)

    # ---- Split train/val ----
    if args.no_eval:
        train_split = train_theorems
        val_split = None
    else:
        split_idx = max(args.batch_size, len(train_theorems) - args.eval_theorems)
        train_split = train_theorems[:split_idx]
        val_split = train_theorems[split_idx:]
    print(f"Train: {len(train_split)} theorems, Val: {len(val_split) if val_split else 0}")

    # ---- Proof checker ----
    print("Initializing proof checker...")
    try:
        checker = BatchChecker(
            timeout=30,
            max_workers=4,
            cache_size=128,
        )
        print("  Proof checker ready (Lean 4 with Mathlib4)")
    except Exception as e:
        print(f"  Warning: proof checker init failed: {e}")
        print("  Continuing without proof checking (proofs will not be validated)")
        checker = None

    # ---- Correspondence modifier ----
    correspondence = None
    era_tracker = None
    if not args.no_correspondence:
        try:
            base_modifier = create_default_modifier()
            print(f"Correspondence modifier loaded: "
                  f"{len(base_modifier.frontier_map.zones)} zones, "
                  f"{len(base_modifier.failure_coords.failure_points)} failure points")

            # Wire era-gated discovery tracking if requested
            if args.era:
                era_tracker = create_era_tracker(args.era)
                correspondence = EraGatedRewardModifier(
                    base_modifier=base_modifier,
                    era_tracker=era_tracker,
                )
                print(f"Era tracking: {args.era} (≤{era_tracker.cutoff_year})")
                print(f"  Known concepts: {len(era_tracker.known_concepts)}")
                print(f"  Discoverable concepts: {len(era_tracker.discoverable_concepts)}")
                for concept in era_tracker.discoverable_concepts:
                    print(f"    {concept.name} ({concept.year}): {concept.description[:60]}...")
            else:
                correspondence = base_modifier
        except Exception as e:
            print(f"Warning: could not load correspondence modifier: {e}")
            print("Training without correspondence shaping.")
            correspondence = None

    # ---- Configs ----
    explorer_config = ExplorerConfig(
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.lr,
        weight_decay=1e-5,
        max_grad_norm=1.0,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        use_correspondence=correspondence is not None,
        correspondence_energy_scale=args.energy_scale,
        log_every=args.log_every,
        save_every=args.save_every,
        heuristic_anneal_epochs=args.heuristic_anneal_epochs,
        heuristic_scale_min=args.heuristic_scale_min,
        resume_epoch=args.resume_epoch,
    )

    mcts_config = MCTSConfig(
        num_simulations=args.mcts_sims,
        c_puct=1.4,
        max_depth=20,
        max_actions_per_node=50,
        temperature=0.5,
        top_k_lemmas=30,
        use_gnn=True,
        use_proof_checker=True,
        verify_timeout=5.0,
    )

    reward_config = load_reward_config() if args.traversal else RewardConfig()
    if args.traversal:
        reward_config.traversal_bonus_enabled = True
        reward_config.traversal_bonus_weight = args.traversal_weight
        reward_config.traversal_hop_threshold = args.traversal_hop_threshold

    # ---- Trainer ----
    trainer = ExplorerTrainer(
        gnn_encoder=gnn,
        dependency_graph=graph,
        proof_checker=checker,
        config=explorer_config,
        mcts_config=mcts_config,
        reward_config=reward_config,
        correspondence_modifier=correspondence,
        device=device,
    )

    # ---- Print setup summary ----
    print()
    print("=" * 60)
    print("Explorer Training Setup")
    print("=" * 60)
    print(f"  Graph:        {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")
    print(f"  GNN params:   {sum(p.numel() for p in gnn.parameters()):,}")
    print(f"  Training:     {args.steps} steps × {args.batch_size} theorems")
    print(f"  MCTS:         {args.mcts_sims} sims per proof")
    print(f"  Group size:   {args.group_size} proofs per theorem (GRPO)")
    print(f"  LR:           {args.lr}")
    print(f"  Loss weights: policy={args.policy_weight}, value={args.value_weight}")
    if correspondence:
        print(f"  Correspondence: {len(correspondence.frontier_map.zones)} zones, "
              f"{len(correspondence.failure_coords.failure_points)} failures")
        if era_tracker:
            print(f"  Era:           {era_tracker.era_name} "
                  f"(≤{era_tracker.cutoff_year}) — "
                  f"{len(era_tracker.discoverable_concepts)} discoverable concepts")
    else:
        print(f"  Correspondence: DISABLED")
    if args.traversal:
        print(f"  Traversal:      ENABLED (weight={args.traversal_weight}, "
              f"hop_threshold={args.traversal_hop_threshold})")
    else:
        print(f"  Traversal:      DISABLED")
    print(f"  Device:       {device}")
    print(f"  Output:       {args.output}")
    print("=" * 60)
    print()

    # ---- Training ----
    print(f"Starting training: {args.steps} steps...")
    t_start = time.time()

    try:
        metrics = trainer.train(
            train_theorems=train_split,
            val_theorems=val_split,
            output_dir=_project_root / args.output,
            num_epochs=args.steps,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.gnn.save(_project_root / args.output / "gnn_interrupted.pt")
        print(f"Saved to {args.output}/gnn_interrupted.pt")
        return
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        # Try to save what we have
        try:
            trainer.gnn.save(_project_root / args.output / "gnn_error.pt")
            print(f"Saved partial state to {args.output}/gnn_error.pt")
        except Exception:
            pass
        sys.exit(1)

    elapsed = time.time() - t_start

    # ---- Final summary ----
    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"  Total time:    {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Steps:         {args.steps}")
    if metrics and "metrics" in metrics:
        mlist = metrics["metrics"]
        if mlist:
            successes = [m.get("success_rate", 0) for m in mlist if "success_rate" in m]
            if successes:
                print(f"  Final success: {successes[-1]:.1%}")
                print(f"  Best success:  {max(successes):.1%}")
            final_losses = [m.get("loss", float("inf")) for m in mlist if "loss" in m]
            if final_losses:
                print(f"  Final loss:    {final_losses[-1]:.4f}")
    print(f"  Checkpoints:   {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
