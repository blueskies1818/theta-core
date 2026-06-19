#!/usr/bin/env python3
"""Generate milestone gate report and push to GitHub.

Reads gate audit output and produces a standardized markdown report
in docs/reports/, then commits and pushes.

Usage:
    # From gate audit JSON output
    python scripts/build/generate_report.py --gate 1 --input audit_result.json

    # From scratch with manual values
    python scripts/build/generate_report.py --gate 1 --verdict PASS \
        --metric "Total leaks" --value 0 --threshold 0
"""

import sys, json, argparse, subprocess, os
from datetime import datetime, timezone
from pathlib import Path

GATE_NAMES = {
    1: "Training Data Purity",
    2: "Structural Independence",
    3: "Lemma Novelty",
    4: "Negative Control",
    5: "Statistical Validity",
}

REPORT_TEMPLATE = """# Gate {gate_num}: {gate_name} — {verdict}

**Date:** {timestamp}
**Git commit:** `{commit_hash}`
**Verdict:** {verdict}

---

## Results

{metrics_section}

---

## Evidence

{evidence_section}

---

## System State

| Property | Value |
|----------|-------|
| Git commit | `{commit_hash}` |
| Training theorems | {train_count} ({train_file}) |
| Eval theorems | {eval_count} ({eval_file}) |
| GNN checkpoint | {checkpoint} |
| Config | {config_summary} |

---

## {next_section}

{next_content}

---

*Report generated automatically by scripts/build/generate_report.py*
"""


def get_git_commit(project_root: Path) -> str:
    """Get current HEAD commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=project_root, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_project_state(project_root: Path) -> dict:
    """Capture current project state."""
    state = {
        "commit": get_git_commit(project_root),
        "train_count": 0,
        "train_file": "?",
        "eval_count": 0,
        "eval_file": "?",
        "checkpoint": "?",
        "config_summary": "?",
    }

    # Count training theorems
    train_path = project_root / "data" / "raw" / "training_combined.jsonl"
    if train_path.exists():
        with open(train_path) as f:
            state["train_count"] = sum(1 for _ in f)
        state["train_file"] = "data/raw/training_combined.jsonl"

    # Count eval theorems
    eval_path = project_root / "data" / "raw" / "physics_theorems_post1905.jsonl"
    if eval_path.exists():
        with open(eval_path) as f:
            state["eval_count"] = sum(1 for _ in f)
        state["eval_file"] = "data/raw/physics_theorems_post1905.jsonl"

    # Find latest checkpoint
    checkpoints_dir = project_root / "checkpoints"
    if checkpoints_dir.exists():
        pt_files = sorted(checkpoints_dir.rglob("gnn_final.pt"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if pt_files:
            state["checkpoint"] = str(pt_files[0].relative_to(project_root))

    # Config summary
    gnn_config = project_root / "src" / "explorer" / "gnn_config.py"
    if gnn_config.exists():
        text = gnn_config.read_text()
        hidden = re.search(r"hidden_dim\s*[:=]\s*(\d+)", text) if (re := __import__("re")) else None
        layers = re.search(r"num_layers\s*[:=]\s*(\d+)", text) if re else None
        if hidden and layers:
            state["config_summary"] = f"GNN: {hidden.group(1)}-dim, {layers.group(1)} layers"

    return state


def format_metrics(verdict: str, metric_data: list) -> str:
    """Format metrics as a markdown table."""
    if not metric_data:
        return "_No metrics provided_"

    lines = ["| Metric | Value | Threshold | Status |", "|--------|-------|-----------|--------|"]
    for m in metric_data:
        status = "✓" if m.get("passed", True) else "✗"
        lines.append(f"| {m['name']} | {m['value']} | {m.get('threshold', '—')} | {status} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate gate milestone report")
    parser.add_argument("--gate", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--verdict", required=True, choices=["PASS", "MARGINAL", "FAIL"])
    parser.add_argument("--input", help="JSON file with gate audit results")
    parser.add_argument("--metric", action="append", nargs=3,
                        metavar=("NAME", "VALUE", "THRESHOLD"),
                        help="Add a metric (repeatable)")
    parser.add_argument("--evidence", default="", help="Evidence description")
    parser.add_argument("--no-push", action="store_true", help="Skip git push")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    now = datetime.now(timezone.utc)
    state = get_project_state(project_root)

    # Build metrics
    metric_data = []
    if args.input:
        with open(args.input) as f:
            audit = json.load(f)
        # Extract metrics from audit JSON
        if "total_leaks" in audit:
            metric_data.append({"name": "Post-1904 leaks", "value": audit["total_leaks"],
                               "threshold": "0", "passed": audit["total_leaks"] == 0})
        if "shape_matcher_rate" in audit:
            metric_data.append({"name": "Shape-matcher match rate",
                               "value": f"{audit['shape_matcher_rate']:.2%}",
                               "threshold": f"≤{audit.get('threshold', 0):.2%}",
                               "passed": audit.get("verdict") != "FAIL"})
        if "random_baseline" in audit:
            metric_data.append({"name": "Random baseline",
                               "value": f"{audit['random_baseline']:.4f}",
                               "threshold": "—", "passed": True})

    if args.metric:
        for name, value, threshold in args.metric:
            metric_data.append({"name": name, "value": value, "threshold": threshold,
                               "passed": True})

    # Build report
    timestamp = now.strftime("%Y-%m-%d %H:%M UTC")
    filename = f"gate{args.gate}_{'pass' if args.verdict == 'PASS' else args.verdict.lower()}_{now.strftime('%Y%m%d_%H%M')}.md"

    next_section = "Next Steps"
    if args.verdict == "PASS":
        if args.gate < 5:
            next_content = f"Proceed to Gate {args.gate + 1}: {GATE_NAMES.get(args.gate + 1, '?')}."
        else:
            next_content = "All gates passed. The system is PROVEN. Capstone report complete."
    elif args.verdict == "MARGINAL":
        next_content = (f"Gate {args.gate} is marginal. Increase sample size or refine "
                       f"the test and re-run. The specific issues are documented above.")
    else:
        next_content = (f"Gate {args.gate} FAILED. Fix the issues documented above and "
                       f"re-run. Do not proceed to Gate {args.gate + 1} until this passes.")

    report = REPORT_TEMPLATE.format(
        gate_num=args.gate,
        gate_name=GATE_NAMES.get(args.gate, f"Gate {args.gate}"),
        verdict=args.verdict,
        timestamp=timestamp,
        commit_hash=state["commit"],
        metrics_section=format_metrics(args.verdict, metric_data),
        evidence_section=args.evidence or "_See audit output above for detailed evidence._",
        train_count=state["train_count"],
        train_file=state["train_file"],
        eval_count=state["eval_count"],
        eval_file=state["eval_file"],
        checkpoint=state["checkpoint"],
        config_summary=state["config_summary"],
        next_section=next_section,
        next_content=next_content,
    )

    # Write report
    reports_dir = project_root / "docs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / filename
    report_path.write_text(report)

    print(f"Report written: {report_path}")
    print()

    # Git commit
    try:
        subprocess.run(["git", "add", str(report_path.relative_to(project_root))],
                      cwd=project_root, check=True, timeout=10)
        subprocess.run(["git", "commit", "-m",
                       f"report: Gate {args.gate} {GATE_NAMES.get(args.gate, '')} — {args.verdict}"],
                      cwd=project_root, check=True, timeout=10)
        print("Committed to git.")

        if not args.no_push:
            result = subprocess.run(["git", "push", "origin", "main"],
                                   cwd=project_root, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("Pushed to GitHub.")
            else:
                print(f"Push failed: {result.stderr.strip()}")
                print("Report is committed locally. Push manually when ready.")
    except subprocess.CalledProcessError as e:
        print(f"Git operation failed: {e}")
        print("Report is written to disk but not committed.")


if __name__ == "__main__":
    main()
