# theta-core — Win Conditions & Pause Conditions

Reference for the orchestrator when creating Kanban tasks.
Every task gets explicit conditions from its category below.

---

## Task Category 1: Architecture Change

Code changes to the core explorer (GNN, MCTS, trainer, reward system).

### Win Condition
- [ ] Code compiles and imports resolve (`python -c "from src.explorer import *"`)
- [ ] All existing tests pass (`python -m pytest tests/ -q`)
- [ ] New tests cover the changed behavior (if behavioral change)
- [ ] A smoke test runs: 1 epoch of training completes without error
- [ ] For scaling changes: memory usage stays within available hardware
- [ ] For gradient changes: `loss.backward()` succeeds, gradients are nonzero
- [ ] Changed files documented in commit message and kanban summary

### Pause Conditions (block task, don't guess)
- OOM on available hardware (needs model-size reduction or hardware upgrade)
- Gradient is zero or NaN after change (architectural bug, needs investigation)
- Existing tests break and can't be trivially fixed (may indicate the change is wrong)
- Change requires a dependency not in pyproject.toml (needs decision on adding)
- GPU-specific code fails on CPU and vice versa (needs device-agnostic rewrite)

### Baseline
- Previous architecture's eval score on held-out set (from most recent wave results)
- Previous architecture's memory usage and train step time

---

## Task Category 2: Training Run

Running a new training experiment with specific config parameters.

### Win Condition
- [ ] Training completes all specified epochs without crash
- [ ] Checkpoint saved at end (`checkpoints/explorer_runN/gnn_final.pt`)
- [ ] Training metrics logged (loss curve, success rate, proof signatures per epoch)
- [ ] Inference eval run on held-out set: `python scripts/eval/infer_explorer.py --checkpoint <path> --compare`
- [ ] Eval results saved as CSV (`data/eval_waveN_results.csv`)
- [ ] New run documented in `docs/training/README.md` run summary table
- [ ] New detailed report written at `docs/training/runN_<descriptor>.md`

### Pause Conditions
- Training loss is NaN or inf after first few epochs (numerical instability)
- Success rate is 0% for entire first 50 epochs (training signal broken)
- OOM during training (reduce batch size, graph size, or model size)
- Proof checker crashes and can't be restarted (Lean/Mathlib environment issue)
- Hardware failure (GPU disappears, disk full)
- Training time exceeds 24h without progress (stuck in degenerate loop)

### Baseline
- Previous run's eval metrics at same H-scale (from `data/eval_wave*.csv`)
- Run 8 GNN at H=0.0: 56% on 25-theorem held-out set

---

## Task Category 3: Theorem Set Design

Creating or modifying the theorem datasets for training or evaluation.

### Win Condition
- [ ] Every new theorem has a VERIFIED ground-truth Lean proof
  - Run: `python scripts/eval/infer_explorer.py --checkpoint <any> --theorems <new file> --verify-ground-truth`
  - Every theorem must have `ground_truth` field that passes Lean checking
- [ ] Theorems actually require the stated capability:
  - "Multi-step" theorems: ground truth proof uses 2+ distinct tactics
  - "Lemma discrimination" theorems: ground truth uses a specific named lemma, not a general tactic
  - "Physics content" theorems: statement encodes a physical assumption, not just math with a physics name
- [ ] Theorems are diverse across tactic types (not all `simp` or all `linarith`)
- [ ] No overlap with held-out eval set (theorems unique to training)
- [ ] File format matches existing (`training_combined.jsonl` or `<name>.jsonl`)

### Pause Conditions
- Can't write a valid Lean proof for a proposed theorem (theorem may be false)
- Theorem requires a Mathlib lemma not in the dependency graph (needs graph rebuild)
- Theorem is provably equivalent to an existing theorem (redundant)
- Theorem requires capabilities the current architecture fundamentally can't learn
  (e.g., 10-step proofs when MCTS max_depth=5)

### Baseline
- Current 55-theorem set: ~60% provable at H=1.0, 56% at H=0.0 (trained GNN)
- linarith ceiling: 64% of current theorems provable by linarith alone
- rfl ceiling: ~15% provable by rfl alone

---

## Task Category 4: Data Pipeline

Graph building, theorem extraction, data format changes.

### Win Condition
- [ ] Output file exists and passes format validation
- [ ] For graph changes: `python -c "from src.explorer.dependency_graph import DependencyGraph; g = DependencyGraph.load('<path>'); print(g.stats())"` succeeds
- [ ] Graph stats match expected: node count, edge count, domain distribution
- [ ] No data corruption: spot-check 5 random entries against source
- [ ] If replacing an existing data file: old file backed up first
- [ ] Script is re-runnable (idempotent or clean-output)

### Pause Conditions
- Source data unavailable (Mathlib not cloned, file missing)
- Extraction produces 0 results (regex/source format changed)
- Output file >1GB (may need streaming or chunking)
- Memory usage exceeds available RAM during build

### Baseline
- Current graph: 58,370 nodes, 160,611 edges, 7 domains
- Current training data: 55 theorems (29 physics + 26 richer)

---

## Task Category 5: Evaluation Infrastructure

Changes to eval scripts, metrics, baselines, or comparison methodology.

### Win Condition
- [ ] Eval script runs to completion: `python scripts/eval/infer_explorer.py --checkpoint <path> --compare`
- [ ] Output includes all required metrics (success rate, per-pattern breakdown, per-zone breakdown)
- [ ] If adding baselines: baseline score is computed and reported
- [ ] If adding statistical rigor: multi-run stats with mean ± std reported
- [ ] Eval results match expectations when run on a known checkpoint (no regression)
- [ ] Output format matches existing CSV schema or documents new schema

### Pause Conditions
- Eval script crashes on known-good checkpoint (regression introduced)
- Baseline computation is wrong (e.g., linarith baseline >100% — impossible)
- Statistical method is inappropriate (e.g., std on N=2)
- New metric doesn't actually measure what it claims to measure

### Baseline
- Current eval: 25 held-out theorems, compare H=0.0 vs H=1.0, per-zone breakdown
- Wave 2 results: `data/eval_wave2_results.csv`

---

## Task Category 6: Analysis / Review

Reading results, diagnosing failures, making recommendations.
These are reviewer-assigned tasks.

### Win Condition
- [ ] Specific, evidence-backed findings (not "maybe try X")
- [ ] Every claim references a data point (eval result, log line, code line)
- [ ] Failure patterns categorized (e.g., "8/11 failures are lemma discrimination, 3/11 are multi-step")
- [ ] Clear recommendation with priority ordering
- [ ] Recommendations are actionable (specific file/parameter to change, not vague direction)
- [ ] Output written to `docs/reviews/<descriptive_name>.md`

### Pause Conditions
- Insufficient data to draw conclusions (need more eval runs, larger sample)
- Conflicting evidence that can't be resolved without additional experiments
- Finding requires domain expertise beyond what's available (e.g., physics validation)

### Baseline
- Most recent review doc (e.g., `roadmap_review_june2026.md`)
- Most recent eval wave results

---

## Task Category 7: Documentation

README updates, training reports, review docs, roadmap updates.

### Win Condition
- [ ] Document is internally consistent (no contradictions)
- [ ] All referenced files/checkpoints/scripts exist at stated paths
- [ ] Numbers match source data (verify eval scores against CSV, not memory)
- [ ] Roadmap status table updated if roadmap doc
- [ ] Date and scope stated in header

### Pause Conditions
- Referenced data/checkpoint doesn't exist (can't verify claims)
- Document contradicts a more recent review doc

---

## Cross-Cutting Rules

These apply to ALL task types:

### Always
- Task must have a SINGLE clear owner (one assignee)
- Task spec must include exact file paths to touch
- Task spec must state which win condition category applies
- Task must state its baseline for comparison

### Never
- Create a task without a verifiable win condition
- Create a task that depends on an uncreated task without using `parents=[...]`
- Mark a task complete when the win condition isn't met
- Skip spec review before code quality review

### When Blocked
- Use `kanban_block` with reason in format: `<category>: <specific issue>`
  Example: "OOM: GNN at 1024-dim with full graph uses 18GB, Arc B70 has 34GB but message-passing intermediates push it over. Options: reduce dim to 512, use subgraph sampling, or acquire cloud GPU."
- Never block without suggesting recovery paths
