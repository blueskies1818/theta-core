#!/usr/bin/env python3
"""Generate and save quantum and relativity synthetic observation data."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.physics.simulators.quantum import generate_all_quantum
from src.physics.simulators.relativity import generate_all_relativity

out_dir = Path(__file__).parent.parent.parent / "data" / "observations"
out_dir.mkdir(parents=True, exist_ok=True)

quantum = generate_all_quantum()
relativity = generate_all_relativity()

q_path = out_dir / "quantum_synthetic.json"
r_path = out_dir / "relativity_synthetic.json"

with open(q_path, "w") as f:
    json.dump(quantum, f, indent=2)
with open(r_path, "w") as f:
    json.dump(relativity, f, indent=2)

print(f"Saved {len(quantum)} quantum scenarios to {q_path}")
print(f"Saved {len(relativity)} relativity scenarios to {r_path}")
