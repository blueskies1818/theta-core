#!/usr/bin/env python3
"""Extract theorem-proof pairs from Mathlib4 and prepare training datasets.

Usage:
    python scripts/prepare_data.py --mathlib-dir ../mathlib4 --output-dir data
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mathlib_extractor import (
    DEFAULT_DOMAINS,
    extract_all_theorems,
    save_theorems_jsonl,
    load_theorems_jsonl,
)

# Use a smaller subset for initial extraction
INITIAL_DOMAINS = [
    "Algebra",
    "GroupTheory",
    "LinearAlgebra",
    "Data/Real",
    "Data/Nat",
    "Data/Int",
]


def main():
    parser = argparse.ArgumentParser(
        description="Extract theorems from Mathlib4 for training"
    )
    parser.add_argument(
        "--mathlib-dir",
        type=str,
        default="../mathlib4",
        help="Path to Mathlib4 repository",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--max-theorems",
        type=int,
        default=50000,
        help="Maximum number of theorems to extract",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=INITIAL_DOMAINS,
        help="Mathlib4 domains to extract from",
    )
    args = parser.parse_args()

    mathlib_dir = Path(args.mathlib_dir)
    if not mathlib_dir.exists():
        print(f"Error: Mathlib4 directory not found at {mathlib_dir}")
        print("Clone it with: git clone https://github.com/leanprover-community/mathlib4.git")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting theorems from Mathlib4: {mathlib_dir}")
    print(f"Domains: {args.domains}")
    print(f"Max theorems: {args.max_theorems}")

    theorems = extract_all_theorems(
        mathlib_dir,
        domains=args.domains,
        max_theorems=args.max_theorems,
    )

    output_path = raw_dir / "mathlib4_theorems.jsonl"
    save_theorems_jsonl(theorems, output_path)

    # Print statistics
    print(f"\nExtraction complete: {len(theorems)} theorems")
    if theorems:
        avg_proof_len = sum(len(t["proof"]) for t in theorems) / len(theorems)
        avg_statement_len = sum(len(t["statement"]) for t in theorems) / len(theorems)
        print(f"Avg statement length: {avg_statement_len:.0f} chars")
        print(f"Avg proof length: {avg_proof_len:.0f} chars")


if __name__ == "__main__":
    main()
