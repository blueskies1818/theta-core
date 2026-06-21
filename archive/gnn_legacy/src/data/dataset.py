"""PyTorch/HuggingFace Dataset classes for theorem-proving data."""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TheoremProofDataset(Dataset):
    """Dataset of theorem-proof pairs for supervised fine-tuning.

    Each item is a formatted string: 'Theorem: <STATEMENT>\nProof: <PROOF>'
    Tokenized and padded to max_seq_length.
    """

    def __init__(
        self,
        theorems: list[dict],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1024,
    ):
        self.theorems = theorems
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Pre-format all examples
        self.examples = []
        for theorem in theorems:
            text = self._format_example(theorem)
            self.examples.append(text)

    @staticmethod
    def _format_example(theorem: dict) -> str:
        """Format a theorem dict into a training string."""
        statement = theorem.get("statement", "")
        proof = theorem.get("proof", "")
        return f"Theorem: {statement}\nProof: {proof}"

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        text = self.examples[idx]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": tokens["input_ids"].squeeze(0),
        }


class ProofGenerationDataset(Dataset):
    """Dataset for RL self-play: provides theorem statements only.

    The model generates proofs; the proof checker provides rewards.
    """

    def __init__(
        self,
        theorems: list[dict],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1024,
    ):
        self.theorems = theorems
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Format theorem statements for generation prompts
        self.prompts = []
        for theorem in theorems:
            statement = theorem.get("statement", "")
            prompt = f"Theorem: {statement}\nProof:"
            self.prompts.append(prompt)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        prompt = self.prompts[idx]
        tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "prompt": prompt,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "statement": self.theorems[idx].get("statement", ""),
            "known_proof": self.theorems[idx].get("proof", ""),
        }


def create_datasets(
    theorems: list[dict],
    tokenizer: PreTrainedTokenizer,
    train_split: float = 0.9,
    max_seq_length: int = 1024,
) -> tuple[TheoremProofDataset, TheoremProofDataset]:
    """Create train/val datasets from extracted theorems."""
    split_idx = int(len(theorems) * train_split)
    train_theorems = theorems[:split_idx]
    val_theorems = theorems[split_idx:]

    train_ds = TheoremProofDataset(train_theorems, tokenizer, max_seq_length)
    val_ds = TheoremProofDataset(val_theorems, tokenizer, max_seq_length)

    return train_ds, val_ds
