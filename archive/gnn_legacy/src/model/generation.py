"""Proof generation with controlled decoding for the mathematical explorer."""

import re
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils.config import GenerationConfig


def _clean_proof_text(text: str) -> str:
    """Clean generated proof by truncating at base-model artifacts.

    The Qwen base model, trained on GitHub, frequently appends
    <commit_msg>, <issue_closed>, and other XML-like artifacts
    after generating valid Lean code. We truncate at the first
    sign of these artifacts.

    Also removes trailing imports that the model sometimes appends.
    """
    # Truncate at artifact markers (XML tags from GitHub training data)
    for marker in ["<commit_", "<issue_", "<pull_", "<diff_", "</", "<|im_end|>",
                    "<|endoftext|>", "<|assistant|>", "<|user|>", "<|system|>"]:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]

    # Truncate at "import" / "open" mid-proof.
    # Model glues these to tokens: "rflimport", "_import", "symmimport Data ..."
    m = re.search(r'(?:^|[^a-zA-Z])import\s', text)
    if m:
        start = m.start() + (1 if m.group(0)[0] != 'i' else 0)
        remainder = text[start:]
        if re.match(r'import\s+(Mathlib|Data|Set|Topology|Analysis|Algebra|Geometry|LinearAlgebra|Tactic|Std|Lean)\b', remainder):
            text = text[:start]
    m2 = re.search(r'(?:^|[^a-zA-Z])open\s+[A-Z]', text)
    if m2:
        start2 = m2.start() + (1 if m2.group(0)[0] != 'o' else 0)
        remainder = text[start2:]
        if re.match(r'open\s+(Mathlib|Data|Set|Topology|Analysis|Algebra|Geometry|LinearAlgebra|Tactic|Std|Lean)\b', remainder):
            text = text[:start2]

    return text.strip()


def generate_proofs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    config: GenerationConfig | None = None,
    num_return_sequences: int = 4,
) -> list[list[str]]:
    """Generate K proofs per theorem statement.

    Args:
        model: Policy model.
        tokenizer: Tokenizer.
        prompts: List of theorem prompt strings.
        config: Generation hyperparameters.
        num_return_sequences: K, number of proofs per theorem.

    Returns:
        List of lists: prompts[i] -> [proof_1, proof_2, ..., proof_K]
    """
    if config is None:
        config = GenerationConfig()

    all_proofs: list[list[str]] = []
    device = next(model.parameters()).device

    # Batch all prompts into a single generate call for efficiency.
    # Looping one prompt at a time is ~4× slower on XPU.
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=False,
            return_dict_in_generate=True,
        )

    # Decode each generated sequence and extract just the new tokens.
    # outputs.sequences layout: for each prompt, num_return_sequences results.
    # Shape: [num_prompts * num_return_sequences, seq_len]
    input_ids = inputs["input_ids"]
    for i, prompt_input_ids in enumerate(input_ids):
        proof_texts: list[str] = []
        start = i * num_return_sequences
        end = start + num_return_sequences
        prompt_len = (prompt_input_ids != tokenizer.pad_token_id).sum().item()
        for j in range(start, end):
            new_tokens = outputs.sequences[j][prompt_len:]
            proof_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            proof_texts.append(_clean_proof_text(proof_text.strip()))
        all_proofs.append(proof_texts)

    return all_proofs


def generate_single_proof(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    config: GenerationConfig | None = None,
) -> str:
    """Generate a single proof for interactive testing."""
    if config is None:
        config = GenerationConfig()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
