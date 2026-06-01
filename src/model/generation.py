"""Proof generation with controlled decoding for the mathematical explorer."""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils.config import GenerationConfig


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

    for prompt in prompts:
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
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=False,
                return_dict_in_generate=True,
            )

        # Decode each generated sequence and extract just the new tokens
        input_len = inputs["input_ids"].shape[1]
        proofs = []
        for output_seq in outputs.sequences:
            new_tokens = output_seq[input_len:]
            proof_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            proofs.append(proof_text.strip())

        all_proofs.append(proofs)

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
