"""Model loading and configuration for Intel XPU (and CUDA fallback)."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.utils.config import ModelConfig
from src.utils.xpu_utils import get_device


def load_model_and_tokenizer(
    config: ModelConfig | None = None,
    device: torch.device | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load base model and tokenizer with configurable precision and device.

    Args:
        config: Model configuration. Uses defaults if None.
        device: Target device. Uses auto-detection if None.

    Returns:
        Tuple of (model, tokenizer).
    """
    if config is None:
        config = ModelConfig()

    if device is None:
        device = get_device()

    torch_dtype = getattr(torch, config.precision.mixed_precision)
    model_name = config.base_model.name

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"Dtype: {config.precision.mixed_precision}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "attn_implementation": config.base_model.attn_implementation,
    }

    if device.type == "xpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            **model_kwargs,
        )
    elif device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            **model_kwargs,
        )

    if config.precision.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def load_model_for_sft(
    config: ModelConfig | None = None,
    device: torch.device | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model configured for supervised fine-tuning."""
    model, tokenizer = load_model_and_tokenizer(config, device)
    model.train()
    return model, tokenizer


def load_model_for_grpo(
    config: ModelConfig | None = None,
    device: torch.device | None = None,
) -> tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]:
    """Load policy and reference models for GRPO training.

    Returns (policy_model, reference_model, tokenizer).
    The reference model is frozen and used for KL divergence penalty.
    """
    policy_model, tokenizer = load_model_and_tokenizer(config, device)
    policy_model.train()

    # Clone for reference model (frozen)
    print("Loading reference model...")
    if config is None:
        config = ModelConfig()
    torch_dtype = getattr(torch, config.precision.mixed_precision)

    ref_model = AutoModelForCausalLM.from_pretrained(
        config.base_model.name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if device.type in ("xpu", "cuda"):
        ref_model = ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return policy_model, ref_model, tokenizer


def apply_lora(
    model: PreTrainedModel,
    config: ModelConfig,
) -> PreTrainedModel:
    """Apply LoRA adapters to the model for memory-efficient training."""
    from peft import LoraConfig, get_peft_model, TaskType

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model
