"""Configuration loading and validation using Pydantic."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    learning_rate: float = 5e-6
    kl_beta: float = 0.01
    group_size: int = 4
    batch_theorems: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 5000
    save_every: int = 200
    eval_every: int = 200
    log_every: int = 10


class GenerationConfig(BaseModel):
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True


class OptimizerConfig(BaseModel):
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0


class ProofCheckerConfig(BaseModel):
    timeout_seconds: float = 10.0
    max_workers: int = 12
    cache_size: int = 50000


class ReplayBufferConfig(BaseModel):
    max_size: int = 10000
    sample_batch_size: int = 16


class GRPOConfig(BaseModel):
    training: TrainingConfig = TrainingConfig()
    generation: GenerationConfig = GenerationConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    proof_checker: ProofCheckerConfig = ProofCheckerConfig()
    replay_buffer: ReplayBufferConfig = ReplayBufferConfig()


class SFTDataConfig(BaseModel):
    train_split: float = 0.9
    max_theorems: Optional[int] = None
    filter_domains: list[str] = ["Analysis", "Geometry/Manifold", "Topology", "LinearAlgebra", "GroupTheory"]


class SFTTrainingConfig(BaseModel):
    num_epochs: int = 2
    learning_rate: float = 2e-5
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.05
    weight_decay: float = 0.1


class SFTOptimizerConfig(BaseModel):
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0


class SFTConfig(BaseModel):
    training: SFTTrainingConfig = SFTTrainingConfig()
    optimizer: SFTOptimizerConfig = SFTOptimizerConfig()
    data: SFTDataConfig = SFTDataConfig()


class BaseModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"


class LoRAConfig(BaseModel):
    use_lora: bool = False
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class PrecisionConfig(BaseModel):
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True


class ModelConfig(BaseModel):
    base_model: BaseModelConfig = BaseModelConfig()
    lora: LoRAConfig = LoRAConfig()
    precision: PrecisionConfig = PrecisionConfig()


class BaseRewardConfig(BaseModel):
    valid_proof: float = 1.0
    invalid_proof: float = 0.0


class LengthBonusConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.1
    reference_tokens: int = 100
    decay_rate: float = 0.002


class RewardConfig(BaseModel):
    base_reward: BaseRewardConfig = BaseRewardConfig()
    length_bonus: LengthBonusConfig = LengthBonusConfig()


def load_yaml_config(path: Path, model_cls: type[BaseModel]) -> BaseModel:
    """Load a YAML config file and validate it with Pydantic."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return model_cls(**data)


def load_grpo_config(path: Path = None) -> GRPOConfig:
    if path is None:
        path = Path(__file__).parent.parent.parent / "configs" / "grpo_config.yaml"
    return load_yaml_config(path, GRPOConfig)


def load_sft_config(path: Path = None) -> SFTConfig:
    if path is None:
        path = Path(__file__).parent.parent.parent / "configs" / "sft_config.yaml"
    return load_yaml_config(path, SFTConfig)


def load_model_config(path: Path = None) -> ModelConfig:
    if path is None:
        path = Path(__file__).parent.parent.parent / "configs" / "model_config.yaml"
    return load_yaml_config(path, ModelConfig)


def load_reward_config(path: Path = None) -> RewardConfig:
    if path is None:
        path = Path(__file__).parent.parent.parent / "configs" / "reward_config.yaml"
    return load_yaml_config(path, RewardConfig)
