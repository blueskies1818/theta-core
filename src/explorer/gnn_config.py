"""Configuration for the GNN encoder (Phase 2.2)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GNNConfig:
    """Configuration for the Graph Neural Network encoder.

    The GNN learns node embeddings over the math dependency graph.
    These embeddings are used by MCTS (Phase 2.3) to evaluate which
    lemmas to apply at each proof step.
    """

    # -- Architecture --------------------------------------------------------

    # Number of message-passing layers (depth of neighbor aggregation)
    # Scaled from 3 → 5 for deeper structural reasoning
    num_layers: int = 5

    # Hidden dimension for node embeddings
    # Scaled from 256 → 768 (~10M params) for lemma discrimination
    hidden_dim: int = 768

    # Input feature dimension (from statement embedding or random init)
    input_dim: int = 768

    # Number of attention heads in GAT layers
    # Scaled from 8 → 12 (768/12 = 64-dim per head)
    num_heads: int = 12

    # Dropout rate for regularization
    dropout: float = 0.1

    # Activation function
    activation: str = "gelu"  # "gelu", "relu", "silu"

    # -- Goal Encoder --------------------------------------------------------
    # The goal encoder projects keyword-averaged lemma embeddings into a
    # learned goal embedding space. Trained jointly with the GNN during
    # both pretraining and GRPO.

    # Whether to use a learned goal encoder (vs. raw keyword-average)
    use_goal_encoder: bool = True

    # Goal encoder hidden dimension multiplier (2× → 2-layer MLP)
    goal_encoder_expansion: int = 2

    # Goal encoder dropout
    goal_encoder_dropout: float = 0.1

    # -- Message Passing -----------------------------------------------------

    # Whether to use edge-type-specific projections
    # (different weights for "uses_in_proof" vs "uses_in_statement")
    use_edge_types: bool = True

    # Number of distinct edge types
    num_edge_types: int = 4

    # Whether to use bidirectional message passing
    # (if True, messages also flow from dependencies → dependents)
    bidirectional: bool = True

    # Whether to add a virtual "supernode" connected to all nodes
    # (improves global information flow, like a [CLS] token)
    use_supernode: bool = False

    # -- Initial Features ----------------------------------------------------

    # How to initialize node features:
    # "transformer" - use the Phase 1 model to embed theorem statements
    # "random" - random initialization, learned during GNN training
    # "onehot" - one-hot encoding of domain + node type
    init_features: str = "random"

    # Path to the SFT model checkpoint (for transformer init)
    sft_checkpoint: str = "checkpoints/sft_v2/best"

    # Base model name (for tokenizer when using transformer init)
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # -- Training ------------------------------------------------------------

    # Learning rate for GNN training
    learning_rate: float = 1e-3

    # Weight decay for Adam optimizer
    weight_decay: float = 1e-5

    # Number of training epochs
    num_epochs: int = 100

    # Batch size for mini-batch training (number of target nodes)
    batch_size: int = 512

    # Number of neighbors to sample per node during training
    # One entry per GNN layer (5 layers in scaled model)
    num_neighbors: list[int] = field(default_factory=lambda: [25, 15, 10, 5, 3])

    # Training objective:
    # "link_prediction" - predict whether an edge exists between two nodes
    # "node_classification" - predict domain/type from embeddings
    # "contrastive" - InfoNCE between local and global representations
    objective: str = "link_prediction"

    # Negative sampling ratio for link prediction
    negative_ratio: int = 5

    # Temperature for contrastive loss
    temperature: float = 0.07

    # -- Inference -----------------------------------------------------------

    # Whether to use full-graph or sampled inference
    full_graph_inference: bool = True

    # Device to run on
    device: str = "cpu"

    # Mixed precision
    use_amp: bool = True
