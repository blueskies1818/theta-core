"""GRPO loss functions and KL penalty computation."""

import torch
import torch.nn.functional as F


def compute_grpo_loss(
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    kl_beta: float = 0.01,
) -> dict:
    """Compute GRPO loss with KL penalty.

    GRPO eliminates the value network by using group-relative advantages.
    Each proof's advantage is relative to other proofs for the SAME theorem.

    Args:
        policy_logprobs: Log probabilities from the policy model.
        reference_logprobs: Log probabilities from the frozen reference model.
        advantages: Group-relative advantages (computed per theorem group).
        kl_beta: KL divergence penalty coefficient.

    Returns:
        Dict with loss, pg_loss, kl_div components for logging.
    """
    # Importance ratio
    log_ratio = policy_logprobs - reference_logprobs
    ratio = torch.exp(torch.clamp(log_ratio, min=-10.0, max=10.0))

    # Policy gradient loss (maximize reward-weighted probability)
    pg_loss = -(ratio * advantages).mean()

    # KL divergence: E[log P - log Q] approximation
    # Using the k1 estimator: 0.5 * (log_ratio)^2 (correct up to O(ratio^3))
    kl_div = 0.5 * (log_ratio**2).mean()

    # Total loss
    loss = pg_loss + kl_beta * kl_div

    return {
        "loss": loss,
        "pg_loss": pg_loss,
        "kl_div": kl_div,
        "ratio_mean": ratio.mean(),
        "ratio_max": ratio.max(),
    }


def compute_sequence_logprob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute sequence-level log probability under the model.

    Returns log P(sequence | model), summed over all tokens.

    Args:
        model: Causal LM.
        input_ids: Token IDs, shape (batch, seq_len).
        attention_mask: Attention mask.

    Returns:
        Log probabilities, shape (batch,).
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )

    # Compute per-token-average log probability for each sequence
    # to keep values comparable across different sequence lengths.
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    mask = attention_mask[:, 1:]
    token_logprobs = token_logprobs * mask

    # Per-token average (not sum) to keep ratios stable
    seq_lens = mask.sum(dim=1).clamp(min=1)
    return token_logprobs.sum(dim=1) / seq_lens
