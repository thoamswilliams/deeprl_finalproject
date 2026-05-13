from __future__ import annotations

import torch
import torch.nn.functional as F

from llm_rl_final_proj.models.load import PolicyModel


def compute_per_token_logprobs(
    model: PolicyModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    enable_grad: bool = True,
) -> torch.Tensor:
    """Returns log p(x_t | x_<t) for t in [1, L-1]. Shape: [B, L-1]."""
    with torch.set_grad_enabled(enable_grad):
        # DONE(student): run the causal LM, align logits with the next-token targets,
        # and return per-token log-probabilities of the observed tokens.
        # Hint: use F.cross_entropy with reduction='none' for memory efficiency.
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        B, L, V = out.logits.shape

        aligned_logits = out.logits[:, :-1, :]
        flat_logits = aligned_logits.reshape(B*(L-1), V)

        aligned_targets = input_ids[:, 1:]
        flat_targets = aligned_targets.reshape(B*(L-1))

        cross_entropy = F.cross_entropy(flat_logits, flat_targets, reduction='none')
        return -1 * cross_entropy.reshape(B, L-1)
        # ENDDONE


def build_completion_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_input_len: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Mask over per-token positions [B, L-1], selecting completion tokens only."""
    B, L = input_ids.shape
    device = input_ids.device

    token_positions = torch.arange(1, L, device=device)
    completion_toks = token_positions >= prompt_input_len
    is_not_padding = attention_mask[:, 1:].to(torch.bool)

    mask = completion_toks.unsqueeze(0) & is_not_padding
    return mask.to(dtype=torch.float32)
    # END DONE


def masked_sum(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def approx_kl_from_logprobs(
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    log_ratio_clip: float = 20.0,
) -> torch.Tensor:
    """Positive KL proxy from sampled actions.

    Uses estimator: exp(delta) - delta - 1 where delta = log p_ref(a) - log p_new(a).
    """
    # DONE(student): implement the sampled-token KL proxy used throughout the codebase.
    # You should mask out non-completion positions and return a scalar batch mean.
    delta = torch.clamp(ref_logprobs - new_logprobs, - log_ratio_clip, log_ratio_clip)
    kl_by_token = torch.exp(delta) - delta - 1
    kl_mean = masked_mean(kl_by_token, mask, eps)

    return kl_mean
    # DONE
