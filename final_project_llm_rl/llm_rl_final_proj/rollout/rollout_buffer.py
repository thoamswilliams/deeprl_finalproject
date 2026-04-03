from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
    # DONE(student): iterate over the rollout in minibatches, optionally shuffling the row indices,
    # and yield RolloutBatch objects containing the selected subset.
    N = batch.input_ids.shape[0]

    if shuffle:
        iter_order = torch.randperm(N, generator=generator)
    else:
        iter_order = torch.arange(N)

    for i in range(0, N, minibatch_size):
        batch_indices = iter_order[i : i + minibatch_size]

        input_ids = batch.input_ids[batch_indices]
        attention_mask = batch.attention_mask[batch_indices]
        completion_mask = batch.completion_mask[batch_indices]
        old_logprobs = batch.old_logprobs[batch_indices]
        ref_logprobs = batch.ref_logprobs[batch_indices]
        rewards = batch.rewards[batch_indices]
        advantages = batch.advantages[batch_indices]

        if batch.task_names:
            task_names = [batch.task_names[batch_ind] for batch_ind in batch_indices]
        else:
            task_names = None

        if batch.completion_texts:
            completion_texts = [batch.completion_texts[batch_ind] for batch_ind in batch_indices]
        else:
            completion_texts = None

        minibatch = RolloutBatch(input_ids, attention_mask, completion_mask, old_logprobs, ref_logprobs, rewards, advantages, task_names, completion_texts)

        if device:
            minibatch = minibatch.to(device)

        yield minibatch
    # END DONE