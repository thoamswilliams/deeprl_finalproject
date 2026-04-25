"""Implements PET Finetuning from https://arxiv.org/abs/2505.20556"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_rl_final_proj.data.ultrafeedback import (
    UltraFeedbackPreferenceDataset,
    build_preference_examples,
    dataset_overview,
    GenerationExample,
)
from llm_rl_final_proj.models.load import load_trainable_lora_reward_model_and_tokenizer, load_lora_policy_model_and_tokenizer
from llm_rl_final_proj.reward_model import RewardPairCollator, evaluate_reward_model_dataset, RewardScoringCollator
from llm_rl_final_proj.reward_model.train import RewardModelConfig
from llm_rl_final_proj.utils.hardware import (
    get_cuda_memory_metrics,
    get_hardware_metrics,
    get_model_device_metrics,
    require_cuda_if_requested,
    resolve_device_and_dtype,
)
from llm_rl_final_proj.utils.seed import set_seed
from llm_rl_final_proj.utils.wandb_utils import WandBLogger

from llm_rl_final_proj.rollout.hf_sampler import HFSampler, SamplingConfig


@dataclass
class FineTuneConfig:
    initial_model_path: str = "/vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445"
    initial_model_config: str = "/vol/runs/wildchat_min4_judged_5k_reward_model_v1/config.json"
    base_policy_model_name: str ="Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "runs/reward_model_finetuned"

    seed: int = 0
    num_train_epochs: float = 1.0

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    grad_accum_steps: int = 16

    max_prompt_tokens: int = 512
    max_response_tokens: int = 256

    max_grad_norm: float = 1.0

    eval_interval: int = 25
    save_interval: int = 50

    # TODO: ADD TO CMD LINE ARGS
    num_samples: int = 32
    pessimistic_coef: float = 10.0
    lr: float = 3e-5

    wandb_project: str = "llm-rl-final-project"
    wandb_name: str = "reward_model_finetuned_PET_2"
    wandb_enabled: bool = True


    # TODO: ADD TO CMD LINE ARGS
    # sampler config
    s_min_new_tokens: int = 8
    s_max_new_tokens: int = 256
    s_temperature: float = 0.9
    s_reward_batch_size: int = 16
    s_top_p: float = 0.95
    s_top_k: int = 0
    s_repetition_penalty: float = 1.0
    s_batch_size: int = 4

    # TODO: ADD TO CMD LINE ARGS
    # policy config
    p_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    p_lora_r: int = 32
    p_lora_alpha: int = 64
    p_lora_dropout: float = 0.05
    p_lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    p_lora_bias: str = "none"
    p_grad_checkpointing: bool = True


def parse_args() -> FineTuneConfig:
    ap = argparse.ArgumentParser(description="Fine-Tune a Bradley-Terry reward model on preference pairs. (Xu et al.)")
    ap.add_argument("--initial_model_path", type=str, default=FineTuneConfig.initial_model_path)
    ap.add_argument("--initial_model_config", type=str, default=FineTuneConfig.initial_model_config)
    ap.add_argument("--output_dir", type=str, default=FineTuneConfig.output_dir)
    ap.add_argument("--base_policy_model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")

    ap.add_argument("--seed", type=int, default=FineTuneConfig.seed)
    ap.add_argument("--num_train_epochs", type=float, default=FineTuneConfig.num_train_epochs)

    ap.add_argument("--per_device_train_batch_size", type=int, default=FineTuneConfig.per_device_train_batch_size)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=FineTuneConfig.per_device_eval_batch_size)
    ap.add_argument("--grad_accum_steps", type=int, default=FineTuneConfig.grad_accum_steps)
    ap.add_argument("--lr", type=float, default=FineTuneConfig.lr)

    ap.add_argument("--max_prompt_tokens", type=int, default=FineTuneConfig.max_prompt_tokens)
    ap.add_argument("--max_response_tokens", type=int, default=FineTuneConfig.max_response_tokens)

    ap.add_argument("--eval_interval", type=int, default=FineTuneConfig.eval_interval)
    ap.add_argument("--save_interval", type=int, default=FineTuneConfig.save_interval)

    ap.add_argument("--wandb_project", type=str, default=FineTuneConfig.wandb_project)
    ap.add_argument("--wandb_name", type=str, default=FineTuneConfig.wandb_name)
    ap.add_argument(
        "--wandb_enabled",
        action=argparse.BooleanOptionalAction,
        default=FineTuneConfig.wandb_enabled,
    )
    args = ap.parse_args()
    return FineTuneConfig(**vars(args))

def reward_model_scores(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits
    if logits.ndim == 2 and logits.shape[-1] == 1:
        return logits[:, 0]
    if logits.ndim == 1:
        return logits
    raise ValueError(f"Unexpected reward-model logits shape: {tuple(logits.shape)}")

def score_prompt_response_pairs_with_grad(
    model: torch.nn.Module,
    tokenizer,
    rows: Sequence[Dict[str, object]],
    *,
    max_prompt_tokens: int,
    max_response_tokens: int,
    per_device_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    collator = RewardScoringCollator(
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_response_tokens=max_response_tokens,
    )
    chunks = []
    for start in range(0, len(rows), per_device_batch_size):
        batch = collator(list(rows[start : start + per_device_batch_size])).to(device)
        chunks.append(reward_model_scores(
            model,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        ))
    return torch.cat(chunks, dim=0)

def _normalize_lora_target_modules(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]

def _normalize_completion_for_reward_scoring(text: str) -> str:
    if text.strip():
        return text
    return "[no response]"

def _sample_batch(examples: Sequence[GenerationExample], batch_size: int, rng: random.Random) -> List[GenerationExample]:
    if not examples:
        raise RuntimeError("Cannot sample prompts from an empty generation split.")
    return [examples[rng.randrange(len(examples))] for _ in range(batch_size)]

def save_checkpoint(model: torch.nn.Module, ft_cfg: FineTuneConfig, base_cfg: RewardModelConfig, step: int) -> None:
    ckpt_dir = Path(ft_cfg.output_dir) / "checkpoints" / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = ckpt_dir / "adapter"
    model.save_pretrained(adapter_dir)
    meta = {
        "step": step,
        "model_type": "finetuned_reward_model",
        "base_model_name": base_cfg.model_name,
        "dataset_name": base_cfg.dataset_name,
        "train_split": base_cfg.train_split,
        "eval_split": base_cfg.eval_split,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _compute_pair_metrics(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> Dict[str, float]:
    margins = (chosen_scores - rejected_scores)
    loss = - torch.nn.functional.logsigmoid(margins).mean()
    return {
        "loss_tensor": loss,
        "reward_model/bt_loss": float(loss.detach().item()),
        "reward_model/bt_pair_accuracy": float((margins.detach() > 0).float().mean().item()),
        "reward_model/bt_margin_mean": float(margins.detach().mean().item()),
        "reward_model/bt_chosen_score_mean": float(chosen_scores.detach().mean().item()),
        "reward_model/bt_rejected_score_mean": float(rejected_scores.detach().mean().item()),
    }


def main() -> None:
    ft_cfg = parse_args()

    with open(ft_cfg.initial_model_config) as f:
        base_cfg = RewardModelConfig(**json.load(f))

    set_seed(ft_cfg.seed)
    require_cuda_if_requested()
    rng = random.Random(ft_cfg.seed)

    output_dir = Path(ft_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_fine_tuned_reward_model_config.json").write_text(
        json.dumps(vars(ft_cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    device, dtype = resolve_device_and_dtype()

    print(f"[setup] device={device} dtype={dtype} model={base_cfg.model_name}")
    print(f"[setup] loadingx dataset={base_cfg.dataset_name} train_split={base_cfg.train_split} eval_split={base_cfg.eval_split}")
    print("[setup][hardware]", json.dumps(get_hardware_metrics(device), indent=2, sort_keys=True))

    # TODO: Should we copy same train test split from training time?
    dataset_info = dataset_overview(base_cfg.dataset_name)
    train_examples = build_preference_examples(base_cfg.dataset_name, base_cfg.train_split, limit=base_cfg.train_limit)
    eval_examples = build_preference_examples(base_cfg.dataset_name, base_cfg.eval_split, limit=base_cfg.eval_limit)
    if not train_examples:
        raise RuntimeError("Training split produced zero examples.")
    if not eval_examples:
        raise RuntimeError("Evaluation split produced zero examples.")
    
    # TODO: Should we be using same Lora config as pretraining here?
    rm_loaded = load_trainable_lora_reward_model_and_tokenizer(
        base_cfg.model_name,
        device=device,
        dtype=dtype,
        grad_checkpointing=base_cfg.grad_checkpointing,
        lora_r=base_cfg.lora_r,
        lora_alpha=base_cfg.lora_alpha,
        lora_dropout=base_cfg.lora_dropout,
        lora_target_modules=_normalize_lora_target_modules(base_cfg.lora_target_modules),
        lora_bias=base_cfg.lora_bias,
        adapter_path=ft_cfg.initial_model_path + "/adapter"
    )
    rm_model = rm_loaded.model
    rm_tokenizer = rm_loaded.tokenizer

    optimizer = torch.optim.AdamW(
        [p for p in rm_model.parameters() if p.requires_grad],
        lr=ft_cfg.lr,
    )

    batches_per_epoch = math.ceil(len(train_examples) / ft_cfg.s_batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / ft_cfg.grad_accum_steps)
    total_steps = int(steps_per_epoch * ft_cfg.num_train_epochs)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # We use this later to generate batches to calculate non-rollout rewards
    collator = RewardPairCollator(
        rm_tokenizer,
        max_prompt_tokens=ft_cfg.max_prompt_tokens,
        max_response_tokens=ft_cfg.max_response_tokens,
    )

    loaded_policy = load_lora_policy_model_and_tokenizer(
        ft_cfg.p_model_name,
        device=device,
        dtype=dtype,
        grad_checkpointing=ft_cfg.p_grad_checkpointing,
        lora_r=ft_cfg.p_lora_r,
        lora_alpha=ft_cfg.p_lora_alpha,
        lora_dropout=ft_cfg.p_lora_dropout,
        lora_target_modules=_normalize_lora_target_modules(ft_cfg.p_lora_target_modules),
        lora_bias=ft_cfg.p_lora_bias,
    )
    policy_model = loaded_policy.model
    policy_tokenizer = loaded_policy.tokenizer

    # We use this to sample from our base policy to calculate pessimistic reward
    sampler = HFSampler(policy_tokenizer, device=device)
    sampling_cfg = SamplingConfig(
        min_new_tokens=ft_cfg.s_min_new_tokens,
        max_new_tokens=ft_cfg.s_max_new_tokens,
        temperature=ft_cfg.s_temperature,
        top_p=ft_cfg.s_top_p,
        top_k=ft_cfg.s_top_k,
        repetition_penalty=ft_cfg.s_repetition_penalty,
        do_sample=ft_cfg.s_temperature > 0.0,
    )

    logger = WandBLogger(
        project=ft_cfg.wandb_project,
        run_name=ft_cfg.wandb_name,
        config=vars(ft_cfg),
        enabled=ft_cfg.wandb_enabled, #ft_cfg.wandb_enabled,
        local_dir=output_dir,
    )
    logger.log(
        {
            "setup/trainable_params": float(rm_loaded.trainable_params),
            "setup/total_params": float(rm_loaded.total_params),
            "setup/trainable_fraction": float(rm_loaded.trainable_params / max(1, rm_loaded.total_params)),
            "dataset/train_examples": float(len(train_examples)),
            "dataset/eval_examples": float(len(eval_examples)),
            **{f"dataset/{k}": float(v) for k, v in dataset_info["splits"].items()},
            **get_hardware_metrics(device),
            **get_model_device_metrics(rm_model),
        },
        step=0,
    )

    # TODO: Is this the right evaluation suite to test the rm on? Given we are finetuning it to the base policy?
    def run_eval(step: int, phase: str) -> Dict[str, float]:
        rm_model.eval()
        eval_metrics = evaluate_reward_model_dataset(
            rm_model,
            rm_tokenizer,
            eval_examples,
            max_prompt_tokens=base_cfg.max_prompt_tokens,
            max_response_tokens=base_cfg.max_response_tokens,
            per_device_eval_batch_size=ft_cfg.per_device_eval_batch_size,
            device=device,
            desc=f"eval[reward_model|{phase}]",
        )
        eval_metrics["eval/step"] = float(step)
        logger.log(eval_metrics, step=step)
        rm_model.train()
        return eval_metrics

    baseline_eval = run_eval(step=0, phase="baseline")
    print("[eval][baseline]", json.dumps(baseline_eval, indent=2, sort_keys=True))

    rm_model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    microbatch_count = 0
    train_start = time.perf_counter()

    progress = tqdm(total=total_steps, desc="fine_tune[reward_model]", dynamic_ncols=True)
    while optimizer_step < total_steps:
        batch = _sample_batch(train_examples, ft_cfg.s_batch_size, rng)
        # Put refrence completions into a form score_prompt_response_pairs understands
        ref_reward_rows = []
        for i, pref_example in enumerate(batch):
            ref_reward_rows.append(
                {
                    "row_id": f"{pref_example.row_id}:{i}",
                    "prompt_messages": pref_example.prompt_messages,
                    "prompt_text": pref_example.prompt_text,
                    "response_text": _normalize_completion_for_reward_scoring(pref_example.chosen_text),
                }
            )
        # Score refrence completions
        rm_model.eval() # Disable dropout for this calculation
        ref_rewards = score_prompt_response_pairs_with_grad(
            rm_model,
            rm_tokenizer,
            ref_reward_rows,
            max_prompt_tokens=ft_cfg.max_prompt_tokens,
            max_response_tokens=ft_cfg.max_response_tokens,
            per_device_batch_size=ft_cfg.s_reward_batch_size,
            device=device,
        )
        rm_model.train()

        # Rollout from our base policy
        rollout = sampler.rollout(
            policy_model=policy_model,
            prompt_messages=[ex.prompt_messages for ex in batch],
            task_names=["synthetic_instruction_following"] * len(batch),
            task_metas=[
                {
                    "row_id": ex.row_id,
                    "prompt_text": ex.prompt_text,
                    "reference_response_text": ex.chosen_text,
                }
                for ex in batch
            ],
            group_size=ft_cfg.num_samples, # This represents the number of time we rejection sample each prompt
            sampling=sampling_cfg,
            max_prompt_tokens=ft_cfg.max_prompt_tokens,
            output_to_cpu=False,
            compute_logprobs=False
        )

        # Put rollout completions into a form that score_prompt_response_pair understands
        policy_reward_rows = []
        for i, completion_text in enumerate(rollout.completion_texts):
            meta = rollout.task_metas[i]
            policy_reward_rows.append(
                {
                    "row_id": f"{meta.get('row_id', i)}:{i}",
                    "prompt_messages": rollout.prompt_messages[i],
                    "prompt_text": str(meta.get("prompt_text", "")),
                    "response_text": _normalize_completion_for_reward_scoring(completion_text),
                }
            )
        del rollout
        # Score policy completions
        rm_model.eval() # Disable dropout for this calculation
        with torch.no_grad(): # No grad pass for intital rejection sampling
            policy_selection_rewards = score_prompt_response_pairs_with_grad(
                rm_model,
                rm_tokenizer,
                policy_reward_rows,
                max_prompt_tokens=ft_cfg.max_prompt_tokens,
                max_response_tokens=ft_cfg.max_response_tokens,
                per_device_batch_size=ft_cfg.s_reward_batch_size,
                device=device,
            )

        policy_winner_ids = policy_selection_rewards.reshape(-1, ft_cfg.num_samples).argmax(dim=-1)
        policy_winner_rows = [policy_reward_rows[prompt_i * ft_cfg.num_samples + k.item()] for prompt_i, k in enumerate(policy_winner_ids)]

        # Pass with grads only on winning samples
        # This prevents OOM errors
        rejection_sampled_policy_rewards = score_prompt_response_pairs_with_grad(
                rm_model,
                rm_tokenizer,
                policy_winner_rows,
                max_prompt_tokens=ft_cfg.max_prompt_tokens,
                max_response_tokens=ft_cfg.max_response_tokens,
                per_device_batch_size=ft_cfg.s_reward_batch_size,
                device=device,
            )
        rm_model.train()

        # Pessimistic loss component
        pess_loss = torch.mean(rejection_sampled_policy_rewards - ref_rewards)

        # Get a batch form that our reward model can understand
        rm_batch = collator(batch).to(device)
        
        # Get regular BT loss 
        chosen_scores = rm_model(
            input_ids=rm_batch.chosen_input_ids,
            attention_mask=rm_batch.chosen_attention_mask,
            use_cache=False,
        ).logits[:, 0]
        rejected_scores = rm_model(
            input_ids=rm_batch.rejected_input_ids,
            attention_mask=rm_batch.rejected_attention_mask,
            use_cache=False,
        ).logits[:, 0]
        metrics = _compute_pair_metrics(chosen_scores, rejected_scores)
        # Bradley terry loss times our pessimistic coeficent
        bt_loss = ft_cfg.pessimistic_coef * metrics.pop("loss_tensor")
        loss = pess_loss + bt_loss
        (loss / ft_cfg.grad_accum_steps).backward()
        microbatch_count += 1

        if microbatch_count % ft_cfg.grad_accum_steps != 0:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(rm_model.parameters(), ft_cfg.max_grad_norm).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        scheduler.step()
        progress.update(1)

        logger.log(
            {
                "train/optimizer_step": float(optimizer_step),
                "train/microbatch_count": float(microbatch_count),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "train/grad_norm": grad_norm,
                "train/pess_loss": pess_loss.detach().item(),
                "train/loss": loss.detach().item(),
                **{f"train/{k}": v for k, v in metrics.items()},
                **get_cuda_memory_metrics(prefix="train"),
            },
            step=optimizer_step,
        )
        progress.set_postfix(
            loss=f"{metrics['reward_model/bt_loss']:.3f}",
            acc=f"{metrics['reward_model/bt_pair_accuracy']:.3f}",
        )

        if ft_cfg.eval_interval > 0 and optimizer_step % ft_cfg.eval_interval == 0:
            run_eval(step=optimizer_step, phase="periodic")
        if ft_cfg.save_interval > 0 and optimizer_step % ft_cfg.save_interval == 0:
            save_checkpoint(rm_model, ft_cfg, base_cfg, optimizer_step)

    progress.close()
    save_checkpoint(rm_model, ft_cfg, base_cfg, optimizer_step)
    final_eval = run_eval(step=optimizer_step, phase="final")
    elapsed = max(1e-6, time.perf_counter() - train_start)
    logger.log(
        {
            "train/elapsed_seconds": elapsed,
            "train/optimizer_steps_completed": float(optimizer_step),
            "train/optimizer_steps_per_second": float(optimizer_step / elapsed),
        },
        step=optimizer_step,
    )
    logger.finish()

    print("[eval][final]", json.dumps(final_eval, indent=2, sort_keys=True))
    print(f"[done] optimizer_steps={optimizer_step} elapsed_seconds={elapsed:.1f}")

if __name__ == "__main__":
    main()
