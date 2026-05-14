# COMMANDS
Log all commands used here and what they were used for.
## Part 1 Comamnds:
### 9.1 Reward Model
Intitial Reward Model Training (Before Assignment Updates)
```bash
uv run modal run -d scripts/modal_train.py::reward_model_train_remote -- \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_prefs \
--eval_split test_prefs \
--output_dir /vol/runs/wildchat_min4_judged_5k_reward_model_v1
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--grad_accum_steps 4 \
--lr 3e-5 \
--num_train_epochs 3
--max_prompt_tokens 700 \
--max_response_tokens 512 \
--eval_interval 25 \
--save_interval 50 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name wildchat_min4_judged_5k_reward_model_v1  \
```
## Finetuning PET RM
### Baseline Runs, To Establish Our Method Consistently Works
#### RM1 finetuning, Seed 42:
```bash
uv run modal run -d scripts/modal_train.py::reward_model_finetune_remote -- \
--initial_model_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445 \
--initial_model_config /vol/runs/wildchat_min4_judged_5k_reward_model_v1/config.json \
--base_policy_model_name Qwen/Qwen2.5-1.5B-Instruct \
--output_dir runs/reward_model/rm_finetuned_PET_v3_seed42 \
--seed 42 \
--num_train_epochs 1.0 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--grad_accum_steps 16 \
--max_prompt_tokens 512 \
--max_response_tokens 256 \
--max_grad_norm 1.0 \
--eval_interval 25 \
--save_interval 50 \
--num_samples 32 \
--pessimistic_coef 10.0 \
--lr 3e-5 \
--wandb_project llm-rl-final-project \
--wandb_name reward_model_finetuned_PET_v3_seed42 \
--wandb_enabled \
--s_min_new_tokens 8 \
--s_max_new_tokens 256 \
--s_temperature 0.9 \
--s_reward_batch_size 16 \
--s_top_p 0.95 \
--s_top_k 0 \
--s_repetition_penalty 1.0 \
--s_batch_size 4 \
--p_model_name Qwen/Qwen2.5-1.5B-Instruct \
--p_lora_r 32 \
--p_lora_alpha 64 \
--p_lora_dropout 0.05 \
--p_lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
--p_lora_bias none \
--p_grad_checkpointing
```
GRPO seed 42, w/ RM1:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/rm_finetuned_PET_v3_seed42/checkpoints/step_000075/adapter \
--output_dir /vol/runs/PET/grpo_PET_RM1_v2_long_seed42 \
--seed 42 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_PET_RM1_v2_long_seed42
```

GRPO seed 43, w/ RM1:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/rm_finetuned_PET_v3_seed42/checkpoints/step_000075/adapter \
--output_dir /vol/runs/PET/grpo_PET_RM1_v2_long_seed43 \
--seed 43 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_PET_RM1_v2_long_seed43
```

GRPO seed 44, w/ RM1:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/rm_finetuned_PET_v3_seed42/checkpoints/step_000075/adapter \
--output_dir /vol/runs/PET/grpo_PET_RM1_v2_long_seed44 \
--seed 44 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_PET_RM1_v2_long_seed44
```

#### RM2 finetuning, Seed 43:
```bash
uv run modal run -d scripts/modal_train.py::reward_model_finetune_remote -- \
--initial_model_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445 \
--initial_model_config /vol/runs/wildchat_min4_judged_5k_reward_model_v1/config.json \
--base_policy_model_name Qwen/Qwen2.5-1.5B-Instruct \
--output_dir runs/reward_model/rm_finetuned_PET_v3_seed43 \
--seed 43 \
--num_train_epochs 1.0 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--grad_accum_steps 16 \
--max_prompt_tokens 512 \
--max_response_tokens 256 \
--max_grad_norm 1.0 \
--eval_interval 25 \
--save_interval 50 \
--num_samples 32 \
--pessimistic_coef 10.0 \
--lr 3e-5 \
--wandb_project llm-rl-final-project \
--wandb_name reward_model_finetuned_PET_v3_seed43 \
--wandb_enabled \
--s_min_new_tokens 8 \
--s_max_new_tokens 256 \
--s_temperature 0.9 \
--s_reward_batch_size 16 \
--s_top_p 0.95 \
--s_top_k 0 \
--s_repetition_penalty 1.0 \
--s_batch_size 4 \
--p_model_name Qwen/Qwen2.5-1.5B-Instruct \
--p_lora_r 32 \
--p_lora_alpha 64 \
--p_lora_dropout 0.05 \
--p_lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
--p_lora_bias none \
--p_grad_checkpointing
```

GRPO seed 42, w/ RM2:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/rm_finetuned_PET_v3_seed43/checkpoints/step_000075/adapter \
--output_dir /vol/runs/PET/grpo_PET_RM2_v2_long_seed42 \
--seed 42 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_PET_RM2_v2_long_seed42
```

GRPO seed 43, w/ RM2:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/rm_finetuned_PET_v3_seed43/checkpoints/step_000075/adapter \
--output_dir /vol/runs/PET/grpo_PET_RM2_v2_long_seed43 \
--seed 43 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_PET_RM2_v2_long_seed43
```

GRPO seed 44, w/ RM2:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/rm_finetuned_PET_v3_seed43/checkpoints/step_000075/adapter \
--output_dir /vol/runs/PET/grpo_PET_RM2_v2_long_seed44 \
--seed 44 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_PET_RM2_v2_long_seed44
```

#### RM3 finetuning, Seed 44:
```bash
uv run modal run -d scripts/modal_train.py::reward_model_finetune_remote -- \
--initial_model_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445 \
--initial_model_config /vol/runs/wildchat_min4_judged_5k_reward_model_v1/config.json \
--base_policy_model_name Qwen/Qwen2.5-1.5B-Instruct \
--output_dir runs/reward_model/rm_finetuned_PET_v3_seed44 \
--seed 44 \
--num_train_epochs 1.0 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--grad_accum_steps 16 \
--max_prompt_tokens 512 \
--max_response_tokens 256 \
--max_grad_norm 1.0 \
--eval_interval 25 \
--save_interval 50 \
--num_samples 32 \
--pessimistic_coef 10.0 \
--lr 3e-5 \
--wandb_project llm-rl-final-project \
--wandb_name reward_model_finetuned_PET_v3_seed44 \
--wandb_enabled \
--s_min_new_tokens 8 \
--s_max_new_tokens 256 \
--s_temperature 0.9 \
--s_reward_batch_size 16 \
--s_top_p 0.95 \
--s_top_k 0 \
--s_repetition_penalty 1.0 \
--s_batch_size 4 \
--p_model_name Qwen/Qwen2.5-1.5B-Instruct \
--p_lora_r 32 \
--p_lora_alpha 64 \
--p_lora_dropout 0.05 \
--p_lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
--p_lora_bias none \
--p_grad_checkpointing
```

GRPO seed 42, w/ RM3:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/rm_finetuned_PET_v3_seed44/checkpoints/step_000075/adapter \
--output_dir /vol/runs/PET/grpo_PET_RM3_v2_long_seed42 \
--seed 42 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_PET_RM3_v2_long_seed42
```

GRPO seed 43, w/ RM3:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/rm_finetuned_PET_v3_seed44/checkpoints/step_000075/adapter \
--output_dir /vol/runs/PET/grpo_PET_RM3_v2_long_seed43 \
--seed 43 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_PET_RM3_v2_long_seed43
```

GRPO seed 44, w/ RM3:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/rm_finetuned_PET_v3_seed44/checkpoints/step_000075/adapter \
--output_dir /vol/runs/PET/grpo_PET_RM3_v2_long_seed44 \
--seed 44 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_PET_RM3_v2_long_seed44
```
### Generating Completions to Score Model Preformance
These are bash loops to run the modal scripts to build policy submissions for all above runs
```bash
uv run modal run -d scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path "/vol/runs/PET/grpo_PET_RM1_v2_long_seed42/checkpoints/step_000100/adapter" \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl "/vol/judge/PET/GRPO/grpo_PET_RM1_v2_long_seed42.jsonl" \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0

uv run modal run -d scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path "/vol/runs/PET/grpo_PET_RM1_v2_long_seed43/checkpoints/step_000100/adapter" \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl "/vol/judge/PET/GRPO/grpo_PET_RM1_v2_long_seed43.jsonl" \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0

uv run modal run -d scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path "/vol/runs/PET/grpo_PET_RM1_v2_long_seed44/checkpoints/step_000100/adapter" \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl "/vol/judge/PET/GRPO/grpo_PET_RM1_v2_long_seed44.jsonl" \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0

uv run modal run -d scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path "/vol/runs/PET/grpo_PET_RM2_v2_long_seed42/checkpoints/step_000100/adapter" \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl "/vol/judge/PET/GRPO/grpo_PET_RM2_v2_long_seed42.jsonl" \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0

uv run modal run -d scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path "/vol/runs/PET/grpo_PET_RM2_v2_long_seed43/checkpoints/step_000100/adapter" \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl "/vol/judge/PET/GRPO/grpo_PET_RM2_v2_long_seed43.jsonl" \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0

uv run modal run -d scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path "/vol/runs/PET/grpo_PET_RM2_v2_long_seed44/checkpoints/step_000100/adapter" \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl "/vol/judge/PET/GRPO/grpo_PET_RM2_v2_long_seed44.jsonl" \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0

uv run modal run -d scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path "/vol/runs/PET/grpo_PET_RM3_v2_long_seed42/checkpoints/step_000100/adapter" \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl "/vol/judge/PET/GRPO/grpo_PET_RM3_v2_long_seed42.jsonl" \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0

uv run modal run -d scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path "/vol/runs/PET/grpo_PET_RM3_v2_long_seed43/checkpoints/step_000100/adapter" \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl "/vol/judge/PET/GRPO/grpo_PET_RM3_v2_long_seed43.jsonl" \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0

uv run modal run -d scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path "/vol/runs/PET/grpo_PET_RM3_v2_long_seed44/checkpoints/step_000100/adapter" \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl "/vol/judge/PET/GRPO/grpo_PET_RM3_v2_long_seed44.jsonl" \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0
```

```bash
uv run modal run -d scripts/modal_train.py::build_reward_model_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path "/vol/runs/reward_model/rm_finetuned_PET_v3_seed42/checkpoints/step_000075/adapter" \
  --prefs_jsonl /root/project/public_eval/public_test_prefs_256.jsonl \
  --output_jsonl "/vol/judge/PET/RM/rm_finetuned_PET_v3_seed42.jsonl" \
  --max_prompt_tokens 700 \
  --max_response_tokens 512

uv run modal run -d scripts/modal_train.py::build_reward_model_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path "/vol/runs/reward_model/rm_finetuned_PET_v3_seed43/checkpoints/step_000075/adapter" \
  --prefs_jsonl /root/project/public_eval/public_test_prefs_256.jsonl \
  --output_jsonl "/vol/judge/PET/RM/rm_finetuned_PET_v3_seed43.jsonl" \
  --max_prompt_tokens 700 \
  --max_response_tokens 512

uv run modal run -d scripts/modal_train.py::build_reward_model_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path "/vol/runs/reward_model/rm_finetuned_PET_v3_seed44/checkpoints/step_000075/adapter" \
  --prefs_jsonl /root/project/public_eval/public_test_prefs_256.jsonl \
  --output_jsonl "/vol/judge/PET/RM/rm_finetuned_PET_v3_seed44.jsonl" \
  --max_prompt_tokens 700 \
  --max_response_tokens 512
```

```bash
mkdir -p policy_generations/PET
```

Download submission jsonls locally
```bash
uv run modal volume get llm-rl-final-project-volume judge/PET/GRPO policy_generations/PET
uv run modal volume get llm-rl-final-project-volume judge/PET/RM policy_generations/PET
```

Score GRPO Runs
```bash
python3 report_utils/judge_dir.py public_eval/public_test_gen_base_responses_128.jsonl policy_generations/PET/GRPO
```

### Ablations
#### Ablation 1, Rejection Sample Size 1
RM Fintuning, Seed 42
```bash
uv run modal run -d scripts/modal_train.py::reward_model_finetune_remote -- \
--initial_model_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445 \
--initial_model_config /vol/runs/wildchat_min4_judged_5k_reward_model_v1/config.json \
--base_policy_model_name Qwen/Qwen2.5-1.5B-Instruct \
--output_dir runs/reward_model/rm_finetuned_PET_v3_seed42_no_rejection \
--seed 42 \
--num_train_epochs 1.0 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--grad_accum_steps 16 \
--max_prompt_tokens 512 \
--max_response_tokens 256 \
--max_grad_norm 1.0 \
--eval_interval 25 \
--save_interval 50 \
--num_samples 1 \
--pessimistic_coef 10.0 \
--lr 3e-5 \
--wandb_project llm-rl-final-project \
--wandb_name reward_model_finetuned_PET_v3_seed42_no_rejection \
--wandb_enabled \
--s_min_new_tokens 8 \
--s_max_new_tokens 256 \
--s_temperature 0.9 \
--s_reward_batch_size 16 \
--s_top_p 0.95 \
--s_top_k 0 \
--s_repetition_penalty 1.0 \
--s_batch_size 4 \
--p_model_name Qwen/Qwen2.5-1.5B-Instruct \
--p_lora_r 32 \
--p_lora_alpha 64 \
--p_lora_dropout 0.05 \
--p_lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
--p_lora_bias none \
--p_grad_checkpointing
```
GRPO seed 42, w/ no rejection:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/rm_finetuned_PET_v3_seed42_no_rejection/checkpoints/step_000075/adapter \
--output_dir /vol/runs/PET/grpo_PET_no_rejection_v2_long_seed42 \
--seed 42 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_PET_no_rejection_v2_long_seed42
```

GRPO seed 43, w/ no rejection:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/rm_finetuned_PET_v3_seed42_no_rejection/checkpoints/step_000075/adapter \
--output_dir /vol/runs/PET/grpo_PET_no_rejection_v2_long_seed43 \
--seed 43 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_PET_no_rejection_v2_long_seed43
```

GRPO seed 44, w/ no rejection:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/rm_finetuned_PET_v3_seed42_no_rejection/checkpoints/step_000075/adapter \
--output_dir /vol/runs/PET/grpo_PET_no_rejection_v2_long_seed44 \
--seed 44 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_PET_no_rejection_v2_long_seed44
```

#### Ablation 2, Rejection Sample Size 2
RM Fintuning, Seed 42
```bash
uv run modal run -d scripts/modal_train.py::reward_model_finetune_remote -- \
--initial_model_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445 \
--initial_model_config /vol/runs/wildchat_min4_judged_5k_reward_model_v1/config.json \
--base_policy_model_name Qwen/Qwen2.5-1.5B-Instruct \
--output_dir runs/reward_model/rm_finetuned_PET_v3_seed42_2_rejection \
--seed 42 \
--num_train_epochs 1.0 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--grad_accum_steps 16 \
--max_prompt_tokens 512 \
--max_response_tokens 256 \
--max_grad_norm 1.0 \
--eval_interval 25 \
--save_interval 50 \
--num_samples 2 \
--pessimistic_coef 10.0 \
--lr 3e-5 \
--wandb_project llm-rl-final-project \
--wandb_name reward_model_finetuned_PET_v3_seed42_2_rejection \
--wandb_enabled \
--s_min_new_tokens 8 \
--s_max_new_tokens 256 \
--s_temperature 0.9 \
--s_reward_batch_size 16 \
--s_top_p 0.95 \
--s_top_k 0 \
--s_repetition_penalty 1.0 \
--s_batch_size 4 \
--p_model_name Qwen/Qwen2.5-1.5B-Instruct \
--p_lora_r 32 \
--p_lora_alpha 64 \
--p_lora_dropout 0.05 \
--p_lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
--p_lora_bias none \
--p_grad_checkpointing
```
GRPO seed 42, w/ 2 rejection:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/rm_finetuned_PET_v3_seed42_2_rejection/checkpoints/step_000075/adapter \
--output_dir /vol/runs/PET/grpo_PET_2_rejection_v2_long_seed42 \
--seed 42 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_PET_2_rejection_v2_long_seed42
```

GRPO seed 43, w/ 2 rejection:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/rm_finetuned_PET_v3_seed42_2_rejection/checkpoints/step_000075/adapter \
--output_dir /vol/runs/PET/grpo_PET_2_rejection_v2_long_seed43 \
--seed 43 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_PET_2_rejection_v2_long_seed43
```

GRPO seed 44, w/ 2 rejection:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/rm_finetuned_PET_v3_seed42_2_rejection/checkpoints/step_000075/adapter \
--output_dir /vol/runs/PET/grpo_PET_2_rejection_v2_long_seed44 \
--seed 44 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_PET_2_rejection_v2_long_seed44
```

#### Ablation 3, Train BT-RM for one more epoch.
RM Retraining, Same hyperparams as base RM, but one extra epoch
```bash
uv run modal run -d scripts/modal_train.py::reward_model_train_remote -- \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_prefs \
--eval_split test_prefs \
--output_dir /vol/runs/reward_model/wildchat_min4_judged_5k_reward_model_v1_LONG \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--grad_accum_steps 4 \
--lr 3e-5 \
--num_train_epochs 4 \
--max_prompt_tokens 700 \
--max_response_tokens 512 \
--eval_interval 25 \
--save_interval 50 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name wildchat_min4_judged_5k_reward_model_v1_LONG  \
```

GRPO seed 42, w/ long RM:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/wildchat_min4_judged_5k_reward_model_v1_LONG/checkpoints/<step>/adapter \
--output_dir /vol/runs/BASELINE/grpo_long_RM_v2_long_seed42 \
--seed 42 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_long_RM_v2_long_seed42
```

GRPO seed 43, w/ long RM:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/wildchat_min4_judged_5k_reward_model_v1_LONG/checkpoints/<step>/adapter \
--output_dir /vol/runs/BASELINE/grpo_long_RM_v2_long_seed43 \
--seed 43 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_long_RM_v2_long_seed43
```

GRPO seed 44, w/ long RM:
```bash
uv run modal run -d scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/reward_model/wildchat_min4_judged_5k_reward_model_v1_LONG/checkpoints/<step>/adapter \
--output_dir /vol/runs/BASELINE/grpo_long_RM_v2_long_seed44 \
--seed 44 \
--steps 100 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 256 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name grpo_long_RM_v2_long_seed44
```

We can use a similar process to our baseline runs to gererate and score completions against the judge for our ablations.