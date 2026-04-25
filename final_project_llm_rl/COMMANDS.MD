# COMMANDS
Log all commands used here and what they were used for.

## Finetuning PET RM
### Baseline Runs, To Establish Our Method Consistently Works
#### Default Settings + Seed: 42
```
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