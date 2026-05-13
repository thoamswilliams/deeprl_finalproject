"""
Run head to head LLM as a judge evaluations between model output JSONs
"""


from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common import (
    DEFAULT_JUDGE_MODEL,
    JudgeConfig,
    grade_policy_submission,
    load_jsonl,
    load_public_data,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Head to head eval between two model's refrence output jsons.",
    )
    ap.add_argument(
        "model_a",
        type=Path,
        help="Path to first model's jsonl outputs, treated as reference.",
    )
    ap.add_argument(
        "model_b",
        type=Path,
        help="Path to first model's jsonl outputs, treated as policy, winrate is reported for this model",
    )
    ap.add_argument(
        "--output_json",
        type=Path,
        default=Path("head_to_head_results.json"),
        help="Path to write json summary",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set to run the head to head eval.")

    if not args.model_a.is_file():
        raise FileNotFoundError(f"model_a file not found: {args.model_a}")
    if not args.model_b.is_file():
        raise FileNotFoundError(f"model_b file not found: {args.model_b}")
    
    thresholds = json.loads((Path(__file__).resolve().parent / "thresholds.json").read_text(encoding="utf-8"))

    judge_model = os.environ.get("LOCAL_AUTOGRADER_JUDGE_MODEL", thresholds.get("judge_model", DEFAULT_JUDGE_MODEL))

    judge_cfg = JudgeConfig(
        api_key=api_key,
        judge_model=judge_model,
        reasoning_effort=str(thresholds.get("reasoning_effort", "none")),
        max_workers=int(os.environ.get("LOCAL_AUTOGRADER_MAX_WORKERS", "16")),
    )
        
    public = load_public_data()
    prompts = public["part1_prompts"]

    model_a_gens = load_jsonl(args.model_a)
    model_b_gens = load_jsonl(args.model_b)

    # grade_policy_submission compares policy vs base
    # we treat model_a as base, and model_b as policy
    metrics = grade_policy_submission(prompts, model_a_gens, model_b_gens, judge_cfg)

    win_rate = metrics["policy_win_rate_pair_agree_usable"]
    usable = metrics["count_pair_agree_usable_rows"]

    summary = {
        "model_a": str(args.model_a),
        "model_b": str(args.model_b),
        "judge_model": judge_model,
        "model_b_win_rate_vs_model_a": win_rate,
        "usable_rows": usable,
        "metrics": metrics,
    }

    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {args.output_json}")
    print(f"model_a: {args.model_a}")
    print(f"model_b: {args.model_b}")
    print(f"model_b win rate vs model_a: {win_rate:.4f} (usable rows: {usable})")
    if metrics.get("error_examples"):
        first_err = str(metrics["error_examples"][0].get("error", ""))
        print(f"errors={len(metrics['error_examples'])} first_error={first_err[:180]}")


if __name__ == "__main__":
    main()