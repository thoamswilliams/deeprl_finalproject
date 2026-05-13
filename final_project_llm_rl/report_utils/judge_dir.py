"""
Run head to head LLM as a judge evaluations for every jsonl in a directory, againts a specified base jsonl
"""


from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import concurrent.futures

from common import (
    DEFAULT_JUDGE_MODEL,
    JudgeConfig,
    grade_policy_submission,
    load_jsonl,
    load_public_data,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "public_eval"

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run head to head LLM as a judge for all files in input_path, against reference at ref_path",
    )
    ap.add_argument(
        "ref_path",
        type=Path,
        help="Path to reference model's jsonl outputs.",
    )
    ap.add_argument(
        "input_path",
        type=Path,
        help="Path to directory containing many jsonl outputs, win rate is reported for these model",
    )
    ap.add_argument(
        "--throttle_time",
        type=int,
        default=2,
        help="Time to wait between job submission to API",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set to run the head to head eval.")

    if not args.ref_path.is_file():
        raise FileNotFoundError(f"ref_path file not found: {args.ref_path}")
    if not args.input_path.is_dir():
        raise FileNotFoundError(f"input_path dir not found: {args.input_path}")
    
    output_json = args.input_path / "judge_results.json"

    judge_model = "gpt-5.4"

    judge_cfg = JudgeConfig(
        api_key=api_key,
        judge_model="gpt-5.4",
        reasoning_effort="none",
        max_workers=int(os.environ.get("LOCAL_AUTOGRADER_MAX_WORKERS", "16")),
    )
        
    prompts = load_jsonl(DATA_DIR / "public_test_gen_prompts_128.jsonl")

    model_gens = {}
    for mp in sorted(args.input_path.glob("*.jsonl")):
        model_gens[mp.stem] = load_jsonl(mp)

    if not model_gens:
        raise RuntimeError(f"No candidate jsonl files found in {args.input_path}")

    base_gens = load_jsonl(args.ref_path)

    def score_model(name_and_gen: tuple[str, list]) -> tuple[str, dict]:
        name, model_gen = name_and_gen
        metrics = grade_policy_submission(prompts, base_gens, model_gen, judge_cfg)
        return (name, metrics)

    # Add parallel scorring
    results: dict[str, dict] = {}
    print(f"Grading {len(model_gens)} models against ref_model={args.ref_path.stem} ...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        future_to_name = {}
        # throttle the submission of jobs
        for item in model_gens.items():
            future_to_name[pool.submit(score_model, item)] = item[0]
            time.sleep(args.throttle_time)

        for fut in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[fut]
            _, metrics = fut.result()
            results[name] = metrics
            win_rate = metrics.get("policy_win_rate_pair_agree_usable")
            usable = metrics.get("count_pair_agree_usable_rows")
            wr_str = f"{win_rate:.4f}"
            print(f"  done: {name}  win_rate={wr_str}  usable={usable}")

    summary = {
    "ref_model": args.ref_path.stem,
    "ref_path": str(args.ref_path),
    "input_path": str(args.input_path),
    "judge_model": judge_model,
    "models": {
        name: {
            "win_rate_vs_ref": metrics.get("policy_win_rate_pair_agree_usable"),
            "usable_rows": metrics.get("count_pair_agree_usable_rows"),
            "metrics": metrics
            } 
            for name, metrics in results.items()
        }
    }

    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
 
    print(f"Wrote {output_json}")
    print(f"ref_model: {args.ref_path.stem}")
    print("-" * 72)
 
    for name, entry in summary["models"].items():
        win_rate = entry.get("win_rate_vs_ref")
        usable = entry.get("usable_rows")
        wr_str = f"{win_rate:.4f}"
        print(f"{name}")
        print(f"  win rate vs ref_model: {wr_str} (usable rows: {usable})")
        metrics = entry.get("metrics", {})
        if metrics.get("error_examples"):
            first_err = str(metrics["error_examples"][0].get("error", ""))
            print(
                f"  errors={len(metrics['error_examples'])} "
                f"first_error={first_err[:180]}"
            )


if __name__ == "__main__":
    main()