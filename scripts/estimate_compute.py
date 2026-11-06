#!/usr/bin/env python3
"""Estimate workload from explicit inputs, not GPU performance claims."""

import argparse
import json
import math


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=positive_int, default=10000)
    parser.add_argument("--decisions-per-game", type=positive_float, default=600,
                        help="All LM calls across both seats, including opponent calls")
    parser.add_argument("--input-tokens", type=positive_int, default=2048)
    parser.add_argument("--output-tokens", type=positive_int, default=32)
    parser.add_argument("--decisions-per-second", type=positive_float, required=True,
                        help="Aggregate completed decisions/s including prefill and environment")
    parser.add_argument("--gpus", type=positive_int, default=8,
                        help="Total GPUs allocated during the entire estimated run")
    parser.add_argument("--overhead", type=positive_float, default=2,
                        help="Total wall time / rollout-only time; includes learning/evaluation")
    parser.add_argument("--usd-per-gpu-hour", type=positive_float, default=None,
                        help="Optional actual quoted rate; no default price")
    args = parser.parse_args()
    if args.overhead < 1:
        parser.error("--overhead must be at least 1")
    decisions = args.games * args.decisions_per_game
    rollout_hours = decisions / args.decisions_per_second / 3600
    total_hours = rollout_hours * args.overhead
    gpu_hours = total_hours * args.gpus
    result = {
        "status": "estimate_from_user_assumptions_not_benchmark",
        "assumptions": vars(args),
        "lm_decisions": decisions,
        "input_tokens_before_prefix_cache": decisions * args.input_tokens,
        "generated_tokens": decisions * args.output_tokens,
        "rollout_hours": rollout_hours,
        "total_hours": total_hours,
        "allocated_gpu_hours": gpu_hours,
        "excluded": "SFT, corpus search, additional seeds, and evaluation beyond the overhead assumption"
    }
    if args.usd_per_gpu_hour is not None:
        result["estimated_usd"] = gpu_hours * args.usd_per_gpu_hour
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
