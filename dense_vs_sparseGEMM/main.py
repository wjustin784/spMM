#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from utils import run_spmm_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CuPy SpGEMM (CSR @ CSR) vs dense GEMM benchmark "
                    "with kernel-only and end-to-end modes."
    )

    parser.add_argument(
        "--size",
        type=int,
        nargs="+",
        required=True,
        help="Matrix dimension for A and B: A is (size x size), B is (size x size).",
    )
    parser.add_argument(
        "--density",
        type=float,
        nargs="+",
        required=True,
        help="Target density for A and B (SciPy CSR).",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Element dtype.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of hot-run repeats per configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Limit BLAS/OpenMP threads on CPU (0 = no limit).",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip an extra warmup run before timing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dtype_map = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
    }
    dtype = dtype_map[args.dtype]

    if args.threads > 0 and threadpool_limits is not None:
        tp_ctx = threadpool_limits(limits=args.threads)
    else:
        tp_ctx = None

    param_combinations = itertools.product(args.size, args.density)

    do_warmup = not args.no_warmup

    for (size, density) in param_combinations:
        run_params = {
            "m": size,
            "n": size,
            "p": size,
            "density": density,
            "dtype": dtype,
            "dtype_str": args.dtype,
            "runs": args.runs,
            "seed": args.seed,
            "do_warmup": do_warmup,
        }

        if tp_ctx is None:
            run_spmm_case(**run_params)
        else:
            with tp_ctx:
                run_spmm_case(**run_params)


if __name__ == "__main__":
    main()
