#!/usr/bin/env python3
"""Compare the fused Numba automatic-grid engine with Python maxFDR."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import numba


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mtag
import mtag_numba


def make_inputs(num_traits, sample_rows, seed):
    rng = np.random.default_rng(seed)
    sample_sizes = rng.integers(
        50_000, 250_000, size=(sample_rows, num_traits)
    ).astype(float)
    raw_omega = rng.normal(size=(num_traits, num_traits))
    omega = raw_omega @ raw_omega.T
    omega *= 2.0e-5 / np.mean(np.diag(omega))
    omega += np.eye(num_traits) * 1.0e-5
    sigma = np.full((num_traits, num_traits), 0.1)
    np.fill_diagonal(sigma, 1.0)
    return sample_sizes, omega, sigma


def python_grid(intervals, states, omega, prepared):
    probabilities = []
    fdr_values = []
    for probability in mtag.simplex_walk(len(states) - 1, intervals + 1):
        pair_probabilities = mtag._causal_pair_probabilities(
            probability, states
        )
        if not np.all(pair_probabilities > 0.0):
            continue
        if not mtag.is_pos_semidef(omega / pair_probabilities):
            continue
        probabilities.append(probability)
        fdr_values.append(
            mtag._compute_fdr_values(probability, omega, states, prepared)
        )
    return np.asarray(probabilities), np.asarray(fdr_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traits", type=int, default=5)
    parser.add_argument("--intervals", type=int, default=2)
    parser.add_argument("--sample-rows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=8128)
    parser.add_argument("--p-threshold", type=float, default=5.0e-8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    numba.set_num_threads(args.threads)

    sample_sizes, omega, sigma = make_inputs(
        args.traits, args.sample_rows, args.seed
    )
    states = mtag.create_S(args.traits)
    sample_counts = np.ones(args.sample_rows)
    prepared = mtag._prepare_fdr_calculation(
        omega, sigma, sample_sizes, sample_counts, args.p_threshold
    )
    total_points = mtag_numba.automatic_grid_size(
        len(states), args.intervals
    )

    # Compile outside the timed region.
    mtag_numba.evaluate_automatic_grid_chunk(
        0, min(total_points, 2), args.intervals, states, omega, prepared
    )

    start = time.perf_counter()
    expected_grid, expected_fdr = python_grid(
        args.intervals, states, omega, prepared
    )
    python_seconds = time.perf_counter() - start

    start = time.perf_counter()
    actual_max, actual_probabilities, feasible_count = (
        mtag_numba.evaluate_automatic_grid_max(
            args.intervals,
            states,
            omega,
            prepared,
            chunk_size=args.chunk_size,
        )
    )
    numba_seconds = time.perf_counter() - start

    expected_indices = np.argmax(expected_fdr, axis=0)
    expected_max = expected_fdr[
        expected_indices, np.arange(args.traits)
    ]
    expected_probabilities = expected_grid[expected_indices]
    assert feasible_count == len(expected_grid)
    np.testing.assert_array_equal(
        actual_probabilities, expected_probabilities
    )
    np.testing.assert_allclose(
        actual_max, expected_max, rtol=1.0e-10, atol=1.0e-15
    )
    print("Traits / intervals: {} / {}".format(args.traits, args.intervals))
    print("Candidate points:    {:,}".format(total_points))
    print("Feasible points:     {:,}".format(feasible_count))
    print("Numba threads:       {}".format(args.threads))
    print("Python:              {:.6f} seconds".format(python_seconds))
    print("Numba:               {:.6f} seconds".format(numba_seconds))
    print("Speedup:             {:.2f}x".format(python_seconds / numba_seconds))
    print(
        "Maximum FDR error:   {:.3e}".format(
            np.max(np.abs(actual_max - expected_max))
        )
    )


if __name__ == "__main__":
    main()
