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
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--sample-chunks",
        type=int,
        default=0,
        help=(
            "Sample this many evenly spaced chunks instead of enumerating "
            "the complete Python reference grid"
        ),
    )
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

    if args.sample_chunks:
        if args.sample_chunks <= 0:
            raise ValueError("--sample-chunks must be positive")
        prepared_arrays = mtag_numba.prepare_fdr_arrays(prepared)
        state_arrays = mtag_numba.prepare_causal_state_arrays(states)
        combinations = mtag_numba.binomial_table(
            args.intervals + len(states) - 1,
            max(len(states) - 1, args.intervals),
        )
        pi_causal_ss = np.zeros(args.traits)
        starts = np.linspace(
            0,
            total_points - min(args.chunk_size, total_points),
            args.sample_chunks,
            dtype=np.int64,
        )
        warmup_stop = min(total_points, 2)
        mtag_numba._evaluate_automatic_grid_max_chunk_sparse(
            0,
            warmup_stop,
            total_points,
            args.intervals,
            combinations,
            states,
            *state_arrays,
            omega,
            *prepared_arrays,
            False,
            pi_causal_ss,
        )

        sampled_candidates = 0
        feasible_count = 0
        start_time = time.perf_counter()
        for start_rank in starts:
            stop_rank = min(
                int(start_rank) + args.chunk_size, total_points
            )
            _, _, block_counts, invalid = (
                mtag_numba._evaluate_automatic_grid_max_chunk_sparse(
                    int(start_rank),
                    stop_rank,
                    total_points,
                    args.intervals,
                    combinations,
                    states,
                    *state_arrays,
                    omega,
                    *prepared_arrays,
                    False,
                    pi_causal_ss,
                )
            )
            assert not np.any(invalid)
            sampled_candidates += stop_rank - int(start_rank)
            feasible_count += int(np.sum(block_counts))
        sample_seconds = time.perf_counter() - start_time
        candidate_rate = sampled_candidates / sample_seconds

        print("Traits / intervals: {} / {}".format(
            args.traits, args.intervals
        ))
        print("Candidate points:    {:,}".format(total_points))
        print("Sampled points:      {:,}".format(sampled_candidates))
        print("Sample feasible:     {:,}".format(feasible_count))
        print("Numba threads:       {}".format(args.threads))
        print("Candidate rate:      {:,.0f} / second".format(
            candidate_rate
        ))
        projected_seconds = total_points / candidate_rate
        if projected_seconds < 3600.0:
            print("Projected runtime:   {:.2f} minutes".format(
                projected_seconds / 60.0
            ))
        else:
            print("Projected runtime:   {:.2f} hours".format(
                projected_seconds / 3600.0
            ))
        return

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
