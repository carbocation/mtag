#!/usr/bin/env python3
"""Compare the optimized CPU kernels with the original MTAG equations."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import scipy.stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mtag


def make_inputs(num_snps, num_traits, seed):
    rng = np.random.default_rng(seed)
    z_scores = rng.normal(size=(num_snps, num_traits))
    sample_sizes = rng.integers(
        50_000, 250_000, size=(num_snps, num_traits)
    ).astype(float)
    raw_omega = rng.normal(size=(num_traits, num_traits))
    omega = raw_omega @ raw_omega.T
    omega *= 2.0e-5 / np.mean(np.diag(omega))
    omega += np.eye(num_traits) * 1.0e-5
    sigma = np.full((num_traits, num_traits), 0.1)
    np.fill_diagonal(sigma, 1.0)
    return z_scores, sample_sizes, omega, sigma


def legacy_mtag_analysis(z_scores, sample_sizes, omega, sigma):
    num_snps, num_traits = z_scores.shape
    w_n = np.einsum(
        "mp,pq->mpq", np.sqrt(sample_sizes), np.eye(num_traits)
    )
    w_n_inverse = np.linalg.inv(w_n)
    sigma_n = np.einsum(
        "mpq,mqr->mpr",
        np.einsum("mpq,qr->mpr", w_n_inverse, sigma),
        w_n_inverse,
    )
    results = [np.zeros_like(z_scores) for _ in range(3)]
    w_inverse_z = np.einsum("mqp,mp->mq", w_n_inverse, z_scores)

    for trait in range(num_traits):
        gamma = omega[:, trait]
        tau_squared = omega[trait, trait]
        inverse = np.linalg.inv(
            omega - np.outer(gamma, gamma) / tau_squared + sigma_n
        )
        yy = gamma / tau_squared
        weighted = np.einsum("q,mqp->mp", yy, inverse)
        denominator = np.einsum("mp,p->m", weighted, yy)
        results[0][:, trait] = (
            np.einsum("mp,mp->m", weighted, w_inverse_z) / denominator
        )
        results[1][:, trait] = np.sqrt(1.0 / denominator)
        results[2][:, trait] = np.einsum(
            "mp,m->m", weighted, 1.0 / denominator
        )
    return results


def legacy_mtag_variance(trait, omega, omega_state, sigma, sample_sizes):
    num_traits = sample_sizes.shape[1]
    w_n = np.einsum(
        "mp,pq->mpq", np.sqrt(sample_sizes), np.eye(num_traits)
    )
    w_n_inverse = np.linalg.inv(w_n)
    sigma_n = np.einsum(
        "mpq,mqr->mpr",
        np.einsum("mpq,qr->mpr", w_n_inverse, sigma),
        w_n_inverse,
    )
    gamma = omega[:, trait]
    tau_squared = omega[trait, trait]
    inverse = np.linalg.inv(
        omega - np.outer(gamma, gamma) / tau_squared + sigma_n
    )
    yy = gamma / tau_squared
    left = np.einsum("p,mpq->mq", yy, inverse)
    right = np.einsum("mpq,q->mp", inverse, yy)
    numerator = np.einsum(
        "mp,mp->m",
        left,
        np.einsum("mpq,mq->mp", omega_state + sigma_n, right),
    )
    denominator = np.einsum("p,mp->m", yy, right)
    return numerator / denominator


def legacy_scale_omega(omega, probability, causal_states):
    scaled = np.zeros_like(omega)
    for first_trait in range(omega.shape[0]):
        for second_trait in range(omega.shape[1]):
            joint_states = np.logical_and(
                causal_states[:, first_trait],
                causal_states[:, second_trait],
            )
            scaled[first_trait, second_trait] = (
                omega[first_trait, second_trait]
                / np.sum(probability[joint_states])
            )
    return scaled


def legacy_some_causal_for_all_traits(probability, causal_states):
    num_traits = causal_states.shape[1]
    for first_trait in range(num_traits):
        for second_trait in range(num_traits):
            joint_states = np.logical_and(
                causal_states[:, first_trait],
                causal_states[:, second_trait],
            )
            if np.sum(probability[joint_states]) == 0.0:
                return False
    return True


def legacy_compute_fdr(
    probability,
    trait,
    omega,
    sigma,
    causal_states,
    sample_sizes,
    sample_counts,
    p_threshold,
):
    z_threshold = scipy.stats.norm.isf(p_threshold / 2.0)
    scaled_omega = legacy_scale_omega(omega, probability, causal_states)
    omega_by_state = (
        np.einsum("st,sr->str", causal_states, causal_states) * scaled_omega
    )
    probability_significant = np.empty(len(causal_states))
    for state, omega_state in enumerate(omega_by_state):
        sd = np.sqrt(
            legacy_mtag_variance(
                trait, omega, omega_state, sigma, sample_sizes
            )
        )
        probability_significant[state] = np.sum(
            2.0
            * scipy.stats.norm.sf(z_threshold, loc=0, scale=sd)
            * sample_counts
        ) / float(np.sum(sample_counts))
    power_by_state = probability_significant * probability
    return (
        np.sum(power_by_state[~causal_states[:, trait]])
        / np.sum(power_by_state)
    )


def benchmark_mtag(args):
    inputs = make_inputs(args.snps, args.traits, args.seed)
    start = time.perf_counter()
    legacy = legacy_mtag_analysis(*inputs)
    legacy_seconds = time.perf_counter() - start

    start = time.perf_counter()
    optimized = mtag.mtag_analysis(*inputs, batch_size=args.batch_size)
    optimized_seconds = time.perf_counter() - start
    max_error = max(
        np.max(np.abs(expected - actual))
        for expected, actual in zip(legacy, optimized)
    )

    print("Main MTAG estimator")
    print("  SNPs / traits: {:,} / {}".format(args.snps, args.traits))
    print("  Legacy:        {:.4f} seconds".format(legacy_seconds))
    print("  Optimized:     {:.4f} seconds".format(optimized_seconds))
    print("  Speedup:       {:.2f}x".format(legacy_seconds / optimized_seconds))
    print("  Maximum error: {:.3e}".format(max_error))


def benchmark_fdr(args):
    _, sample_sizes, omega, sigma = make_inputs(
        args.fdr_sample_rows, args.fdr_traits, args.seed
    )
    causal_states = mtag.create_S(args.fdr_traits)
    candidate_grid = list(
        mtag.simplex_walk(
            len(causal_states) - 1, args.fdr_intervals + 1
        )
    )

    start = time.perf_counter()
    legacy_grid = np.asarray(
        [
            probability
            for probability in candidate_grid
            if legacy_some_causal_for_all_traits(
                probability, causal_states
            )
            and mtag.is_pos_semidef(
                legacy_scale_omega(omega, probability, causal_states)
            )
        ]
    )
    legacy_filter_seconds = time.perf_counter() - start

    start = time.perf_counter()
    optimized_grid = []
    for probability in candidate_grid:
        pair_probabilities = mtag._causal_pair_probabilities(
            probability, causal_states
        )
        if np.all(pair_probabilities > 0.0) and mtag.is_pos_semidef(
            omega / pair_probabilities
        ):
            optimized_grid.append(probability)
    probability_grid = np.asarray(optimized_grid)
    optimized_filter_seconds = time.perf_counter() - start
    np.testing.assert_array_equal(probability_grid, legacy_grid)
    sample_counts = np.ones(len(sample_sizes))

    start = time.perf_counter()
    legacy = np.asarray(
        [
            [
                legacy_compute_fdr(
                    probability,
                    trait,
                    omega,
                    sigma,
                    causal_states,
                    sample_sizes,
                    sample_counts,
                    args.p_threshold,
                )
                for trait in range(args.fdr_traits)
            ]
            for probability in probability_grid
        ]
    )
    legacy_seconds = time.perf_counter() - start

    start = time.perf_counter()
    prepared = mtag._prepare_fdr_calculation(
        omega, sigma, sample_sizes, sample_counts, args.p_threshold
    )
    optimized = np.asarray(
        [
            mtag._compute_fdr_values(
                probability, omega, causal_states, prepared
            )
            for probability in probability_grid
        ]
    )
    optimized_seconds = time.perf_counter() - start
    max_error = np.max(np.abs(legacy - optimized))

    print("maxFDR grid feasibility filter")
    print("  Candidate points: {:,}".format(len(candidate_grid)))
    print("  Legacy:           {:.4f} seconds".format(legacy_filter_seconds))
    print("  Optimized:        {:.4f} seconds".format(optimized_filter_seconds))
    print(
        "  Speedup:          {:.2f}x".format(
            legacy_filter_seconds / optimized_filter_seconds
        )
    )
    print("maxFDR numerical kernel")
    print(
        "  Traits / grid points: {} / {:,}".format(
            args.fdr_traits, len(probability_grid)
        )
    )
    print("  Legacy:        {:.4f} seconds".format(legacy_seconds))
    print("  Optimized:     {:.4f} seconds".format(optimized_seconds))
    print("  Speedup:       {:.2f}x".format(legacy_seconds / optimized_seconds))
    print("  Maximum error: {:.3e}".format(max_error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snps", type=int, default=100_000)
    parser.add_argument("--traits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--fdr-traits", type=int, default=3)
    parser.add_argument("--fdr-intervals", type=int, default=3)
    parser.add_argument("--fdr-sample-rows", type=int, default=3)
    parser.add_argument("--p-threshold", type=float, default=5.0e-8)
    parser.add_argument("--seed", type=int, default=8128)
    parser.add_argument("--skip-mtag", action="store_true")
    parser.add_argument("--skip-fdr", action="store_true")
    args = parser.parse_args()

    if not args.skip_mtag:
        benchmark_mtag(args)
    if not args.skip_fdr:
        if not args.skip_mtag:
            print()
        benchmark_fdr(args)


if __name__ == "__main__":
    main()
