"""Optional Numba kernels for fused automatic maxFDR grid evaluation."""

import math

import numpy as np
from numba import njit, prange


def automatic_grid_size(num_states, intervals):
    """Return the number of lattice points in the automatic simplex grid."""
    return math.comb(intervals + num_states - 1, num_states - 1)


def binomial_table(max_n, max_k):
    """Build the integer binomial table used for lexicographic unranking."""
    table = np.zeros((max_n + 1, max_k + 1), dtype=np.int64)
    table[:, 0] = 1
    for n in range(1, max_n + 1):
        upper = min(n, max_k)
        for k in range(1, upper + 1):
            value = int(table[n - 1, k - 1]) + int(table[n - 1, k])
            if value > np.iinfo(np.int64).max:
                raise OverflowError("automatic maxFDR grid exceeds int64")
            table[n, k] = value
    return table


def prepare_fdr_arrays(prepared):
    """Convert the Python maxFDR preparation dictionary to dense arrays."""
    trait_terms = prepared["trait_terms"]
    return (
        np.asarray([terms[0] for terms in trait_terms], dtype=np.float64),
        np.asarray([terms[1] for terms in trait_terms], dtype=np.float64),
        np.asarray([terms[2] for terms in trait_terms], dtype=np.float64),
        np.asarray([terms[3] for terms in trait_terms], dtype=np.float64),
        np.asarray(prepared["n_counts"], dtype=np.float64),
        float(prepared["n_total"]),
        float(prepared["z_threshold"]),
    )


@njit(inline="always")
def _unrank_composition(rank, intervals, num_states, combinations, counts):
    """Decode a lexicographic combinations rank into a weak composition."""
    num_bars = num_states - 1
    num_positions = intervals + num_states - 1
    previous = -1
    remaining_rank = rank

    for bar_index in range(num_bars):
        bars_remaining = num_bars - bar_index - 1
        candidate = previous + 1
        while candidate < num_positions:
            positions_remaining = num_positions - candidate - 1
            if bars_remaining == 0:
                suffix_count = 1
            elif positions_remaining < bars_remaining:
                suffix_count = 0
            else:
                suffix_count = combinations[
                    positions_remaining, bars_remaining
                ]
            if remaining_rank < suffix_count:
                break
            remaining_rank -= suffix_count
            candidate += 1
        counts[bar_index] = candidate - previous - 1
        previous = candidate

    counts[num_states - 1] = num_positions - previous - 1


@njit(parallel=True, cache=True)
def _evaluate_automatic_grid_chunk(
    start_rank,
    stop_rank,
    intervals,
    combinations,
    causal_states,
    omega,
    num_left,
    num_right,
    denominator,
    sigma_numerator,
    n_counts,
    n_total,
    z_threshold,
    fit_ss,
    pi_causal_ss,
):
    """Generate, filter, and evaluate one rank range entirely in native code."""
    chunk_size = stop_rank - start_rank
    num_states, num_traits = causal_states.shape
    num_n_values = n_counts.shape[0]
    probability_grid = np.zeros((chunk_size, num_states), dtype=np.float64)
    fdr_matrix = np.zeros((chunk_size, num_traits), dtype=np.float64)
    feasible = np.zeros(chunk_size, dtype=np.uint8)
    sqrt_two = math.sqrt(2.0)
    eps = np.finfo(np.float64).eps

    for row in prange(chunk_size):
        counts = np.empty(num_states, dtype=np.int64)
        _unrank_composition(
            start_rank + row,
            intervals,
            num_states,
            combinations,
            counts,
        )

        pair_counts = np.zeros((num_traits, num_traits), dtype=np.int64)
        valid = True
        for first_trait in range(num_traits):
            for second_trait in range(num_traits):
                joint_count = 0
                for state in range(num_states):
                    if (
                        causal_states[state, first_trait]
                        and causal_states[state, second_trait]
                    ):
                        joint_count += counts[state]
                pair_counts[first_trait, second_trait] = joint_count
                if joint_count == 0:
                    valid = False
        if not valid:
            continue

        if fit_ss:
            for trait in range(num_traits):
                causal_probability = pair_counts[trait, trait] / intervals
                if (
                    abs(causal_probability - pi_causal_ss[trait])
                    >= 1.0 / intervals
                ):
                    valid = False
                    break
        if not valid:
            continue

        scaled_omega = np.empty((num_traits, num_traits), dtype=np.float64)
        for first_trait in range(num_traits):
            for second_trait in range(num_traits):
                scaled_omega[first_trait, second_trait] = (
                    omega[first_trait, second_trait]
                    * intervals
                    / pair_counts[first_trait, second_trait]
                )

        eigenvalues = np.linalg.eigvalsh(scaled_omega)
        max_abs_eigenvalue = 1.0
        for eigenvalue in eigenvalues:
            max_abs_eigenvalue = max(max_abs_eigenvalue, abs(eigenvalue))
        tolerance = eps * max_abs_eigenvalue * num_traits
        if eigenvalues[0] < -tolerance:
            continue

        feasible[row] = 1
        for state in range(num_states):
            probability_grid[row, state] = counts[state] / intervals

        for trait in range(num_traits):
            total_power = 0.0
            false_discovery_power = 0.0
            for state in range(num_states):
                probability_significant = 0.0
                for n_index in range(num_n_values):
                    omega_numerator = 0.0
                    for first_trait in range(num_traits):
                        if not causal_states[state, first_trait]:
                            continue
                        for second_trait in range(num_traits):
                            if causal_states[state, second_trait]:
                                omega_numerator += (
                                    num_left[trait, n_index, first_trait]
                                    * scaled_omega[first_trait, second_trait]
                                    * num_right[
                                        trait, n_index, second_trait
                                    ]
                                )
                    variance = (
                        omega_numerator
                        + sigma_numerator[trait, n_index]
                    ) / denominator[trait, n_index]
                    sd = math.sqrt(variance)
                    probability_significant += (
                        math.erfc(z_threshold / (sqrt_two * sd))
                        * n_counts[n_index]
                    )
                probability_significant /= n_total
                state_power = (
                    probability_significant * counts[state] / intervals
                )
                total_power += state_power
                if not causal_states[state, trait]:
                    false_discovery_power += state_power
            fdr_matrix[row, trait] = false_discovery_power / total_power

    return probability_grid, fdr_matrix, feasible


def evaluate_automatic_grid_chunk(
    start_rank,
    stop_rank,
    intervals,
    causal_states,
    omega,
    prepared,
    pi_causal_ss=None,
):
    """Evaluate a chunk and return only feasible probability/FDR rows."""
    num_states = len(causal_states)
    total_points = automatic_grid_size(num_states, intervals)
    if start_rank < 0 or stop_rank < start_rank or stop_rank > total_points:
        raise ValueError("invalid automatic maxFDR grid rank range")
    combinations = binomial_table(
        intervals + num_states - 1, num_states - 1
    )
    prepared_arrays = prepare_fdr_arrays(prepared)
    if pi_causal_ss is None:
        fit_ss = False
        pi_causal_ss = np.zeros(causal_states.shape[1], dtype=np.float64)
    else:
        fit_ss = True
        pi_causal_ss = np.asarray(pi_causal_ss, dtype=np.float64)

    probability_grid, fdr_matrix, feasible = _evaluate_automatic_grid_chunk(
        int(start_rank),
        int(stop_rank),
        int(intervals),
        combinations,
        np.asarray(causal_states, dtype=np.bool_),
        np.asarray(omega, dtype=np.float64),
        *prepared_arrays,
        fit_ss,
        pi_causal_ss,
    )
    selected = feasible.astype(bool)
    return probability_grid[selected], fdr_matrix[selected]


def evaluate_automatic_grid(
    intervals,
    causal_states,
    omega,
    prepared,
    pi_causal_ss=None,
    chunk_size=100_000,
    progress_callback=None,
):
    """Evaluate the complete automatic grid in bounded native chunks."""
    if chunk_size <= 0:
        raise ValueError("Numba maxFDR chunk size must be positive")
    total_points = automatic_grid_size(len(causal_states), intervals)
    probability_chunks = []
    fdr_chunks = []
    for start_rank in range(0, total_points, chunk_size):
        stop_rank = min(start_rank + chunk_size, total_points)
        probability_chunk, fdr_chunk = evaluate_automatic_grid_chunk(
            start_rank,
            stop_rank,
            intervals,
            causal_states,
            omega,
            prepared,
            pi_causal_ss=pi_causal_ss,
        )
        if len(probability_chunk):
            probability_chunks.append(probability_chunk)
            fdr_chunks.append(fdr_chunk)
        if progress_callback is not None:
            progress_callback(stop_rank, total_points)

    if not probability_chunks:
        return (
            np.empty((0, len(causal_states)), dtype=float),
            np.empty((0, causal_states.shape[1]), dtype=float),
        )
    return np.concatenate(probability_chunks), np.concatenate(fdr_chunks)
