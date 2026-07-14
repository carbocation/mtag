"""Optional Numba kernels for fused automatic maxFDR grid evaluation."""

import itertools
import math

import numpy as np
from numba import njit, prange


class BranchSearchLimitExceeded(RuntimeError):
    """Raised when exact branch traversal would exceed its memory guard."""


def automatic_grid_size(num_states, intervals):
    """Return the number of lattice points in the automatic simplex grid."""
    total_points = math.comb(
        intervals + num_states - 1, num_states - 1
    )
    if total_points > np.iinfo(np.int64).max:
        raise OverflowError("automatic maxFDR grid exceeds int64 ranks")
    return total_points


def binomial_table(max_n, max_k):
    """Build the integer binomial table used for lexicographic unranking."""
    table = np.zeros((max_n + 1, max_k + 1), dtype=np.int64)
    table[:, 0] = 1
    for n in range(1, max_n + 1):
        upper = min(n, max_k)
        for k in range(1, upper + 1):
            value = int(table[n - 1, k - 1]) + int(table[n - 1, k])
            table[n, k] = min(value, np.iinfo(np.int64).max)
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


def prepare_causal_state_arrays(causal_states):
    """Precompute compact trait and pair lists for every causal state."""
    causal_states = np.asarray(causal_states, dtype=np.bool_)
    num_states, num_traits = causal_states.shape
    num_pairs = num_traits * (num_traits + 1) // 2
    pair_index = np.empty((num_traits, num_traits), dtype=np.int64)
    pair_number = 0
    for first_trait in range(num_traits):
        for second_trait in range(first_trait, num_traits):
            pair_index[first_trait, second_trait] = pair_number
            pair_index[second_trait, first_trait] = pair_number
            pair_number += 1

    state_trait_ids = np.zeros(
        (num_states, num_traits), dtype=np.int64
    )
    state_trait_counts = np.zeros(num_states, dtype=np.int64)
    state_pair_indices = np.zeros(
        (num_states, num_pairs), dtype=np.int64
    )
    state_pair_counts = np.zeros(num_states, dtype=np.int64)
    for state in range(num_states):
        trait_ids = np.flatnonzero(causal_states[state])
        state_trait_counts[state] = len(trait_ids)
        state_trait_ids[state, : len(trait_ids)] = trait_ids
        pair_count = 0
        for first_offset, first_trait in enumerate(trait_ids):
            for second_trait in trait_ids[first_offset:]:
                state_pair_indices[state, pair_count] = pair_index[
                    first_trait, second_trait
                ]
                pair_count += 1
        state_pair_counts[state] = pair_count

    return (
        state_trait_ids,
        state_trait_counts,
        state_pair_indices,
        state_pair_counts,
        pair_index,
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


@njit(inline="always")
def _unrank_sparse_composition(
    rank,
    total_points,
    intervals,
    num_states,
    combinations,
    state_ids,
    multiplicities,
):
    """Decode a grid rank into only its occupied causal states."""
    num_positions = intervals + num_states - 1
    remaining_rank = total_points - rank - 1
    previous = -1
    num_occupied = 0

    # A composition's stars are the complement of its bars. The bar
    # combinations used by simplex_walk are in the reverse lexicographic
    # order of the complementary star combinations.
    for star_index in range(intervals):
        stars_remaining = intervals - star_index - 1
        candidate = previous + 1
        while candidate < num_positions:
            positions_remaining = num_positions - candidate - 1
            if stars_remaining == 0:
                suffix_count = 1
            elif positions_remaining < stars_remaining:
                suffix_count = 0
            else:
                suffix_count = combinations[
                    positions_remaining, stars_remaining
                ]
            if remaining_rank < suffix_count:
                break
            remaining_rank -= suffix_count
            candidate += 1

        state = candidate - star_index
        if num_occupied and state_ids[num_occupied - 1] == state:
            multiplicities[num_occupied - 1] += 1
        else:
            state_ids[num_occupied] = state
            multiplicities[num_occupied] = 1
            num_occupied += 1
        previous = candidate

    return num_occupied


@njit(inline="always")
def _unrank_star_positions(
    rank,
    total_points,
    intervals,
    num_states,
    combinations,
    star_positions,
):
    """Decode a grid rank to its complementary star combination."""
    num_positions = intervals + num_states - 1
    remaining_rank = total_points - rank - 1
    previous = -1
    for star_index in range(intervals):
        stars_remaining = intervals - star_index - 1
        candidate = previous + 1
        while candidate < num_positions:
            positions_remaining = num_positions - candidate - 1
            if stars_remaining == 0:
                suffix_count = 1
            elif positions_remaining < stars_remaining:
                suffix_count = 0
            else:
                suffix_count = combinations[
                    positions_remaining, stars_remaining
                ]
            if remaining_rank < suffix_count:
                break
            remaining_rank -= suffix_count
            candidate += 1
        star_positions[star_index] = candidate
        previous = candidate


@njit(inline="always")
def _previous_star_combination(star_positions, num_positions):
    """Move one step backward in lexicographic combination order."""
    num_stars = len(star_positions)
    for star_index in range(num_stars - 1, -1, -1):
        minimum = 0
        if star_index:
            minimum = star_positions[star_index - 1] + 1
        if star_positions[star_index] > minimum:
            star_positions[star_index] -= 1
            for suffix_index in range(star_index + 1, num_stars):
                star_positions[suffix_index] = (
                    num_positions - num_stars + suffix_index
                )
            return


@njit(inline="always")
def _fast_psd_status(matrix):
    """Return -1 for non-PSD, 1 for clearly PD, or 0 for eig fallback."""
    size = matrix.shape[0]
    eps = np.finfo(np.float64).eps
    spectral_upper_bound = 0.0
    for row in range(size):
        row_sum = 0.0
        for column in range(size):
            row_sum += abs(matrix[row, column])
        spectral_upper_bound = max(spectral_upper_bound, row_sum)

    scale = max(1.0, spectral_upper_bound)
    tolerance_upper_bound = eps * scale * size

    # By eigenvalue interlacing, a sufficiently negative eigenvalue of any
    # 1x1 or 2x2 principal submatrix proves that the full matrix fails the
    # reference eigvalsh tolerance. The row-sum spectral bound makes this a
    # conservative rejection screen.
    for first in range(size):
        if matrix[first, first] < -tolerance_upper_bound:
            return -1
        for second in range(first + 1, size):
            half_trace = 0.5 * (
                matrix[first, first] + matrix[second, second]
            )
            half_difference = 0.5 * (
                matrix[first, first] - matrix[second, second]
            )
            radius = math.hypot(
                half_difference, matrix[first, second]
            )
            if half_trace - radius < -tolerance_upper_bound:
                return -1

    # A Cholesky factorization whose pivots are comfortably separated from
    # zero establishes positive definiteness. Ambiguous and near-singular
    # cases deliberately fall back to the reference eigensolver.
    pivot_margin = math.sqrt(eps) * scale * size
    cholesky = np.zeros((size, size), dtype=np.float64)
    for row in range(size):
        for column in range(row + 1):
            value = matrix[row, column]
            for inner in range(column):
                value -= (
                    cholesky[row, inner] * cholesky[column, inner]
                )
            if row == column:
                if value <= pivot_margin:
                    return 0
                cholesky[row, column] = math.sqrt(value)
            else:
                cholesky[row, column] = (
                    value / cholesky[column, column]
                )
    return 1


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


@njit(inline="always")
def _evaluate_occupied_candidate(
    state_ids,
    multiplicities,
    num_occupied,
    intervals,
    causal_states,
    state_trait_ids,
    state_trait_counts,
    state_pair_indices,
    state_pair_counts,
    pair_index,
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
    candidate_fdr,
):
    """Evaluate one occupied-state representation and store its FDR."""
    _, num_traits = causal_states.shape
    num_pairs = num_traits * (num_traits + 1) // 2
    num_n_values = n_counts.shape[0]
    sqrt_two = math.sqrt(2.0)
    eps = np.finfo(np.float64).eps

    pair_counts = np.zeros(num_pairs, dtype=np.int64)
    for occupied_index in range(num_occupied):
        state = state_ids[occupied_index]
        count = multiplicities[occupied_index]
        for state_pair_offset in range(state_pair_counts[state]):
            pair_counts[
                state_pair_indices[state, state_pair_offset]
            ] += count

    for pair_number in range(num_pairs):
        if pair_counts[pair_number] == 0:
            return False

    if fit_ss:
        for trait in range(num_traits):
            causal_probability = (
                pair_counts[pair_index[trait, trait]] / intervals
            )
            if (
                abs(causal_probability - pi_causal_ss[trait])
                >= 1.0 / intervals
            ):
                return False

    scaled_omega = np.empty(
        (num_traits, num_traits), dtype=np.float64
    )
    for first_trait in range(num_traits):
        for second_trait in range(num_traits):
            scaled_omega[first_trait, second_trait] = (
                omega[first_trait, second_trait]
                * intervals
                / pair_counts[pair_index[first_trait, second_trait]]
            )

    psd_status = _fast_psd_status(scaled_omega)
    if psd_status < 0:
        return False
    if psd_status == 0:
        eigenvalues = np.linalg.eigvalsh(scaled_omega)
        max_abs_eigenvalue = 1.0
        for eigenvalue in eigenvalues:
            max_abs_eigenvalue = max(
                max_abs_eigenvalue, abs(eigenvalue)
            )
        tolerance = eps * max_abs_eigenvalue * num_traits
        if eigenvalues[0] < -tolerance:
            return False

    for trait in range(num_traits):
        total_power = 0.0
        false_discovery_power = 0.0
        for occupied_index in range(num_occupied):
            state = state_ids[occupied_index]
            probability_significant = 0.0
            for n_index in range(num_n_values):
                omega_numerator = 0.0
                state_trait_count = state_trait_counts[state]
                for first_offset in range(state_trait_count):
                    first_trait = state_trait_ids[state, first_offset]
                    for second_offset in range(state_trait_count):
                        second_trait = state_trait_ids[
                            state, second_offset
                        ]
                        omega_numerator += (
                            num_left[trait, n_index, first_trait]
                            * scaled_omega[first_trait, second_trait]
                            * num_right[trait, n_index, second_trait]
                        )
                variance = (
                    omega_numerator + sigma_numerator[trait, n_index]
                ) / denominator[trait, n_index]
                sd = math.sqrt(variance)
                probability_significant += (
                    math.erfc(z_threshold / (sqrt_two * sd))
                    * n_counts[n_index]
                )
            probability_significant /= n_total
            state_power = (
                probability_significant
                * multiplicities[occupied_index]
                / intervals
            )
            total_power += state_power
            if not causal_states[state, trait]:
                false_discovery_power += state_power
        candidate_fdr[trait] = false_discovery_power / total_power
    return True


@njit(inline="always")
def _evaluate_sparse_candidate(
    star_positions,
    intervals,
    causal_states,
    state_trait_ids,
    state_trait_counts,
    state_pair_indices,
    state_pair_counts,
    pair_index,
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
    candidate_fdr,
):
    """Evaluate one sparse composition and store its per-trait FDR."""
    state_ids = np.empty(intervals, dtype=np.int64)
    multiplicities = np.empty(intervals, dtype=np.int64)
    num_occupied = 0
    for star_index in range(intervals):
        state = star_positions[star_index] - star_index
        if num_occupied and state_ids[num_occupied - 1] == state:
            multiplicities[num_occupied - 1] += 1
        else:
            state_ids[num_occupied] = state
            multiplicities[num_occupied] = 1
            num_occupied += 1

    return _evaluate_occupied_candidate(
        state_ids,
        multiplicities,
        num_occupied,
        intervals,
        causal_states,
        state_trait_ids,
        state_trait_counts,
        state_pair_indices,
        state_pair_counts,
        pair_index,
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
        candidate_fdr,
    )


@njit(inline="always")
def _evaluate_counts_candidate(
    counts,
    intervals,
    causal_states,
    state_trait_ids,
    state_trait_counts,
    state_pair_indices,
    state_pair_counts,
    pair_index,
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
    candidate_fdr,
):
    """Evaluate one dense composition-count row."""
    num_states = len(counts)
    state_ids = np.empty(num_states, dtype=np.int64)
    multiplicities = np.empty(num_states, dtype=np.int64)
    num_occupied = 0
    for state in range(num_states):
        if counts[state] > 0:
            state_ids[num_occupied] = state
            multiplicities[num_occupied] = counts[state]
            num_occupied += 1

    return _evaluate_occupied_candidate(
        state_ids,
        multiplicities,
        num_occupied,
        intervals,
        causal_states,
        state_trait_ids,
        state_trait_counts,
        state_pair_indices,
        state_pair_counts,
        pair_index,
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
        candidate_fdr,
    )


@njit(parallel=True, cache=True)
def _evaluate_automatic_grid_max_chunk_sparse(
    start_rank,
    stop_rank,
    total_points,
    intervals,
    combinations,
    causal_states,
    state_trait_ids,
    state_trait_counts,
    state_pair_indices,
    state_pair_counts,
    pair_index,
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
    """Evaluate a max-only rank range using occupied states only."""
    chunk_size = stop_rank - start_rank
    num_traits = causal_states.shape[1]
    iteration_block_size = 256
    num_blocks = (
        chunk_size + iteration_block_size - 1
    ) // iteration_block_size
    num_positions = intervals + causal_states.shape[0] - 1
    block_max = np.full(
        (num_blocks, num_traits), -np.inf, dtype=np.float64
    )
    block_ranks = np.full(
        (num_blocks, num_traits), -1, dtype=np.int64
    )
    block_feasible_counts = np.zeros(num_blocks, dtype=np.int64)
    block_invalid = np.zeros(num_blocks, dtype=np.uint8)

    for block_index in prange(num_blocks):
        first_row = block_index * iteration_block_size
        stop_row = min(first_row + iteration_block_size, chunk_size)
        star_positions = np.empty(intervals, dtype=np.int64)
        _unrank_star_positions(
            start_rank + first_row,
            total_points,
            intervals,
            causal_states.shape[0],
            combinations,
            star_positions,
        )
        candidate_fdr = np.empty(num_traits, dtype=np.float64)
        feasible_count = 0
        for row in range(first_row, stop_row):
            if row != first_row:
                _previous_star_combination(
                    star_positions, num_positions
                )
            if _evaluate_sparse_candidate(
                star_positions,
                intervals,
                causal_states,
                state_trait_ids,
                state_trait_counts,
                state_pair_indices,
                state_pair_counts,
                pair_index,
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
                candidate_fdr,
            ):
                feasible_count += 1
                for trait in range(num_traits):
                    value = candidate_fdr[trait]
                    if (
                        not math.isfinite(value)
                        or value < -1.0e-12
                        or value > 1.0 + 1.0e-12
                    ):
                        block_invalid[block_index] = 1
                    value = min(1.0, max(0.0, value))
                    if value > block_max[block_index, trait]:
                        block_max[block_index, trait] = value
                        block_ranks[block_index, trait] = (
                            start_rank + row
                        )
        block_feasible_counts[block_index] = feasible_count

    return (
        block_max,
        block_ranks,
        block_feasible_counts,
        block_invalid,
    )


@njit(inline="always")
def _branch_counts_survive(
    counts,
    intervals,
    causal_states,
    state_pair_indices,
    state_pair_counts,
    pair_index,
    omega,
    prune_tolerance,
    fit_ss,
    pi_causal_ss,
):
    """Apply safe principal-matrix pruning to one partial table."""
    num_states, num_traits = causal_states.shape
    num_pairs = num_traits * (num_traits + 1) // 2
    pair_counts = np.zeros(num_pairs, dtype=np.int64)

    for state in range(num_states):
        count = counts[state]
        if count == 0:
            continue
        for pair_offset in range(state_pair_counts[state]):
            pair_counts[
                state_pair_indices[state, pair_offset]
            ] += count

    for pair_number in range(num_pairs):
        if pair_counts[pair_number] == 0:
            return False

    if fit_ss:
        for trait in range(num_traits):
            causal_probability = (
                pair_counts[pair_index[trait, trait]] / intervals
            )
            if (
                abs(causal_probability - pi_causal_ss[trait])
                >= 1.0 / intervals
            ):
                return False

    scaled_omega = np.empty(
        (num_traits, num_traits), dtype=np.float64
    )
    for first_trait in range(num_traits):
        for second_trait in range(num_traits):
            scaled_omega[first_trait, second_trait] = (
                omega[first_trait, second_trait]
                * intervals
                / pair_counts[pair_index[first_trait, second_trait]]
            )

    # The tolerance is an upper bound on the final full-matrix tolerance.
    # Eigenvalue interlacing therefore makes this rejection conservative:
    # no table accepted by the historical full PSD rule can be pruned here.
    eigenvalues = np.linalg.eigvalsh(scaled_omega)
    return eigenvalues[0] >= -prune_tolerance


@njit(cache=True)
def _enumerate_branch_seed_counts_kernel(
    total_points,
    intervals,
    combinations,
    causal_states,
    state_pair_indices,
    state_pair_counts,
    pair_index,
    omega,
    prune_tolerance,
    fit_ss,
    pi_causal_ss,
):
    """Enumerate the small seed grid and retain safe partial tables."""
    num_states = causal_states.shape[0]
    retained = np.empty((total_points, num_states), dtype=np.int64)
    retained_count = 0
    counts = np.empty(num_states, dtype=np.int64)

    for rank in range(total_points):
        _unrank_composition(
            rank, intervals, num_states, combinations, counts
        )
        if _branch_counts_survive(
            counts,
            intervals,
            causal_states,
            state_pair_indices,
            state_pair_counts,
            pair_index,
            omega,
            prune_tolerance,
            fit_ss,
            pi_causal_ss,
        ):
            retained[retained_count] = counts
            retained_count += 1

    return retained[:retained_count].copy()


def enumerate_branch_seed_counts(
    intervals,
    causal_states,
    omega,
    prune_tolerance,
    pi_causal_ss=None,
):
    """Return conservatively feasible count tables for a small trait seed."""
    causal_states = np.asarray(causal_states, dtype=np.bool_)
    omega = np.asarray(omega, dtype=np.float64)
    num_states, num_traits = causal_states.shape
    total_points = automatic_grid_size(num_states, intervals)
    combinations = binomial_table(
        intervals + num_states - 1,
        max(num_states - 1, intervals),
    )
    causal_state_arrays = prepare_causal_state_arrays(causal_states)
    if pi_causal_ss is None:
        fit_ss = False
        pi_causal_ss = np.zeros(num_traits, dtype=np.float64)
    else:
        fit_ss = True
        pi_causal_ss = np.asarray(pi_causal_ss, dtype=np.float64)
    return _enumerate_branch_seed_counts_kernel(
        int(total_points),
        int(intervals),
        combinations,
        causal_states,
        causal_state_arrays[2],
        causal_state_arrays[3],
        causal_state_arrays[4],
        omega,
        float(prune_tolerance),
        fit_ss,
        pi_causal_ss,
    )


@njit(cache=True)
def _expand_and_prune_counts_kernel(
    parents,
    max_children,
    intervals,
    causal_states,
    state_pair_indices,
    state_pair_counts,
    pair_index,
    omega,
    prune_tolerance,
    fit_ss,
    pi_causal_ss,
):
    """Split each partial table for one new trait and prune descendants."""
    parent_states = parents.shape[1]
    child_states = parent_states * 2
    retained = np.empty((max_children, child_states), dtype=np.int64)
    retained_count = 0
    visited = 0

    for parent_index in range(len(parents)):
        parent = parents[parent_index]
        causal_counts = np.zeros(parent_states, dtype=np.int64)
        child = np.empty(child_states, dtype=np.int64)
        finished = False
        while not finished:
            visited += 1
            for state in range(parent_states):
                child[2 * state] = parent[state] - causal_counts[state]
                child[2 * state + 1] = causal_counts[state]

            if _branch_counts_survive(
                child,
                intervals,
                causal_states,
                state_pair_indices,
                state_pair_counts,
                pair_index,
                omega,
                prune_tolerance,
                fit_ss,
                pi_causal_ss,
            ):
                retained[retained_count] = child
                retained_count += 1

            position = parent_states - 1
            while position >= 0:
                if causal_counts[position] < parent[position]:
                    causal_counts[position] += 1
                    break
                causal_counts[position] = 0
                position -= 1
            if position < 0:
                finished = True

    return retained[:retained_count].copy(), visited


def branch_extension_count(parents):
    """Count the exact number of one-trait table splits as a Python int."""
    total = 0
    for parent in np.asarray(parents):
        total += math.prod(int(count) + 1 for count in parent)
    return total


def expand_and_prune_counts(
    parents,
    intervals,
    causal_states,
    omega,
    prune_tolerance,
    pi_causal_ss=None,
):
    """Expand one branch level and retain safe principal-matrix tables."""
    parents = np.asarray(parents, dtype=np.int64)
    causal_states = np.asarray(causal_states, dtype=np.bool_)
    omega = np.asarray(omega, dtype=np.float64)
    num_traits = causal_states.shape[1]
    max_children = branch_extension_count(parents)
    causal_state_arrays = prepare_causal_state_arrays(causal_states)
    if pi_causal_ss is None:
        fit_ss = False
        pi_causal_ss = np.zeros(num_traits, dtype=np.float64)
    else:
        fit_ss = True
        pi_causal_ss = np.asarray(pi_causal_ss, dtype=np.float64)
    return _expand_and_prune_counts_kernel(
        parents,
        int(max_children),
        int(intervals),
        causal_states,
        causal_state_arrays[2],
        causal_state_arrays[3],
        causal_state_arrays[4],
        omega,
        float(prune_tolerance),
        fit_ss,
        pi_causal_ss,
    )


def state_mapping_to_original_order(trait_order):
    """Map state indices in a reordered trait basis to original indices."""
    trait_order = np.asarray(trait_order, dtype=np.int64)
    num_traits = len(trait_order)
    num_states = 1 << num_traits
    mapping = np.empty(num_states, dtype=np.int64)
    for reordered_state in range(num_states):
        original_state = 0
        for reordered_trait, original_trait in enumerate(trait_order):
            bit = (
                reordered_state >> (num_traits - reordered_trait - 1)
            ) & 1
            original_state |= bit << (num_traits - original_trait - 1)
        mapping[reordered_state] = original_state
    return mapping


@njit(cache=True)
def _evaluate_branch_leaves_kernel(
    reordered_counts,
    reordered_to_original_state,
    intervals,
    causal_states,
    state_trait_ids,
    state_trait_counts,
    state_pair_indices,
    state_pair_counts,
    pair_index,
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
    """Evaluate complete branch leaves with exact historical semantics."""
    num_states, num_traits = causal_states.shape
    max_fdr = np.full(num_traits, -np.inf, dtype=np.float64)
    best_counts = np.zeros((num_traits, num_states), dtype=np.int64)
    candidate_fdr = np.empty(num_traits, dtype=np.float64)
    original_counts = np.empty(num_states, dtype=np.int64)
    feasible_count = 0
    invalid = False

    for row in range(len(reordered_counts)):
        original_counts[:] = 0
        for reordered_state in range(num_states):
            original_counts[
                reordered_to_original_state[reordered_state]
            ] = reordered_counts[row, reordered_state]

        if not _evaluate_counts_candidate(
            original_counts,
            intervals,
            causal_states,
            state_trait_ids,
            state_trait_counts,
            state_pair_indices,
            state_pair_counts,
            pair_index,
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
            candidate_fdr,
        ):
            continue

        feasible_count += 1
        for trait in range(num_traits):
            value = candidate_fdr[trait]
            if (
                not math.isfinite(value)
                or value < -1.0e-12
                or value > 1.0 + 1.0e-12
            ):
                invalid = True
            value = min(1.0, max(0.0, value))
            if (
                value > max_fdr[trait]
                or (
                    value == max_fdr[trait]
                    and _count_vector_precedes(
                        original_counts, best_counts[trait]
                    )
                )
            ):
                max_fdr[trait] = value
                best_counts[trait] = original_counts

    maximizing_probabilities = best_counts / intervals
    return max_fdr, maximizing_probabilities, feasible_count, invalid


@njit(inline="always")
def _count_vector_precedes(first, second):
    """Compare compositions in the historical simplex_walk order."""
    for state in range(len(first)):
        if first[state] != second[state]:
            return first[state] < second[state]
    return False


def evaluate_automatic_grid_max_branch(
    intervals,
    causal_states,
    omega,
    prepared,
    pi_causal_ss=None,
    candidate_limit=10_000_000,
    table_byte_limit=256 * 1024 * 1024,
):
    """Exactly maximize maxFDR by pruning impossible principal matrices."""
    if intervals <= 0:
        raise ValueError("maxFDR intervals must be positive")
    causal_states = np.asarray(causal_states, dtype=np.bool_)
    omega = np.asarray(omega, dtype=np.float64)
    num_states, num_traits = causal_states.shape
    if num_states != 1 << num_traits:
        raise ValueError("branch maxFDR requires all binary causal states")

    exhaustive_candidates = automatic_grid_size(num_states, intervals)
    prune_scale_bound = max(
        1.0,
        intervals * float(np.max(np.sum(np.abs(omega), axis=1))),
    )
    prune_tolerance = (
        np.finfo(np.float64).eps * prune_scale_bound * num_traits
    )
    if pi_causal_ss is None:
        original_pi = None
    else:
        original_pi = np.asarray(pi_causal_ss, dtype=np.float64)

    seed_traits = min(3, num_traits)
    seed_candidates = automatic_grid_size(1 << seed_traits, intervals)
    seed_required_bytes = seed_candidates * (1 << seed_traits) * 8
    if (
        seed_candidates > candidate_limit
        or seed_required_bytes > table_byte_limit
    ):
        raise BranchSearchLimitExceeded(
            "branch seed would visit {:,} candidates and allocate "
            "approximately {:.1f} MiB".format(
                seed_candidates,
                seed_required_bytes / (1024.0 * 1024.0),
            )
        )
    best_seed = None
    best_order = None
    seed_subsets_tested = 0
    for seed_order in itertools.combinations(range(num_traits), seed_traits):
        seed_subsets_tested += 1
        seed_order_array = np.asarray(seed_order, dtype=np.int64)
        seed_omega = omega[np.ix_(seed_order_array, seed_order_array)]
        seed_states = np.asarray(
            list(itertools.product((False, True), repeat=seed_traits)),
            dtype=np.bool_,
        )
        seed_pi = (
            None
            if original_pi is None
            else original_pi[seed_order_array]
        )
        retained = enumerate_branch_seed_counts(
            intervals,
            seed_states,
            seed_omega,
            prune_tolerance,
            pi_causal_ss=seed_pi,
        )
        if best_seed is None or len(retained) < len(best_seed):
            best_seed = retained
            best_order = list(seed_order)

    parents = best_seed
    order = best_order
    remaining = [trait for trait in range(num_traits) if trait not in order]
    level_diagnostics = []
    while remaining and len(parents):
        extension_count = branch_extension_count(parents)
        child_states = parents.shape[1] * 2
        required_bytes = extension_count * child_states * 8
        if (
            extension_count > candidate_limit
            or required_bytes > table_byte_limit
        ):
            raise BranchSearchLimitExceeded(
                "branch level would visit {:,} candidates and allocate "
                "approximately {:.1f} MiB".format(
                    extension_count, required_bytes / (1024.0 * 1024.0)
                )
            )

        best_children = None
        best_trait = None
        for candidate_trait in remaining:
            candidate_order = order + [candidate_trait]
            order_array = np.asarray(candidate_order, dtype=np.int64)
            candidate_omega = omega[np.ix_(order_array, order_array)]
            candidate_states = np.asarray(
                list(
                    itertools.product(
                        (False, True), repeat=len(candidate_order)
                    )
                ),
                dtype=np.bool_,
            )
            candidate_pi = (
                None
                if original_pi is None
                else original_pi[order_array]
            )
            children, visited = expand_and_prune_counts(
                parents,
                intervals,
                candidate_states,
                candidate_omega,
                prune_tolerance,
                pi_causal_ss=candidate_pi,
            )
            if visited != extension_count:
                raise RuntimeError("branch maxFDR extension count mismatch")
            if best_children is None or len(children) < len(best_children):
                best_children = children
                best_trait = candidate_trait

        order.append(best_trait)
        remaining.remove(best_trait)
        parents = best_children
        level_diagnostics.append(
            {
                "trait": int(best_trait),
                "candidates_per_choice": int(extension_count),
                "choices_tested": int(len(remaining) + 1),
                "retained": int(len(parents)),
            }
        )

    if remaining:
        order.extend(remaining)
        parents = np.empty((0, num_states), dtype=np.int64)

    prepared_arrays = prepare_fdr_arrays(prepared)
    causal_state_arrays = prepare_causal_state_arrays(causal_states)
    state_mapping = state_mapping_to_original_order(order)
    if original_pi is None:
        fit_ss = False
        final_pi = np.zeros(num_traits, dtype=np.float64)
    else:
        fit_ss = True
        final_pi = original_pi
    max_fdr, maximizing_probabilities, feasible_count, invalid = (
        _evaluate_branch_leaves_kernel(
            parents,
            state_mapping,
            int(intervals),
            causal_states,
            *causal_state_arrays,
            omega,
            *prepared_arrays,
            fit_ss,
            final_pi,
        )
    )
    if invalid:
        raise ValueError(
            "maxFDR branch search returned a non-finite value or a value "
            "outside [0, 1]"
        )
    diagnostics = {
        "exhaustive_candidates": int(exhaustive_candidates),
        "trait_order": [int(trait) for trait in order],
        "seed_traits": int(seed_traits),
        "seed_candidates_per_subset": int(seed_candidates),
        "seed_subsets_tested": int(seed_subsets_tested),
        "seed_retained": int(len(best_seed)),
        "levels": level_diagnostics,
        "final_pruned_leaves": int(len(parents)),
    }
    return (
        max_fdr,
        maximizing_probabilities,
        int(feasible_count),
        diagnostics,
    )


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


def evaluate_automatic_grid_max(
    intervals,
    causal_states,
    omega,
    prepared,
    pi_causal_ss=None,
    chunk_size=1_000_000,
    progress_callback=None,
):
    """Reduce an automatic grid to each trait's maximum in bounded memory."""
    if chunk_size <= 0:
        raise ValueError("Numba maxFDR chunk size must be positive")

    causal_states = np.asarray(causal_states, dtype=np.bool_)
    omega = np.asarray(omega, dtype=np.float64)
    num_states, num_traits = causal_states.shape
    total_points = automatic_grid_size(num_states, intervals)
    combinations = binomial_table(
        intervals + num_states - 1,
        max(num_states - 1, intervals),
    )
    prepared_arrays = prepare_fdr_arrays(prepared)
    causal_state_arrays = prepare_causal_state_arrays(causal_states)
    if pi_causal_ss is None:
        fit_ss = False
        pi_causal_ss = np.zeros(num_traits, dtype=np.float64)
    else:
        fit_ss = True
        pi_causal_ss = np.asarray(pi_causal_ss, dtype=np.float64)

    max_fdr = np.full(num_traits, -np.inf, dtype=float)
    maximizing_probabilities = np.empty(
        (num_traits, num_states), dtype=float
    )
    feasible_count = 0

    for start_rank in range(0, total_points, chunk_size):
        stop_rank = min(start_rank + chunk_size, total_points)
        (
            block_max,
            block_ranks,
            block_feasible_counts,
            block_invalid,
        ) = _evaluate_automatic_grid_max_chunk_sparse(
            int(start_rank),
            int(stop_rank),
            int(total_points),
            int(intervals),
            combinations,
            causal_states,
            *causal_state_arrays,
            omega,
            *prepared_arrays,
            fit_ss,
            pi_causal_ss,
        )
        if np.any(block_invalid):
            raise ValueError(
                "maxFDR grid search returned a non-finite value or a "
                "value outside [0, 1]"
            )
        chunk_feasible_count = int(np.sum(block_feasible_counts))
        feasible_count += chunk_feasible_count
        if chunk_feasible_count:
            trait_indices = np.arange(num_traits)
            chunk_block_indices = np.argmax(block_max, axis=0)
            chunk_max = block_max[
                chunk_block_indices, trait_indices
            ]
            improved = chunk_max > max_fdr
            max_fdr[improved] = chunk_max[improved]
            for trait in np.flatnonzero(improved):
                counts = np.empty(num_states, dtype=np.int64)
                _unrank_composition(
                    int(block_ranks[chunk_block_indices[trait], trait]),
                    intervals,
                    num_states,
                    combinations,
                    counts,
                )
                maximizing_probabilities[trait] = counts / intervals
        if progress_callback is not None:
            progress_callback(stop_rank, total_points)

    return max_fdr, maximizing_probabilities, feasible_count
