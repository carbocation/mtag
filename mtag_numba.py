"""Optional Numba kernels for fused automatic maxFDR grid evaluation."""

import itertools
import math

import numpy as np
from numba import njit, prange


class BranchSearchLimitExceeded(RuntimeError):
    """Raised when exact branch traversal would exceed its memory guard."""


class SeedOrderNotCompetitive(RuntimeError):
    """Raised when a seed has already retained as many rows as the best."""


class SparseProbabilityRows:
    """Compact maximizing probability rows for very large causal spaces."""

    def __init__(
        self, state_ids, counts, occupied, intervals, num_states
    ):
        self.state_ids = np.asarray(state_ids, dtype=np.uint64)
        self.counts = np.asarray(counts, dtype=np.int64)
        self.occupied = np.asarray(occupied, dtype=np.int64)
        self.intervals = int(intervals)
        self.shape = (len(self.occupied), int(num_states))

    def to_dense(self):
        dense = np.zeros(self.shape, dtype=np.float64)
        for row, occupied in enumerate(self.occupied):
            ids = self.state_ids[row, :occupied].astype(np.intp)
            dense[row, ids] = self.counts[row, :occupied] / self.intervals
        return dense

    def __array__(self, dtype=None, copy=None):
        dense = self.to_dense()
        if dtype is not None:
            dense = dense.astype(dtype, copy=False)
        if copy:
            dense = dense.copy()
        return dense

    def format_row(self, row):
        occupied = int(self.occupied[row])
        entries = [
            (int(self.state_ids[row, index]),
             self.counts[row, index] / self.intervals)
            for index in range(occupied)
        ]
        return "sparse(state_index: probability) {} of {:,} states".format(
            dict(entries), self.shape[1]
        )


def nominal_grid_size(num_states, intervals):
    """Return the Python-integer size of a simplex grid of any rank."""
    return math.comb(intervals + num_states - 1, num_states - 1)


def automatic_grid_size(num_states, intervals):
    """Return the number of lattice points in the automatic simplex grid."""
    total_points = nominal_grid_size(num_states, intervals)
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




class SparseBranchTables:
    """Fixed-width occupied-state tables carried between branch levels."""

    def __init__(
        self,
        state_ids,
        counts,
        occupied,
        pair_counts,
        cholesky,
        factor_valid,
    ):
        self.state_ids = np.asarray(state_ids, dtype=np.uint64)
        self.counts = np.asarray(counts, dtype=np.int64)
        self.occupied = np.asarray(occupied, dtype=np.int64)
        self.pair_counts = np.asarray(pair_counts, dtype=np.int64)
        self.cholesky = np.asarray(cholesky, dtype=np.float64)
        self.factor_valid = np.asarray(factor_valid, dtype=np.uint8)

    def __len__(self):
        return len(self.occupied)


@njit(inline="always")
def _lower_pair_index(first_trait, second_trait):
    """Return an append-stable lower-triangle pair index."""
    high = max(first_trait, second_trait)
    low = min(first_trait, second_trait)
    return high * (high + 1) // 2 + low


@njit(inline="always")
def _sparse_state_has_trait(state, trait, num_traits):
    shift = num_traits - trait - 1
    return ((state >> np.uint64(shift)) & np.uint64(1)) != 0


@njit(inline="always")
def _build_scaled_omega_lower(
    pair_counts, intervals, omega, scaled_omega
):
    num_traits = omega.shape[0]
    for first_trait in range(num_traits):
        for second_trait in range(num_traits):
            scaled_omega[first_trait, second_trait] = (
                omega[first_trait, second_trait]
                * intervals
                / pair_counts[
                    _lower_pair_index(first_trait, second_trait)
                ]
            )


@njit(inline="always")
def _clear_cholesky(matrix, tolerance, output):
    """Factor a clearly positive-definite matrix, or decline safely."""
    size = matrix.shape[0]
    output[:] = 0.0
    margin = 16.0 * tolerance
    for row in range(size):
        for column in range(row + 1):
            value = matrix[row, column]
            for offset in range(column):
                value -= output[row, offset] * output[column, offset]
            if row == column:
                if not math.isfinite(value) or value <= margin:
                    output[:] = 0.0
                    return False
                output[row, column] = math.sqrt(value)
            else:
                output[row, column] = value / output[column, column]
    return True


@njit(inline="always")
def _seed_pair_counts(
    state_ids, counts, occupied, num_traits, pair_counts
):
    pair_counts[:] = 0
    for occupied_index in range(occupied):
        state = state_ids[occupied_index]
        count = counts[occupied_index]
        for high_trait in range(num_traits):
            if not _sparse_state_has_trait(
                state, high_trait, num_traits
            ):
                continue
            for low_trait in range(high_trait + 1):
                if _sparse_state_has_trait(
                    state, low_trait, num_traits
                ):
                    pair_counts[
                        _lower_pair_index(high_trait, low_trait)
                    ] += count


@njit(inline="always")
def _marginals_match_spike_slab_lower(
    pair_counts, intervals, num_traits, pi_causal_ss
):
    for trait in range(num_traits):
        causal_probability = (
            pair_counts[_lower_pair_index(trait, trait)] / intervals
        )
        if (
            abs(causal_probability - pi_causal_ss[trait])
            >= 1.0 / intervals
        ):
            return False
    return True


@njit(cache=True)
def _enumerate_sparse_seed_chunk_kernel(
    start_rank,
    stop_rank,
    total_points,
    intervals,
    num_traits,
    combinations,
    omega,
    prune_tolerance,
    fit_ss,
    pi_causal_ss,
    retained_limit,
):
    """Enumerate one seed-grid chunk into occupied-state tables."""
    chunk_size = stop_rank - start_rank
    num_states = 1 << num_traits
    num_pairs = num_traits * (num_traits + 1) // 2
    retained_ids = np.zeros(
        (chunk_size, intervals), dtype=np.uint64
    )
    retained_counts = np.zeros(
        (chunk_size, intervals), dtype=np.int64
    )
    retained_occupied = np.zeros(chunk_size, dtype=np.int64)
    retained_pairs = np.zeros(
        (chunk_size, num_pairs), dtype=np.int64
    )
    retained_cholesky = np.zeros(
        (chunk_size, num_traits, num_traits), dtype=np.float64
    )
    retained_factor_valid = np.zeros(chunk_size, dtype=np.uint8)
    state_ids = np.empty(intervals, dtype=np.uint64)
    counts = np.empty(intervals, dtype=np.int64)
    pair_counts = np.empty(num_pairs, dtype=np.int64)
    scaled_omega = np.empty(
        (num_traits, num_traits), dtype=np.float64
    )
    factor = np.empty((num_traits, num_traits), dtype=np.float64)
    retained_count = 0

    for rank in range(start_rank, stop_rank):
        counts[:] = 0
        occupied = _unrank_sparse_composition(
            rank,
            total_points,
            intervals,
            num_states,
            combinations,
            state_ids,
            counts,
        )
        _seed_pair_counts(
            state_ids, counts, occupied, num_traits, pair_counts
        )
        valid = True
        for pair_number in range(num_pairs):
            if pair_counts[pair_number] == 0:
                valid = False
                break
        if not valid:
            continue
        if fit_ss and not _marginals_match_spike_slab_lower(
            pair_counts,
            intervals,
            num_traits,
            pi_causal_ss,
        ):
            continue

        _build_scaled_omega_lower(
            pair_counts, intervals, omega, scaled_omega
        )
        eigenvalues = np.linalg.eigvalsh(scaled_omega)
        if eigenvalues[0] < -prune_tolerance:
            continue

        retained_ids[retained_count, :occupied] = state_ids[:occupied]
        retained_counts[retained_count, :occupied] = counts[:occupied]
        retained_occupied[retained_count] = occupied
        retained_pairs[retained_count] = pair_counts
        if _clear_cholesky(
            scaled_omega, prune_tolerance, factor
        ):
            retained_cholesky[retained_count] = factor
            retained_factor_valid[retained_count] = 1
        retained_count += 1
        if retained_limit >= 0 and retained_count > retained_limit:
            break

    return (
        retained_ids[:retained_count].copy(),
        retained_counts[:retained_count].copy(),
        retained_occupied[:retained_count].copy(),
        retained_pairs[:retained_count].copy(),
        retained_cholesky[:retained_count].copy(),
        retained_factor_valid[:retained_count].copy(),
    )


def _empty_sparse_branch_tables(intervals, num_traits):
    num_pairs = num_traits * (num_traits + 1) // 2
    return SparseBranchTables(
        np.empty((0, intervals), dtype=np.uint64),
        np.empty((0, intervals), dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty((0, num_pairs), dtype=np.int64),
        np.empty((0, num_traits, num_traits), dtype=np.float64),
        np.empty(0, dtype=np.uint8),
    )


def _concatenate_sparse_branch_tables(chunks, intervals, num_traits):
    chunks = [chunk for chunk in chunks if len(chunk)]
    if not chunks:
        return _empty_sparse_branch_tables(intervals, num_traits)
    return SparseBranchTables(
        np.concatenate([chunk.state_ids for chunk in chunks]),
        np.concatenate([chunk.counts for chunk in chunks]),
        np.concatenate([chunk.occupied for chunk in chunks]),
        np.concatenate([chunk.pair_counts for chunk in chunks]),
        np.concatenate([chunk.cholesky for chunk in chunks]),
        np.concatenate([chunk.factor_valid for chunk in chunks]),
    )


def _sparse_table_bytes_per_row(intervals, num_traits):
    num_pairs = num_traits * (num_traits + 1) // 2
    return (
        16 * intervals
        + 8
        + 8 * num_pairs
        + 8 * num_traits * num_traits
        + 1
    )


def enumerate_sparse_branch_seed(
    intervals,
    num_traits,
    omega,
    prune_tolerance,
    pi_causal_ss=None,
    chunk_size=100_000,
    retained_byte_limit=256 * 1024 * 1024,
    retained_row_limit=None,
):
    """Enumerate a bounded-memory exact seed grid."""
    num_states = 1 << num_traits
    total_points = automatic_grid_size(num_states, intervals)
    combinations = binomial_table(
        intervals + num_states - 1,
        max(num_states - 1, intervals),
    )
    if pi_causal_ss is None:
        fit_ss = False
        pi_causal_ss = np.zeros(num_traits, dtype=np.float64)
    else:
        fit_ss = True
        pi_causal_ss = np.asarray(pi_causal_ss, dtype=np.float64)
    chunks = []
    retained_rows = 0
    bytes_per_row = _sparse_table_bytes_per_row(
        intervals, num_traits
    )
    for start_rank in range(0, total_points, chunk_size):
        stop_rank = min(start_rank + chunk_size, total_points)
        chunk_retained_limit = -1
        if retained_row_limit is not None:
            chunk_retained_limit = retained_row_limit - retained_rows
        arrays = _enumerate_sparse_seed_chunk_kernel(
            int(start_rank),
            int(stop_rank),
            int(total_points),
            int(intervals),
            int(num_traits),
            combinations,
            np.asarray(omega, dtype=np.float64),
            float(prune_tolerance),
            fit_ss,
            pi_causal_ss,
            int(chunk_retained_limit),
        )
        chunk = SparseBranchTables(*arrays)
        retained_rows += len(chunk)
        if (
            retained_row_limit is not None
            and retained_rows > retained_row_limit
        ):
            raise SeedOrderNotCompetitive
        if retained_rows * bytes_per_row * 2 > retained_byte_limit:
            raise BranchSearchLimitExceeded(
                "branch seed retained tables would require more than "
                "{:.1f} MiB including concatenation".format(
                    retained_byte_limit / (1024.0 * 1024.0)
                )
            )
        chunks.append(chunk)
    return _concatenate_sparse_branch_tables(
        chunks, intervals, num_traits
    )


def sparse_branch_extension_count(parents):
    """Count all exact one-trait splits of occupied parent states."""
    total = 0
    for row in range(len(parents)):
        occupied = int(parents.occupied[row])
        total += math.prod(
            int(count) + 1
            for count in parents.counts[row, :occupied]
        )
    return total


@njit(inline="always")
def _full_principal_psd_check(
    pair_counts,
    intervals,
    omega,
    prune_tolerance,
    factor,
):
    size = omega.shape[0]
    matrix = np.empty((size, size), dtype=np.float64)
    _build_scaled_omega_lower(pair_counts, intervals, omega, matrix)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] < -prune_tolerance:
        factor[:] = 0.0
        return False, False
    factor_valid = _clear_cholesky(
        matrix, prune_tolerance, factor
    )
    return True, factor_valid


@njit(inline="always")
def _bordered_principal_psd_check(
    parent_factor,
    parent_factor_valid,
    pair_counts,
    intervals,
    omega,
    prune_tolerance,
    child_factor,
):
    """Check a bordered PSD matrix, reusing a clear parent factor."""
    size = omega.shape[0]
    parent_size = size - 1
    child_factor[:] = 0.0
    if parent_factor_valid:
        right = np.empty(parent_size, dtype=np.float64)
        solved = np.empty(parent_size, dtype=np.float64)
        inverse_product = np.empty(parent_size, dtype=np.float64)
        old_pairs = parent_size * (parent_size + 1) // 2
        for trait in range(parent_size):
            right[trait] = (
                omega[trait, parent_size]
                * intervals
                / pair_counts[old_pairs + trait]
            )

        for row in range(parent_size):
            value = right[row]
            for column in range(row):
                value -= parent_factor[row, column] * solved[column]
            solved[row] = value / parent_factor[row, row]

        for row in range(parent_size - 1, -1, -1):
            value = solved[row]
            for column in range(row + 1, parent_size):
                value -= (
                    parent_factor[column, row]
                    * inverse_product[column]
                )
            inverse_product[row] = value / parent_factor[row, row]

        diagonal = (
            omega[parent_size, parent_size]
            * intervals
            / pair_counts[old_pairs + parent_size]
        )
        schur = diagonal
        for trait in range(parent_size):
            schur -= solved[trait] * solved[trait]
        inverse_norm = 1.0
        for value in inverse_product:
            inverse_norm += value * value
        margin = 16.0 * prune_tolerance

        if schur > margin * inverse_norm:
            child_factor[:parent_size, :parent_size] = parent_factor
            for trait in range(parent_size):
                child_factor[parent_size, trait] = solved[trait]
            child_factor[parent_size, parent_size] = math.sqrt(schur)
            return True, True, 1

        rayleigh_upper = schur / inverse_norm
        if rayleigh_upper < -margin:
            return False, False, -1

    valid, factor_valid = _full_principal_psd_check(
        pair_counts,
        intervals,
        omega,
        prune_tolerance,
        child_factor,
    )
    return valid, factor_valid, 0


@njit(cache=True)
def _expand_sparse_branch_kernel(
    parent_ids,
    parent_counts,
    parent_occupied,
    parent_pair_counts,
    parent_cholesky,
    parent_factor_valid,
    max_children,
    intervals,
    omega,
    prune_tolerance,
    fit_ss,
    pi_causal_ss,
):
    """Incrementally split sparse tables and check bordered PSD matrices."""
    parent_traits = parent_cholesky.shape[1]
    child_traits = parent_traits + 1
    parent_pairs = parent_traits * (parent_traits + 1) // 2
    child_pairs = child_traits * (child_traits + 1) // 2
    retained_ids = np.zeros(
        (max_children, intervals), dtype=np.uint64
    )
    retained_counts = np.zeros(
        (max_children, intervals), dtype=np.int64
    )
    retained_occupied = np.zeros(max_children, dtype=np.int64)
    retained_pairs = np.zeros(
        (max_children, child_pairs), dtype=np.int64
    )
    retained_cholesky = np.zeros(
        (max_children, child_traits, child_traits), dtype=np.float64
    )
    retained_factor_valid = np.zeros(max_children, dtype=np.uint8)
    causal_counts = np.zeros(intervals, dtype=np.int64)
    pair_counts = np.empty(child_pairs, dtype=np.int64)
    child_factor = np.empty(
        (child_traits, child_traits), dtype=np.float64
    )
    retained_count = 0
    visited = 0
    fast_accepts = 0
    fast_rejects = 0
    eigen_fallbacks = 0

    for parent_index in range(len(parent_occupied)):
        occupied = parent_occupied[parent_index]
        causal_counts[:] = 0
        finished = False
        while not finished:
            visited += 1
            pair_counts[:parent_pairs] = parent_pair_counts[parent_index]
            for trait in range(parent_traits + 1):
                pair_counts[parent_pairs + trait] = 0

            for occupied_index in range(occupied):
                causal_count = causal_counts[occupied_index]
                if causal_count == 0:
                    continue
                state = parent_ids[parent_index, occupied_index]
                for trait in range(parent_traits):
                    if _sparse_state_has_trait(
                        state, trait, parent_traits
                    ):
                        pair_counts[parent_pairs + trait] += causal_count
                pair_counts[parent_pairs + parent_traits] += causal_count

            valid_pairs = True
            for pair_number in range(parent_pairs, child_pairs):
                if pair_counts[pair_number] == 0:
                    valid_pairs = False
                    break
            if valid_pairs and (
                not fit_ss
                or abs(
                    pair_counts[child_pairs - 1] / intervals
                    - pi_causal_ss[parent_traits]
                ) < 1.0 / intervals
            ):
                valid, factor_valid, fast_status = (
                    _bordered_principal_psd_check(
                        parent_cholesky[parent_index],
                        parent_factor_valid[parent_index] != 0,
                        pair_counts,
                        intervals,
                        omega,
                        prune_tolerance,
                        child_factor,
                    )
                )
                if fast_status > 0:
                    fast_accepts += 1
                elif fast_status < 0:
                    fast_rejects += 1
                else:
                    eigen_fallbacks += 1
                if valid:
                    child_occupied = 0
                    for occupied_index in range(occupied):
                        parent_state = parent_ids[
                            parent_index, occupied_index
                        ]
                        causal_count = causal_counts[occupied_index]
                        noncausal_count = (
                            parent_counts[parent_index, occupied_index]
                            - causal_count
                        )
                        if noncausal_count:
                            retained_ids[
                                retained_count, child_occupied
                            ] = parent_state * np.uint64(2)
                            retained_counts[
                                retained_count, child_occupied
                            ] = noncausal_count
                            child_occupied += 1
                        if causal_count:
                            retained_ids[
                                retained_count, child_occupied
                            ] = parent_state * np.uint64(2) + np.uint64(1)
                            retained_counts[
                                retained_count, child_occupied
                            ] = causal_count
                            child_occupied += 1
                    retained_occupied[retained_count] = child_occupied
                    retained_pairs[retained_count] = pair_counts
                    retained_cholesky[retained_count] = child_factor
                    if factor_valid:
                        retained_factor_valid[retained_count] = 1
                    retained_count += 1

            position = occupied - 1
            while position >= 0:
                if (
                    causal_counts[position]
                    < parent_counts[parent_index, position]
                ):
                    causal_counts[position] += 1
                    break
                causal_counts[position] = 0
                position -= 1
            if position < 0:
                finished = True

    return (
        retained_ids[:retained_count].copy(),
        retained_counts[:retained_count].copy(),
        retained_occupied[:retained_count].copy(),
        retained_pairs[:retained_count].copy(),
        retained_cholesky[:retained_count].copy(),
        retained_factor_valid[:retained_count].copy(),
        visited,
        fast_accepts,
        fast_rejects,
        eigen_fallbacks,
    )


def expand_sparse_branch_tables(
    parents,
    intervals,
    omega,
    prune_tolerance,
    pi_causal_ss=None,
):
    """Expand one sparse level using incremental counts and PSD factors."""
    child_traits = parents.cholesky.shape[1] + 1
    if pi_causal_ss is None:
        fit_ss = False
        pi_causal_ss = np.zeros(child_traits, dtype=np.float64)
    else:
        fit_ss = True
        pi_causal_ss = np.asarray(pi_causal_ss, dtype=np.float64)
    max_children = sparse_branch_extension_count(parents)
    result = _expand_sparse_branch_kernel(
        parents.state_ids,
        parents.counts,
        parents.occupied,
        parents.pair_counts,
        parents.cholesky,
        parents.factor_valid,
        int(max_children),
        int(intervals),
        np.asarray(omega, dtype=np.float64),
        float(prune_tolerance),
        fit_ss,
        pi_causal_ss,
    )
    return SparseBranchTables(*result[:6]), result[6:]


def _absolute_omega_correlation(omega):
    diagonal = np.maximum(np.diag(omega), np.finfo(float).tiny)
    scale = np.sqrt(np.outer(diagonal, diagonal))
    correlation = np.abs(omega / scale)
    np.fill_diagonal(correlation, 0.0)
    return correlation


def candidate_seed_orders(omega, seed_traits=3, trial_limit=32):
    """Choose a bounded deterministic set of promising seed orders."""
    num_traits = len(omega)
    if seed_traits >= num_traits:
        return [tuple(range(num_traits))]
    correlation = _absolute_omega_correlation(omega)

    def score(order):
        return sum(
            correlation[order[first], order[second]]
            for first in range(len(order))
            for second in range(first + 1, len(order))
        )

    total_subsets = math.comb(num_traits, seed_traits)
    if total_subsets <= trial_limit:
        orders = list(
            itertools.combinations(range(num_traits), seed_traits)
        )
        return sorted(orders, key=lambda order: (-score(order), order))
    if seed_traits != 3:
        return [tuple(range(seed_traits))]

    candidates = []
    seen = set()

    def add(candidate):
        candidate = tuple(sorted(candidate))
        if len(set(candidate)) == 3 and candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    add((0, 1, 2))
    pairs = [
        (correlation[first, second], first, second)
        for first in range(num_traits)
        for second in range(first + 1, num_traits)
    ]
    pairs.sort(key=lambda value: (-value[0], value[1], value[2]))
    for _, first, second in pairs:
        possible = [
            trait
            for trait in range(num_traits)
            if trait not in (first, second)
        ]
        third = max(
            possible,
            key=lambda trait: (
                correlation[first, trait] + correlation[second, trait],
                -trait,
            ),
        )
        add((first, second, third))
        if len(candidates) >= trial_limit:
            break
    return sorted(
        candidates, key=lambda order: (-score(order), order)
    )


def candidate_extension_traits(
    omega, order, remaining, trial_limit=16
):
    """Bound trait-order trials while leaving branch enumeration exact."""
    if len(remaining) <= trial_limit:
        return list(remaining)
    correlation = _absolute_omega_correlation(omega)
    return sorted(
        remaining,
        key=lambda trait: (
            -sum(correlation[trait, selected] for selected in order),
            trait,
        ),
    )[:trial_limit]


@njit(inline="always")
def _map_sparse_state_to_original(state, trait_order):
    num_traits = len(trait_order)
    original_state = np.uint64(0)
    for reordered_trait in range(num_traits):
        if _sparse_state_has_trait(
            state, reordered_trait, num_traits
        ):
            original_trait = trait_order[reordered_trait]
            shift = num_traits - original_trait - 1
            original_state |= np.uint64(1) << np.uint64(shift)
    return original_state


@njit(inline="always")
def _sort_sparse_state_counts(state_ids, counts, occupied):
    for index in range(1, occupied):
        state = state_ids[index]
        count = counts[index]
        position = index - 1
        while position >= 0 and state_ids[position] > state:
            state_ids[position + 1] = state_ids[position]
            counts[position + 1] = counts[position]
            position -= 1
        state_ids[position + 1] = state
        counts[position + 1] = count


@njit(inline="always")
def _sparse_count_vector_precedes(
    first_ids,
    first_counts,
    first_occupied,
    second_ids,
    second_counts,
    second_occupied,
):
    first_index = 0
    second_index = 0
    while first_index < first_occupied or second_index < second_occupied:
        if (
            second_index >= second_occupied
            or (
                first_index < first_occupied
                and first_ids[first_index] < second_ids[second_index]
            )
        ):
            first_count = first_counts[first_index]
            second_count = 0
            first_index += 1
        elif (
            first_index >= first_occupied
            or second_ids[second_index] < first_ids[first_index]
        ):
            first_count = 0
            second_count = second_counts[second_index]
            second_index += 1
        else:
            first_count = first_counts[first_index]
            second_count = second_counts[second_index]
            first_index += 1
            second_index += 1
        if first_count != second_count:
            return first_count < second_count
    return False


@njit(cache=True)
def _evaluate_sparse_branch_leaves_kernel(
    state_ids,
    counts,
    occupied,
    trait_order,
    intervals,
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
    """Evaluate sparse leaves with exact final PSD and FDR semantics."""
    num_traits = omega.shape[0]
    max_fdr = np.full(num_traits, -np.inf, dtype=np.float64)
    best_ids = np.zeros((num_traits, intervals), dtype=np.uint64)
    best_counts = np.zeros((num_traits, intervals), dtype=np.int64)
    best_occupied = np.zeros(num_traits, dtype=np.int64)
    candidate_original_ids = np.empty(intervals, dtype=np.uint64)
    candidate_original_counts = np.empty(intervals, dtype=np.int64)
    candidate_fdr = np.empty(num_traits, dtype=np.float64)
    original_pair_counts = np.empty(
        num_traits * (num_traits + 1) // 2, dtype=np.int64
    )
    scaled_omega = np.empty(
        (num_traits, num_traits), dtype=np.float64
    )
    sqrt_two = math.sqrt(2.0)
    eps = np.finfo(np.float64).eps
    feasible_count = 0
    invalid = False

    for row in range(len(occupied)):
        row_occupied = occupied[row]
        for occupied_index in range(row_occupied):
            candidate_original_ids[occupied_index] = (
                _map_sparse_state_to_original(
                    state_ids[row, occupied_index], trait_order
                )
            )
            candidate_original_counts[occupied_index] = counts[
                row, occupied_index
            ]
        _sort_sparse_state_counts(
            candidate_original_ids,
            candidate_original_counts,
            row_occupied,
        )
        _seed_pair_counts(
            candidate_original_ids,
            candidate_original_counts,
            row_occupied,
            num_traits,
            original_pair_counts,
        )
        if fit_ss and not _marginals_match_spike_slab_lower(
            original_pair_counts,
            intervals,
            num_traits,
            pi_causal_ss,
        ):
            continue
        _build_scaled_omega_lower(
            original_pair_counts, intervals, omega, scaled_omega
        )
        psd_status = _fast_psd_status(scaled_omega)
        if psd_status < 0:
            continue
        if psd_status == 0:
            eigenvalues = np.linalg.eigvalsh(scaled_omega)
            max_abs_eigenvalue = 1.0
            for eigenvalue in eigenvalues:
                max_abs_eigenvalue = max(
                    max_abs_eigenvalue, abs(eigenvalue)
                )
            tolerance = eps * max_abs_eigenvalue * num_traits
            if eigenvalues[0] < -tolerance:
                continue

        for trait in range(num_traits):
            total_power = 0.0
            false_discovery_power = 0.0
            for occupied_index in range(row_occupied):
                state = candidate_original_ids[occupied_index]
                probability_significant = 0.0
                for n_index in range(len(n_counts)):
                    omega_numerator = 0.0
                    for first_trait in range(num_traits):
                        if not _sparse_state_has_trait(
                            state, first_trait, num_traits
                        ):
                            continue
                        for second_trait in range(num_traits):
                            if _sparse_state_has_trait(
                                state, second_trait, num_traits
                            ):
                                omega_numerator += (
                                    num_left[
                                        trait, n_index, first_trait
                                    ]
                                    * scaled_omega[
                                        first_trait, second_trait
                                    ]
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
                    probability_significant
                    * candidate_original_counts[occupied_index]
                    / intervals
                )
                total_power += state_power
                if not _sparse_state_has_trait(
                    state, trait, num_traits
                ):
                    false_discovery_power += state_power
            candidate_fdr[trait] = false_discovery_power / total_power

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
                    and _sparse_count_vector_precedes(
                        candidate_original_ids,
                        candidate_original_counts,
                        row_occupied,
                        best_ids[trait],
                        best_counts[trait],
                        best_occupied[trait],
                    )
                )
            ):
                max_fdr[trait] = value
                best_ids[trait, :row_occupied] = (
                    candidate_original_ids[:row_occupied]
                )
                best_counts[trait, :row_occupied] = (
                    candidate_original_counts[:row_occupied]
                )
                best_occupied[trait] = row_occupied

    return (
        max_fdr,
        best_ids,
        best_counts,
        best_occupied,
        feasible_count,
        invalid,
    )


def _materialize_sparse_probability_rows(
    state_ids,
    counts,
    occupied,
    intervals,
    num_states,
    byte_limit,
):
    required_bytes = len(occupied) * num_states * 8
    sparse = SparseProbabilityRows(
        state_ids, counts, occupied, intervals, num_states
    )
    if required_bytes > byte_limit:
        return sparse
    return sparse.to_dense()


def evaluate_automatic_grid_max_branch(
    intervals,
    causal_states,
    omega,
    prepared,
    pi_causal_ss=None,
    candidate_limit=10_000_000,
    table_byte_limit=256 * 1024 * 1024,
    seed_trial_limit=32,
    extension_trial_limit=16,
):
    """Exactly maximize maxFDR with sparse, incremental branch pruning."""
    if intervals <= 0:
        raise ValueError("maxFDR intervals must be positive")
    omega = np.asarray(omega, dtype=np.float64)
    if np.isscalar(causal_states):
        num_traits = int(causal_states)
    else:
        causal_states = np.asarray(causal_states, dtype=np.bool_)
        if causal_states.ndim != 2:
            raise ValueError("causal states must be a matrix or trait count")
        num_states_from_input, num_traits = causal_states.shape
        if num_states_from_input != 1 << num_traits:
            raise ValueError(
                "branch maxFDR requires all binary causal states"
            )
    if num_traits <= 0 or omega.shape != (num_traits, num_traits):
        raise ValueError("branch maxFDR trait dimensions do not match")
    if num_traits > 63:
        raise BranchSearchLimitExceeded(
            "sparse branch state identifiers currently support at most "
            "63 traits"
        )

    num_states = 1 << num_traits
    exhaustive_candidates = nominal_grid_size(num_states, intervals)
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
        if original_pi.shape != (num_traits,):
            raise ValueError("spike-slab probabilities must match traits")

    seed_traits = min(3, num_traits)
    seed_candidates = automatic_grid_size(1 << seed_traits, intervals)
    if seed_candidates > candidate_limit:
        raise BranchSearchLimitExceeded(
            "each branch seed would visit {:,} candidates".format(
                seed_candidates
            )
        )
    seed_bytes_per_row = _sparse_table_bytes_per_row(
        intervals, seed_traits
    )
    seed_chunk_size = max(
        1,
        min(
            100_000,
            int(table_byte_limit // max(1, seed_bytes_per_row * 2)),
        ),
    )
    seed_orders = candidate_seed_orders(
        omega,
        seed_traits=seed_traits,
        trial_limit=seed_trial_limit,
    )
    best_seed = None
    best_order = None
    seed_orders_skipped = 0
    seed_orders_uncompetitive = 0
    for seed_order in seed_orders:
        seed_order_array = np.asarray(seed_order, dtype=np.int64)
        seed_omega = omega[np.ix_(seed_order_array, seed_order_array)]
        seed_pi = (
            None
            if original_pi is None
            else original_pi[seed_order_array]
        )
        try:
            retained = enumerate_sparse_branch_seed(
                intervals,
                seed_traits,
                seed_omega,
                prune_tolerance,
                pi_causal_ss=seed_pi,
                chunk_size=seed_chunk_size,
                retained_byte_limit=table_byte_limit,
                retained_row_limit=(
                    None if best_seed is None else len(best_seed) - 1
                ),
            )
        except SeedOrderNotCompetitive:
            seed_orders_uncompetitive += 1
            continue
        except BranchSearchLimitExceeded:
            seed_orders_skipped += 1
            continue
        if best_seed is None or len(retained) < len(best_seed):
            best_seed = retained
            best_order = list(seed_order)

    if best_seed is None:
        raise BranchSearchLimitExceeded(
            "all {:,} candidate branch seeds exceeded the retained-table "
            "memory guard".format(len(seed_orders))
        )

    parents = best_seed
    order = best_order
    remaining = [trait for trait in range(num_traits) if trait not in order]
    level_diagnostics = []
    while remaining and len(parents):
        extension_count = sparse_branch_extension_count(parents)
        child_traits = len(order) + 1
        required_bytes = (
            extension_count
            * _sparse_table_bytes_per_row(intervals, child_traits)
        )
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
        best_stats = None
        extension_choices = candidate_extension_traits(
            omega,
            order,
            remaining,
            trial_limit=extension_trial_limit,
        )
        for candidate_trait in extension_choices:
            candidate_order = order + [candidate_trait]
            order_array = np.asarray(candidate_order, dtype=np.int64)
            candidate_omega = omega[np.ix_(order_array, order_array)]
            candidate_pi = (
                None
                if original_pi is None
                else original_pi[order_array]
            )
            children, stats = expand_sparse_branch_tables(
                parents,
                intervals,
                candidate_omega,
                prune_tolerance,
                pi_causal_ss=candidate_pi,
            )
            visited = stats[0]
            if visited != extension_count:
                raise RuntimeError("branch maxFDR extension count mismatch")
            if best_children is None or len(children) < len(best_children):
                best_children = children
                best_trait = candidate_trait
                best_stats = stats

        order.append(best_trait)
        remaining.remove(best_trait)
        parents = best_children
        level_diagnostics.append(
            {
                "trait": int(best_trait),
                "candidates_per_choice": int(extension_count),
                "choices_tested": int(len(extension_choices)),
                "retained": int(len(parents)),
                "fast_psd_accepts": int(best_stats[1]),
                "fast_psd_rejects": int(best_stats[2]),
                "eigen_fallbacks": int(best_stats[3]),
            }
        )

    if remaining:
        order.extend(remaining)
        parents = _empty_sparse_branch_tables(intervals, num_traits)

    order_array = np.asarray(order, dtype=np.int64)
    prepared_arrays = prepare_fdr_arrays(prepared)
    if original_pi is None:
        fit_ss = False
        final_pi = np.zeros(num_traits, dtype=np.float64)
    else:
        fit_ss = True
        final_pi = original_pi
    (
        ordered_max_fdr,
        ordered_best_ids,
        ordered_best_counts,
        ordered_best_occupied,
        feasible_count,
        invalid,
    ) = _evaluate_sparse_branch_leaves_kernel(
            parents.state_ids,
            parents.counts,
            parents.occupied,
            order_array,
            int(intervals),
            omega,
            *prepared_arrays,
            fit_ss,
            final_pi,
        )
    if invalid:
        raise ValueError(
            "maxFDR branch search returned a non-finite value or a value "
            "outside [0, 1]"
        )
    max_fdr = ordered_max_fdr
    best_ids = ordered_best_ids
    best_counts = ordered_best_counts
    best_occupied = ordered_best_occupied
    maximizing_probabilities = _materialize_sparse_probability_rows(
        best_ids,
        best_counts,
        best_occupied,
        intervals,
        num_states,
        table_byte_limit,
    )
    diagnostics = {
        "exhaustive_candidates": int(exhaustive_candidates),
        "trait_order": [int(trait) for trait in order],
        "seed_traits": int(seed_traits),
        "seed_candidates_per_subset": int(seed_candidates),
        "seed_subsets_available": int(
            math.comb(num_traits, seed_traits)
        ),
        "seed_subsets_considered": int(len(seed_orders)),
        "seed_subsets_tested": int(
            len(seed_orders)
            - seed_orders_skipped
            - seed_orders_uncompetitive
        ),
        "seed_subsets_skipped": int(seed_orders_skipped),
        "seed_subsets_short_circuited": int(seed_orders_uncompetitive),
        "seed_retained": int(len(best_seed)),
        "levels": level_diagnostics,
        "final_pruned_leaves": int(len(parents)),
        "representation": "sparse",
        "maximum_occupied_states": int(intervals),
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
