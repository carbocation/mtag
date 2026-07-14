"""Parity tests for the optional fused Numba maxFDR engine."""

from argparse import Namespace

import numpy as np
import pytest


numba = pytest.importorskip("numba")

import mtag
import mtag_numba


OMEGA = np.array([[0.5, 0.2], [0.2, 0.4]])
SIGMA = np.array([[1.0, 0.1], [0.1, 1.0]])


def _args(output_prefix, backend, **overrides):
    values = {
        "cores": 1,
        "fdr_backend": backend,
        "fdr_chunk_size": 3,
        "fdr_write_full_grid": False,
        "fit_ss": False,
        "grid_file": None,
        "intervals": 2,
        "n_approx": True,
        "omega_hat": OMEGA.copy(),
        "out": str(output_prefix),
        "p_sig": 5.0e-8,
        "sigma_hat": SIGMA.copy(),
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture(autouse=True)
def one_numba_thread():
    previous = numba.get_num_threads()
    numba.set_num_threads(1)
    yield
    numba.set_num_threads(previous)


def test_numba_streaming_max_matches_python_backend(tmp_path):
    sample_sizes = np.array(
        [
            [8_000.0, 20_000.0],
            [10_000.0, 12_000.0],
            [25_000.0, 9_000.0],
            [10_000.0, 12_000.0],
        ]
    )
    z_scores = np.zeros_like(sample_sizes)

    expected_fdr, expected_grid = mtag.fdr(
        _args(tmp_path / "python", "python", n_approx=False),
        sample_sizes,
        z_scores,
    )
    actual_max, actual_probabilities = mtag.fdr(
        _args(tmp_path / "numba", "numba", n_approx=False),
        sample_sizes,
        z_scores,
    )

    expected_indices = np.argmax(expected_fdr, axis=0)
    expected_max = expected_fdr[expected_indices, np.arange(2)]
    expected_probabilities = expected_grid[expected_indices]
    np.testing.assert_allclose(
        actual_max, expected_max, rtol=1.0e-10, atol=1.0e-15
    )
    np.testing.assert_array_equal(
        actual_probabilities, expected_probabilities
    )
    np.testing.assert_allclose(
        np.atleast_1d(np.loadtxt(tmp_path / "numba_max_fdr.txt")),
        expected_max,
    )
    assert not (tmp_path / "numba_prob_grid.txt").exists()
    assert not (tmp_path / "numba_fdr_mat.txt").exists()
    assert (
        mtag_numba._evaluate_automatic_grid_max_chunk_sparse
        .nopython_signatures
    )


def test_numba_spike_slab_grid_restriction_matches_python(tmp_path):
    sample_sizes = np.array(
        [
            [8_000.0, 20_000.0],
            [10_000.0, 12_000.0],
            [25_000.0, 9_000.0],
            [10_000.0, 12_000.0],
        ]
    )
    z_scores = np.array(
        [
            [-2.0, -1.5],
            [-0.5, 0.25],
            [1.0, 1.25],
            [2.5, 2.0],
        ]
    )

    expected_fdr, expected_grid = mtag.fdr(
        _args(tmp_path / "python-fit", "python", fit_ss=True),
        sample_sizes,
        z_scores,
    )
    actual_max, actual_probabilities = mtag.fdr(
        _args(tmp_path / "numba-fit", "numba", fit_ss=True),
        sample_sizes,
        z_scores,
    )

    expected_indices = np.argmax(expected_fdr, axis=0)
    expected_max = expected_fdr[expected_indices, np.arange(2)]
    np.testing.assert_allclose(
        actual_max, expected_max, rtol=1.0e-10, atol=1.0e-15
    )
    np.testing.assert_array_equal(
        actual_probabilities, expected_grid[expected_indices]
    )


def test_numba_full_grid_output_remains_available(tmp_path):
    sample_sizes = np.array([[8_000.0, 20_000.0], [10_000.0, 12_000.0]])
    z_scores = np.zeros_like(sample_sizes)

    expected_fdr, expected_grid = mtag.fdr(
        _args(tmp_path / "python-full", "python"),
        sample_sizes,
        z_scores,
    )
    actual_fdr, actual_grid = mtag.fdr(
        _args(
            tmp_path / "numba-full",
            "numba",
            fdr_write_full_grid=True,
        ),
        sample_sizes,
        z_scores,
    )

    np.testing.assert_array_equal(actual_grid, expected_grid)
    np.testing.assert_allclose(
        actual_fdr, expected_fdr, rtol=1.0e-10, atol=1.0e-15
    )
    assert (tmp_path / "numba-full_prob_grid.txt").exists()
    assert (tmp_path / "numba-full_fdr_mat.txt").exists()
    assert (tmp_path / "numba-full_max_fdr.txt").exists()


def test_numba_backend_rejects_custom_probability_grid(tmp_path):
    grid_path = tmp_path / "grid.txt"
    np.savetxt(grid_path, [[0.0, 0.5, 0.0, 0.5]])
    with pytest.raises(ValueError, match="automatic grid only"):
        mtag.fdr(
            _args(tmp_path / "custom", "numba", grid_file=str(grid_path)),
            np.array([[10_000.0, 12_000.0]]),
            np.zeros((1, 2)),
        )


def test_six_trait_ten_interval_grid_supports_int64_ranks():
    intervals = 10
    num_states = 64
    total_points = mtag_numba.automatic_grid_size(num_states, intervals)
    assert total_points == 621_324_937_376

    combinations = mtag_numba.binomial_table(73, 63)
    assert combinations[73, 63] == total_points
    assert combinations[73, 36] == np.iinfo(np.int64).max

    counts = np.empty(num_states, dtype=np.int64)
    mtag_numba._unrank_composition(
        0, intervals, num_states, combinations, counts
    )
    np.testing.assert_array_equal(
        counts, np.r_[np.zeros(num_states - 1, dtype=int), intervals]
    )
    mtag_numba._unrank_composition(
        total_points - 1,
        intervals,
        num_states,
        combinations,
        counts,
    )
    np.testing.assert_array_equal(
        counts, np.r_[intervals, np.zeros(num_states - 1, dtype=int)]
    )


@pytest.mark.parametrize("num_states,intervals", [(4, 3), (8, 2)])
def test_sparse_unranking_matches_dense_grid_order(num_states, intervals):
    total_points = mtag_numba.automatic_grid_size(
        num_states, intervals
    )
    combinations = mtag_numba.binomial_table(
        intervals + num_states - 1,
        max(num_states - 1, intervals),
    )
    for rank in range(total_points):
        expected = np.empty(num_states, dtype=np.int64)
        mtag_numba._unrank_composition(
            rank, intervals, num_states, combinations, expected
        )
        state_ids = np.empty(intervals, dtype=np.int64)
        multiplicities = np.empty(intervals, dtype=np.int64)
        num_occupied = mtag_numba._unrank_sparse_composition(
            rank,
            total_points,
            intervals,
            num_states,
            combinations,
            state_ids,
            multiplicities,
        )
        actual = np.zeros(num_states, dtype=np.int64)
        actual[state_ids[:num_occupied]] = multiplicities[:num_occupied]
        np.testing.assert_array_equal(actual, expected)


def test_sparse_max_matches_dense_numba_grid_for_four_traits():
    causal_states = mtag.create_S(4)
    raw_omega = np.array(
        [
            [1.0, 0.2, 0.1, 0.05],
            [0.2, 0.9, 0.15, 0.1],
            [0.1, 0.15, 0.8, 0.2],
            [0.05, 0.1, 0.2, 0.7],
        ]
    )
    omega = raw_omega * 1.0e-5
    sigma = np.full((4, 4), 0.1)
    np.fill_diagonal(sigma, 1.0)
    sample_sizes = np.array([[80_000.0, 90_000.0, 100_000.0, 110_000.0]])
    prepared = mtag._prepare_fdr_calculation(
        omega, sigma, sample_sizes, np.ones(1), 5.0e-8
    )

    dense_grid, dense_fdr = mtag_numba.evaluate_automatic_grid(
        2,
        causal_states,
        omega,
        prepared,
        chunk_size=17,
    )
    actual_max, actual_probabilities, feasible_count = (
        mtag_numba.evaluate_automatic_grid_max(
            2,
            causal_states,
            omega,
            prepared,
            chunk_size=17,
        )
    )

    expected_indices = np.argmax(dense_fdr, axis=0)
    expected_max = dense_fdr[expected_indices, np.arange(4)]
    assert feasible_count == len(dense_grid)
    np.testing.assert_allclose(
        actual_max, expected_max, rtol=1.0e-10, atol=1.0e-15
    )
    np.testing.assert_array_equal(
        actual_probabilities, dense_grid[expected_indices]
    )


def test_six_trait_sparse_rank_ranges_match_dense_kernel():
    traits = 6
    intervals = 10
    causal_states = mtag.create_S(traits)
    rng = np.random.default_rng(8128)
    raw_omega = rng.normal(size=(traits, traits))
    omega = raw_omega @ raw_omega.T
    omega *= 2.0e-5 / np.mean(np.diag(omega))
    omega += np.eye(traits) * 1.0e-5
    sigma = np.full((traits, traits), 0.1)
    np.fill_diagonal(sigma, 1.0)
    sample_sizes = np.array(
        [[80_000.0, 90_000.0, 100_000.0, 110_000.0, 120_000.0, 130_000.0]]
    )
    prepared = mtag._prepare_fdr_calculation(
        omega, sigma, sample_sizes, np.ones(1), 5.0e-8
    )
    prepared_arrays = mtag_numba.prepare_fdr_arrays(prepared)
    state_arrays = mtag_numba.prepare_causal_state_arrays(causal_states)
    total_points = mtag_numba.automatic_grid_size(
        len(causal_states), intervals
    )
    combinations = mtag_numba.binomial_table(
        intervals + len(causal_states) - 1,
        max(len(causal_states) - 1, intervals),
    )
    pi_causal_ss = np.zeros(traits)

    for start_rank in (0, total_points // 2, total_points - 2_000):
        stop_rank = start_rank + 2_000
        _, dense_fdr, dense_feasible = (
            mtag_numba._evaluate_automatic_grid_chunk(
                start_rank,
                stop_rank,
                intervals,
                combinations,
                causal_states,
                omega,
                *prepared_arrays,
                False,
                pi_causal_ss,
            )
        )
        block_max, block_ranks, block_counts, invalid = (
            mtag_numba._evaluate_automatic_grid_max_chunk_sparse(
                start_rank,
                stop_rank,
                total_points,
                intervals,
                combinations,
                causal_states,
                *state_arrays,
                omega,
                *prepared_arrays,
                False,
                pi_causal_ss,
            )
        )

        assert not np.any(invalid)
        assert int(np.sum(block_counts)) == int(np.sum(dense_feasible))
        selected_rows = np.flatnonzero(dense_feasible)
        if not len(selected_rows):
            continue
        expected_indices = np.argmax(dense_fdr[selected_rows], axis=0)
        expected_ranks = start_rank + selected_rows[expected_indices]
        expected_max = dense_fdr[
            expected_ranks - start_rank, np.arange(traits)
        ]
        actual_max = np.full(traits, -np.inf)
        actual_ranks = np.full(traits, -1, dtype=np.int64)
        for block_index in range(len(block_counts)):
            for trait in range(traits):
                if block_max[block_index, trait] > actual_max[trait]:
                    actual_max[trait] = block_max[block_index, trait]
                    actual_ranks[trait] = block_ranks[block_index, trait]
        np.testing.assert_allclose(
            actual_max, expected_max, rtol=1.0e-10, atol=1.0e-15
        )
        np.testing.assert_array_equal(actual_ranks, expected_ranks)


def test_fast_psd_classifications_agree_with_reference():
    rng = np.random.default_rng(20260714)
    matrices = []
    for _ in range(100):
        raw = rng.normal(size=(6, 6))
        matrices.append(raw @ raw.T + np.eye(6) * 0.1)
        symmetric = raw + raw.T
        matrices.append(symmetric)
    matrices.extend(
        [
            np.diag([1.0, 1.0, 1.0, 1.0, 1.0, -1.0e-18]),
            np.ones((6, 6)),
        ]
    )

    for matrix in matrices:
        status = mtag_numba._fast_psd_status(matrix)
        if status < 0:
            assert not mtag.is_pos_semidef(matrix)
        elif status > 0:
            assert mtag.is_pos_semidef(matrix)


def test_numba_backend_cli_options_parse():
    parsed = mtag.parser.parse_args(
        [
            "--fdr-backend",
            "numba",
            "--fdr-chunk-size",
            "4096",
            "--fdr-write-full-grid",
        ]
    )
    assert parsed.fdr_backend == "numba"
    assert parsed.fdr_chunk_size == 4096
    assert parsed.fdr_write_full_grid
