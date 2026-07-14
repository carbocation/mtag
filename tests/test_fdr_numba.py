"""Parity tests for the default fused Numba maxFDR engine."""

from argparse import Namespace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


numba = pytest.importorskip("numba")

import mtag
import mtag_numba


ROOT = Path(__file__).resolve().parents[1]
OMEGA = np.array([[0.5, 0.2], [0.2, 0.4]])
SIGMA = np.array([[1.0, 0.1], [0.1, 1.0]])


def _args(output_prefix, backend, **overrides):
    values = {
        "cores": 1,
        "fdr_backend": backend,
        "fdr_chunk_size": 3,
        "fdr_search": "auto",
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


def test_branch_search_rejects_full_grid_output(tmp_path):
    with pytest.raises(ValueError, match="max-only output"):
        mtag.fdr(
            _args(
                tmp_path / "branch-full",
                "numba",
                fdr_search="branch",
                fdr_write_full_grid=True,
            ),
            np.array([[10_000.0, 12_000.0]]),
            np.zeros((1, 2)),
        )


def test_python_backend_rejects_numba_search_selection(tmp_path):
    with pytest.raises(ValueError, match="applies only"):
        mtag.fdr(
            _args(
                tmp_path / "python-branch",
                "python",
                fdr_search="branch",
            ),
            np.array([[10_000.0, 12_000.0]]),
            np.zeros((1, 2)),
        )


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


@pytest.mark.parametrize("num_states,intervals", [(2, 3), (4, 3), (8, 2)])
def test_count_comparison_matches_historical_grid_order(
    num_states, intervals
):
    total_points = mtag_numba.automatic_grid_size(num_states, intervals)
    combinations = mtag_numba.binomial_table(
        intervals + num_states - 1,
        max(num_states - 1, intervals),
    )
    counts = np.empty(num_states, dtype=np.int64)
    previous = None

    for rank in range(total_points):
        mtag_numba._unrank_composition(
            rank, intervals, num_states, combinations, counts
        )
        if previous is not None:
            previous_ids = np.flatnonzero(previous).astype(np.uint64)
            current_ids = np.flatnonzero(counts).astype(np.uint64)
            previous_counts = previous[previous_ids.astype(int)]
            current_counts = counts[current_ids.astype(int)]
            assert mtag_numba._sparse_count_vector_precedes(
                previous_ids,
                previous_counts,
                len(previous_ids),
                current_ids,
                current_counts,
                len(current_ids),
            )
            assert not mtag_numba._sparse_count_vector_precedes(
                current_ids,
                current_counts,
                len(current_ids),
                previous_ids,
                previous_counts,
                len(previous_ids),
            )
        previous = counts.copy()


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


@pytest.mark.parametrize("traits,intervals,seed", [(4, 3, 314), (5, 3, 159)])
def test_exact_branch_search_matches_exhaustive_numba(
    traits, intervals, seed
):
    causal_states = mtag.create_S(traits)
    rng = np.random.default_rng(seed)
    raw_omega = rng.normal(size=(traits, traits))
    omega = raw_omega @ raw_omega.T
    omega *= 2.0e-5 / np.mean(np.diag(omega))
    omega += np.eye(traits) * 1.0e-5
    sigma = np.full((traits, traits), 0.05)
    np.fill_diagonal(sigma, 1.0)
    sample_sizes = np.array(
        [np.linspace(80_000.0, 130_000.0, traits)]
    )
    prepared = mtag._prepare_fdr_calculation(
        omega, sigma, sample_sizes, np.ones(1), 5.0e-8
    )

    expected_max, expected_probabilities, expected_count = (
        mtag_numba.evaluate_automatic_grid_max(
            intervals,
            causal_states,
            omega,
            prepared,
            chunk_size=17,
        )
    )
    actual_max, actual_probabilities, actual_count, diagnostics = (
        mtag_numba.evaluate_automatic_grid_max_branch(
            intervals, causal_states, omega, prepared
        )
    )

    assert diagnostics["exhaustive_candidates"] == (
        mtag_numba.automatic_grid_size(len(causal_states), intervals)
    )
    assert diagnostics["representation"] == "sparse"
    assert diagnostics["traversal"] == "depth_first"
    assert diagnostics["frontier_rows_materialized"] == diagnostics[
        "seed_retained"
    ]
    assert actual_count == expected_count
    np.testing.assert_array_equal(actual_max, expected_max)
    np.testing.assert_array_equal(
        actual_probabilities, expected_probabilities
    )


def test_deterministic_subtree_shards_match_single_worker_exactly():
    traits = 5
    intervals = 3
    omega = np.eye(traits) * 1.0e-5
    sigma = np.eye(traits)
    sample_sizes = np.full((1, traits), 100_000.0)
    prepared = mtag._prepare_fdr_calculation(
        omega, sigma, sample_sizes, np.ones(1), 5.0e-8
    )

    expected = mtag_numba.evaluate_automatic_grid_max_branch(
        intervals, traits, omega, prepared, workers=1
    )
    expected_levels = [
        (level["candidates_per_choice"], level["retained"])
        for level in expected[3]["levels"]
    ]

    for workers in (2, 4):
        actual = mtag_numba.evaluate_automatic_grid_max_branch(
            intervals, traits, omega, prepared, workers=workers
        )

        assert actual[2] == expected[2]
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert actual[3]["traversal"] == "depth_first_sharded"
        assert actual[3]["subtree_workers_requested"] == workers
        assert 1 < actual[3]["subtree_workers_used"] <= workers
        assert actual[3]["subtree_shards"] >= workers
        assert [
            (level["candidates_per_choice"], level["retained"])
            for level in actual[3]["levels"]
        ] == expected_levels


def test_subtree_worker_exception_is_propagated(monkeypatch):
    frontier = mtag_numba.SparseBranchTables(
        np.zeros((2, 1), dtype=np.uint64),
        np.ones((2, 1), dtype=np.int64),
        np.ones(2, dtype=np.int64),
        np.ones((2, 1), dtype=np.int64),
        np.ones((2, 1, 1), dtype=np.float64),
        np.ones(2, dtype=np.uint8),
    )

    def fail_nonempty(shard, _traits, _arguments):
        if len(shard) == 0:
            return None
        raise RuntimeError("shard failed")

    monkeypatch.setattr(
        mtag_numba,
        "_evaluate_sparse_branch_frontier",
        fail_nonempty,
    )
    with pytest.raises(RuntimeError, match="shard failed"):
        mtag_numba._run_sparse_branch_shards(
            frontier, 1, (), workers=2
        )


def test_simd_branch_power_layout_matches_multiple_n_bins():
    traits = 4
    intervals = 3
    rng = np.random.default_rng(8675309)
    raw_omega = rng.normal(size=(traits, traits))
    omega = raw_omega @ raw_omega.T + np.eye(traits)
    omega *= 2.0e-5 / np.mean(np.diag(omega))
    sigma = np.full((traits, traits), 0.05)
    np.fill_diagonal(sigma, 1.0)
    sample_sizes = np.array(
        [
            [75_000.0, 85_000.0, 95_000.0, 105_000.0],
            [90_000.0, 100_000.0, 110_000.0, 120_000.0],
        ]
    )
    prepared = mtag._prepare_fdr_calculation(
        omega, sigma, sample_sizes, np.array([3.0, 2.0]), 5.0e-8
    )
    causal_states = mtag.create_S(traits)

    expected = mtag_numba.evaluate_automatic_grid_max(
        intervals,
        causal_states,
        omega,
        prepared,
        chunk_size=17,
    )
    actual = mtag_numba.evaluate_automatic_grid_max_branch(
        intervals, traits, omega, prepared
    )

    assert actual[2] == expected[2]
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_verified_branch_caches_preserve_results_under_collisions():
    traits = 5
    intervals = 5
    omega = np.eye(traits) * 1.0e-5
    sigma = np.eye(traits)
    sample_sizes = np.full((1, traits), 100_000.0)
    prepared = mtag._prepare_fdr_calculation(
        omega, sigma, sample_sizes, np.ones(1), 5.0e-8
    )

    seed_candidate_limit = mtag_numba.automatic_grid_size(8, intervals)
    cached = mtag_numba.evaluate_automatic_grid_max_branch(
        intervals,
        traits,
        omega,
        prepared,
        candidate_limit=seed_candidate_limit,
        pair_cache_size=4096,
        power_cache_size=65536,
    )
    collision_heavy = mtag_numba.evaluate_automatic_grid_max_branch(
        intervals,
        traits,
        omega,
        prepared,
        candidate_limit=seed_candidate_limit,
        pair_cache_size=1,
        power_cache_size=1,
    )

    assert cached[3]["pair_signature_cache_hits"] > 0
    assert cached[3]["state_power_cache_hits"] > 0
    assert max(
        level["candidates_per_choice"] for level in cached[3]["levels"]
    ) > seed_candidate_limit
    assert collision_heavy[3]["pair_signature_cache_size"] == 1
    assert collision_heavy[3]["state_power_cache_size"] == 1
    assert collision_heavy[2] == cached[2]
    np.testing.assert_array_equal(collision_heavy[0], cached[0])
    np.testing.assert_array_equal(collision_heavy[1], cached[1])


def test_sparse_branch_spike_slab_restriction_matches_exhaustive():
    traits = 5
    intervals = 3
    rng = np.random.default_rng(2718)
    raw_omega = rng.normal(size=(traits, traits))
    omega = raw_omega @ raw_omega.T + np.eye(traits)
    omega *= 2.0e-5 / np.mean(np.diag(omega))
    sigma = np.eye(traits)
    sample_sizes = np.full((1, traits), 100_000.0)
    prepared = mtag._prepare_fdr_calculation(
        omega, sigma, sample_sizes, np.ones(1), 5.0e-8
    )
    pi_causal = np.full(traits, 0.5)

    expected = mtag_numba.evaluate_automatic_grid_max(
        intervals,
        mtag.create_S(traits),
        omega,
        prepared,
        pi_causal_ss=pi_causal,
        chunk_size=1000,
    )
    actual = mtag_numba.evaluate_automatic_grid_max_branch(
        intervals,
        traits,
        omega,
        prepared,
        pi_causal_ss=pi_causal,
        workers=3,
    )

    assert actual[2] == expected[2]
    assert actual[3]["subtree_workers_used"] > 1
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_real_five_trait_branch_regression_matches_historical_run():
    omega = np.array(
        [
            [
                3.875220595894177e-06,
                1.5020955496521807e-06,
                -2.494193026680934e-06,
                1.0109516024171428e-07,
                1.5842123683951421e-06,
            ],
            [
                1.5020955496521807e-06,
                4.020099400582166e-06,
                -1.2586329813655462e-06,
                4.932638357576401e-07,
                1.3957082703225266e-06,
            ],
            [
                -2.494193026680934e-06,
                -1.2586329813655462e-06,
                4.26107618042203e-06,
                -1.8029895784903273e-07,
                -1.949985504003797e-06,
            ],
            [
                1.0109516024171428e-07,
                4.932638357576401e-07,
                -1.8029895784903273e-07,
                1.084780947488981e-06,
                1.2027420559655031e-06,
            ],
            [
                1.5842123683951421e-06,
                1.3957082703225266e-06,
                -1.949985504003797e-06,
                1.2027420559655031e-06,
                2.167377501054478e-06,
            ],
        ]
    )
    sigma = np.array(
        [
            [
                1.0056843864584488,
                0.3573467020556312,
                -0.6191929149077187,
                0.0085777296016904,
                -0.003160735055182767,
            ],
            [
                0.3573467020556312,
                1.0249187640689477,
                -0.24200411068335428,
                0.007531040089317779,
                -8.288716959050474e-06,
            ],
            [
                -0.6191929149077187,
                -0.24200411068335428,
                0.9960809804692066,
                -0.012647249724174673,
                -0.012601533252360603,
            ],
            [
                0.0085777296016904,
                0.007531040089317779,
                -0.012647249724174673,
                0.9776546122748961,
                0.10790627935366807,
            ],
            [
                -0.003160735055182767,
                -8.288716959050474e-06,
                -0.012601533252360603,
                0.10790627935366807,
                1.0146291642511813,
            ],
        ]
    )
    sample_sizes = np.array(
        [
            [
                78_581.0,
                78_581.0,
                78_581.0,
                218_704.4102766652,
                48_437.15624649526,
            ]
        ]
    )
    prepared = mtag._prepare_fdr_calculation(
        omega, sigma, sample_sizes, np.ones(1), 5.0e-8
    )

    max_fdr, probabilities, feasible_count, diagnostics = (
        mtag_numba.evaluate_automatic_grid_max_branch(
            10, mtag.create_S(5), omega, prepared
        )
    )
    expected_max = np.array(
        [
            0.0027121832941844446,
            0.003013923520981947,
            0.00177892174753345,
            0.00394098847348618,
            0.0019085208967734293,
        ]
    )
    expected_probabilities = np.zeros((5, 32))
    expected_probabilities[:, 8] = 0.3
    expected_probabilities[:, 31] = 0.7
    expected_probabilities[1, 8] = 0.0
    expected_probabilities[1, 23] = 0.3

    assert feasible_count == 48
    assert diagnostics["exhaustive_candidates"] == 1_121_099_408
    assert diagnostics["final_pruned_leaves"] == 48
    assert diagnostics["trait_order"] != list(range(5))
    assert all(
        level["fast_psd_accepts"] + level["fast_psd_rejects"] > 0
        for level in diagnostics["levels"]
    )
    np.testing.assert_array_equal(max_fdr, expected_max)
    np.testing.assert_array_equal(probabilities, expected_probabilities)

    refined_max, _, refined_count, refined_diagnostics = (
        mtag_numba.evaluate_automatic_grid_max_branch(
            20, 5, omega, prepared
        )
    )
    expected_refined_max = np.array(
        [
            0.0028095775895184047,
            0.003013923520981947,
            0.0017800904937819437,
            0.00394098847348618,
            0.0019085208967734293,
        ]
    )
    assert refined_count == 437
    assert refined_diagnostics["exhaustive_candidates"] == 77_535_155_627_160
    assert refined_diagnostics["final_pruned_leaves"] == 437
    assert refined_diagnostics["seed_traits"] == 2
    assert refined_diagnostics["traversal"] == "depth_first"
    np.testing.assert_array_equal(refined_max, expected_refined_max)
    assert np.all(refined_max >= max_fdr)

    omega_six = np.zeros((6, 6))
    omega_six[:5, :5] = omega
    omega_six[5, 5] = np.mean(np.diag(omega))
    sigma_six = np.zeros((6, 6))
    sigma_six[:5, :5] = sigma
    sigma_six[5, 5] = 1.0
    sample_sizes_six = np.c_[sample_sizes, 100_000.0]
    prepared_six = mtag._prepare_fdr_calculation(
        omega_six,
        sigma_six,
        sample_sizes_six,
        np.ones(1),
        5.0e-8,
    )
    six_max, _, six_count, six_diagnostics = (
        mtag_numba.evaluate_automatic_grid_max_branch(
            10, 6, omega_six, prepared_six
        )
    )
    assert np.all(np.isfinite(six_max))
    assert six_count == 1_966
    assert six_diagnostics["exhaustive_candidates"] == 621_324_937_376
    assert six_diagnostics["final_pruned_leaves"] == 1_966


def test_branch_path_does_not_materialize_all_binary_states(
    tmp_path, monkeypatch
):
    traits = 5
    omega = np.eye(traits) * 1.0e-5
    sigma = np.eye(traits)
    sample_sizes = np.full((1, traits), 100_000.0)

    def unexpected_dense_states(_):
        raise AssertionError("dense causal state table was materialized")

    monkeypatch.setattr(mtag, "create_S", unexpected_dense_states)
    max_fdr, probabilities = mtag.fdr(
        _args(
            tmp_path / "sparse-branch",
            "numba",
            fdr_search="branch",
            intervals=2,
            omega_hat=omega,
            sigma_hat=sigma,
        ),
        sample_sizes,
        np.zeros_like(sample_sizes),
    )

    assert np.all(np.isfinite(max_fdr))
    assert probabilities.shape == (traits, 1 << traits)


def test_large_auto_grid_does_not_silently_fall_back_to_exhaustive(
    tmp_path, monkeypatch
):
    traits = 5

    def guarded_branch(*args, **kwargs):
        raise mtag_numba.BranchSearchLimitExceeded("test guard")

    monkeypatch.setattr(
        mtag_numba,
        "evaluate_automatic_grid_max_branch",
        guarded_branch,
    )
    with pytest.raises(ValueError, match="fallback is disabled"):
        mtag.fdr(
            _args(
                tmp_path / "guarded-auto",
                "numba",
                intervals=10,
                omega_hat=np.eye(traits) * 1.0e-5,
                sigma_hat=np.eye(traits),
            ),
            np.full((1, traits), 100_000.0),
            np.zeros((1, traits)),
        )


def test_trait_order_trials_and_sparse_memory_are_bounded():
    traits = 20
    omega = np.eye(traits)
    omega += 0.05 * (np.ones((traits, traits)) - np.eye(traits))

    seed_orders = mtag_numba.candidate_seed_orders(
        omega, trial_limit=12
    )
    extension_traits = mtag_numba.candidate_extension_traits(
        omega,
        order=[0, 1, 2],
        remaining=list(range(3, traits)),
        trial_limit=7,
    )

    assert len(seed_orders) <= 12
    assert len(set(seed_orders)) == len(seed_orders)
    assert all(len(set(order)) == 3 for order in seed_orders)
    assert len(extension_traits) == 7
    assert mtag_numba._sparse_table_bytes_per_row(10, traits) < 6_000
    assert (1 << traits) * 8 > 8_000_000


def test_sparse_maximizing_probability_rows_materialize_on_demand():
    state_ids = np.array([[3, 7, 0], [1, 6, 0]], dtype=np.uint64)
    counts = np.array([[4, 6, 0], [2, 8, 0]], dtype=np.int64)
    occupied = np.array([2, 2], dtype=np.int64)
    probabilities = mtag_numba._materialize_sparse_probability_rows(
        state_ids,
        counts,
        occupied,
        intervals=10,
        num_states=8,
        byte_limit=1,
    )

    assert isinstance(probabilities, mtag_numba.SparseProbabilityRows)
    assert probabilities.shape == (2, 8)
    expected = np.zeros((2, 8))
    expected[0, [3, 7]] = [0.4, 0.6]
    expected[1, [1, 6]] = [0.2, 0.8]
    np.testing.assert_array_equal(np.asarray(probabilities), expected)
    assert "3" in probabilities.format_row(0)


def test_large_nominal_grid_size_does_not_require_int64_rank():
    total = mtag_numba.nominal_grid_size(1 << 9, 10)
    assert total > np.iinfo(np.int64).max
    with pytest.raises(OverflowError):
        mtag_numba.automatic_grid_size(1 << 9, 10)


def test_bordered_psd_reuse_has_conservative_boundary_fallbacks():
    intervals = 10
    tolerance = 1.0e-12
    pair_counts = np.full(6, intervals, dtype=np.int64)
    parent_factor = np.eye(2)
    child_factor = np.empty((3, 3))

    positive = np.eye(3)
    valid, factor_valid, status = (
        mtag_numba._bordered_principal_psd_check(
            parent_factor,
            True,
            pair_counts,
            intervals,
            positive,
            tolerance,
            child_factor,
        )
    )
    assert valid and factor_valid and status == 1

    near_boundary = np.diag([1.0, 1.0, -0.5 * tolerance])
    valid, factor_valid, status = (
        mtag_numba._bordered_principal_psd_check(
            parent_factor,
            True,
            pair_counts,
            intervals,
            near_boundary,
            tolerance,
            child_factor,
        )
    )
    assert valid and not factor_valid and status == 0

    clearly_indefinite = np.diag([1.0, 1.0, -32.0 * tolerance])
    valid, factor_valid, status = (
        mtag_numba._bordered_principal_psd_check(
            parent_factor,
            True,
            pair_counts,
            intervals,
            clearly_indefinite,
            tolerance,
            child_factor,
        )
    )
    assert not valid and not factor_valid and status == -1


def test_mixed_radix_split_updates_pair_counts_incrementally():
    parent_traits = 3
    parent_ids = np.array([1, 3, 6], dtype=np.uint64)
    parent_counts = np.array([2, 1, 2], dtype=np.int64)
    split_counts = np.zeros(3, dtype=np.int64)
    split_pair_counts = np.zeros(4, dtype=np.int64)
    observed = 0

    while True:
        expected = np.zeros(4, dtype=np.int64)
        for state, count in zip(parent_ids, split_counts):
            for trait in range(parent_traits):
                if mtag_numba._sparse_state_has_trait(
                    state, trait, parent_traits
                ):
                    expected[trait] += count
            expected[parent_traits] += count
        np.testing.assert_array_equal(split_pair_counts, expected)
        observed += 1
        if not mtag_numba._advance_sparse_split_with_pair_counts(
            split_counts,
            parent_ids,
            parent_counts,
            len(parent_counts),
            parent_traits,
            split_pair_counts,
        ):
            break

    assert observed == np.prod(parent_counts + 1)
    np.testing.assert_array_equal(split_counts, np.zeros(3, dtype=int))
    np.testing.assert_array_equal(
        split_pair_counts, np.zeros(4, dtype=int)
    )


def test_reordered_pair_counts_map_to_original_trait_order():
    trait_order = np.array([2, 0, 3, 1], dtype=np.int64)
    reordered_ids = np.array([1, 6, 11, 15], dtype=np.uint64)
    counts = np.array([2, 1, 3, 4], dtype=np.int64)
    reordered_pairs = np.empty(10, dtype=np.int64)
    original_pairs = np.empty(10, dtype=np.int64)
    expected = np.empty(10, dtype=np.int64)

    mtag_numba._seed_pair_counts(
        reordered_ids, counts, len(counts), 4, reordered_pairs
    )
    mtag_numba._map_pair_counts_to_original(
        reordered_pairs, trait_order, original_pairs
    )
    original_ids = np.array(
        [
            mtag_numba._map_sparse_state_to_original(
                state, trait_order
            )
            for state in reordered_ids
        ],
        dtype=np.uint64,
    )
    mtag_numba._seed_pair_counts(
        original_ids, counts, len(counts), 4, expected
    )

    np.testing.assert_array_equal(original_pairs, expected)


def test_branch_search_memory_guard_is_explicit():
    traits = 4
    causal_states = mtag.create_S(traits)
    omega = np.eye(traits) * 1.0e-5
    sigma = np.eye(traits)
    sample_sizes = np.full((1, traits), 100_000.0)
    prepared = mtag._prepare_fdr_calculation(
        omega, sigma, sample_sizes, np.ones(1), 5.0e-8
    )

    with pytest.raises(mtag_numba.BranchSearchLimitExceeded):
        mtag_numba.evaluate_automatic_grid_max_branch(
            2,
            causal_states,
            omega,
            prepared,
            candidate_limit=1,
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
            "--fdr-search",
            "branch",
            "--fdr-write-full-grid",
        ]
    )
    assert parsed.fdr_backend == "numba"
    assert parsed.fdr_chunk_size == 4096
    assert parsed.fdr_search == "branch"
    assert parsed.fdr_write_full_grid


def test_optimized_cli_paths_are_defaults_and_legacy_is_explicit():
    defaults = mtag.parser.parse_args([])
    assert defaults.load_backend == "polars"
    assert defaults.output_backend == "polars"
    assert defaults.fdr_backend == "numba"
    assert defaults.fdr_search == "auto"
    assert defaults.n_approx
    assert defaults.cores is None
    assert not defaults.fdr_write_full_grid
    assert not defaults.legacy_loader

    compatibility = mtag.parser.parse_args(
        ["--legacy-loader", "--fdr-backend", "python"]
    )
    assert compatibility.legacy_loader
    assert compatibility.fdr_backend == "python"

    assert mtag.parser.parse_args(["--cores", "auto"]).cores is None
    assert mtag.parser.parse_args(["--cores", "3"]).cores == 3


def test_automatic_core_selection_respects_allocation_limits(
    monkeypatch,
):
    for variable in mtag.CPU_ALLOCATION_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(mtag.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(
        mtag.os,
        "sched_getaffinity",
        lambda _pid: set(range(8)),
        raising=False,
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    monkeypatch.setenv("NUMBA_NUM_THREADS", "4")

    assert mtag._available_cpu_workers() == 4
    assert mtag._resolve_fdr_cores(None) == 4
    assert mtag._resolve_fdr_cores(3) == 3
    assert mtag._resolve_fdr_cores(20) == 4


def test_default_skip_mtag_uses_exact_numba_auto_search(tmp_path):
    traits = 5
    output_prefix = tmp_path / "default-fast"
    mtag._write_maxfdr_inputs(
        Namespace(out=str(output_prefix)),
        np.full((1, traits), 100_000.0),
    )
    np.savetxt(
        tmp_path / "default-fast_omega_hat.txt",
        np.eye(traits) * 1.0e-5,
    )
    np.savetxt(
        tmp_path / "default-fast_sigma_hat.txt", np.eye(traits)
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "mtag.py"),
            "--skip_mtag",
            "--out",
            str(output_prefix),
            "--intervals",
            "2",
            "--cores",
            "2",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "default-fast_max_fdr.txt").exists()
    assert not (tmp_path / "default-fast_prob_grid.txt").exists()
    assert not (tmp_path / "default-fast_fdr_mat.txt").exists()
    log = (tmp_path / "default-fast.FDR.log").read_text()
    assert "Exact branch-and-prune maxFDR search" in log
    assert "Deterministic subtree traversal used 2 workers" in log
