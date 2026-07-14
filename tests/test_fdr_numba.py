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
    assert mtag_numba._evaluate_automatic_grid_chunk.nopython_signatures


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
