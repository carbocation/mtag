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


def test_numba_automatic_grid_matches_python_backend(tmp_path):
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
    actual_fdr, actual_grid = mtag.fdr(
        _args(tmp_path / "numba", "numba", n_approx=False),
        sample_sizes,
        z_scores,
    )

    np.testing.assert_array_equal(actual_grid, expected_grid)
    np.testing.assert_allclose(
        actual_fdr, expected_fdr, rtol=1.0e-10, atol=1.0e-15
    )
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
    actual_fdr, actual_grid = mtag.fdr(
        _args(tmp_path / "numba-fit", "numba", fit_ss=True),
        sample_sizes,
        z_scores,
    )

    np.testing.assert_array_equal(actual_grid, expected_grid)
    np.testing.assert_allclose(
        actual_fdr, expected_fdr, rtol=1.0e-10, atol=1.0e-15
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


def test_numba_backend_cli_options_parse():
    parsed = mtag.parser.parse_args(
        ["--fdr-backend", "numba", "--fdr-chunk-size", "4096"]
    )
    assert parsed.fdr_backend == "numba"
    assert parsed.fdr_chunk_size == 4096
