"""Regression and CLI tests for the maxFDR implementation."""

from argparse import Namespace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import mtag


ROOT = Path(__file__).resolve().parents[1]
OMEGA = np.array([[0.5, 0.2], [0.2, 0.4]])
SIGMA = np.array([[1.0, 0.1], [0.1, 1.0]])


def _fdr_args(output_prefix, **overrides):
    values = {
        "cores": 1,
        "fit_ss": False,
        "grid_file": None,
        "intervals": 2,
        "n_approx": True,
        "omega_hat": OMEGA,
        "out": str(output_prefix),
        "p_sig": 5.0e-8,
        "sigma_hat": SIGMA,
    }
    values.update(overrides)
    return Namespace(**values)


def test_simplex_and_compute_fdr_regression():
    grid = np.asarray(list(mtag.simplex_walk(3, 3)))
    assert grid.shape == (10, 4)
    np.testing.assert_allclose(np.sum(grid, axis=1), 1.0)
    assert set(np.unique(grid)) == {0.0, 0.5, 1.0}

    causal_states = mtag.create_S(2)
    value = mtag.compute_fdr(
        np.array([0.0, 0.5, 0.0, 0.5]),
        0,
        OMEGA,
        SIGMA,
        causal_states,
        np.array([[10_000.0, 12_000.0]]),
        np.ones(1),
        5.0e-8,
    )
    assert value == pytest.approx(5.227828076989659e-08, rel=1.0e-10)


def test_probability_grid_loader_accepts_one_row_and_excludes_invalid_rows(
    tmp_path,
):
    one_row = tmp_path / "one-row-grid.txt"
    np.savetxt(one_row, [[0.0, 0.5, 0.0, 0.5]])
    loaded = mtag.load_probability_grid(one_row, 4)
    assert loaded.shape == (1, 4)

    mixed = tmp_path / "mixed-grid.txt"
    np.savetxt(
        mixed,
        [
            [0.0, 0.5, 0.0, 0.5],
            [0.2, 0.2, 0.2, 0.2],
            [-0.1, 0.6, 0.0, 0.5],
        ],
    )
    loaded = mtag.load_probability_grid(mixed, 4)
    np.testing.assert_allclose(loaded, [[0.0, 0.5, 0.0, 0.5]])

    wrong_width = tmp_path / "wrong-width-grid.txt"
    np.savetxt(wrong_width, [[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="must contain 4 columns"):
        mtag.load_probability_grid(wrong_width, 4)


def test_automatic_probability_grid_is_filtered_while_streaming(
    tmp_path, monkeypatch
):
    state = {"yielded": None, "filtered": None}
    points = [
        np.array([0.0, 0.5, 0.0, 0.5]),
        np.array([0.5, 0.0, 0.0, 0.5]),
    ]

    def tracked_grid(*args):
        for index, point in enumerate(points):
            if index:
                assert state["filtered"] == index - 1
            state["yielded"] = index
            yield point

    def tracked_filter(probability, causal_states):
        state["filtered"] = state["yielded"]
        return True

    monkeypatch.setattr(mtag, "simplex_walk", tracked_grid)
    monkeypatch.setattr(mtag, "some_causal_for_allT", tracked_filter)
    monkeypatch.setattr(mtag, "is_pos_semidef", lambda matrix: True)
    monkeypatch.setattr(mtag, "compute_fdr", lambda *args: 0.25)

    fdr_matrix, probability_grid = mtag.fdr(
        _fdr_args(tmp_path / "streaming"),
        np.array([[10_000.0, 12_000.0]]),
        np.zeros((1, 2)),
    )

    assert state["filtered"] == len(points) - 1
    np.testing.assert_allclose(probability_grid, points)
    np.testing.assert_allclose(fdr_matrix, 0.25)


def test_exact_sample_sizes_and_fitted_spike_slab_paths(tmp_path):
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

    approximate_fdr, approximate_grid = mtag.fdr(
        _fdr_args(tmp_path / "approximate"), sample_sizes, z_scores
    )
    exact_fdr, exact_grid = mtag.fdr(
        _fdr_args(tmp_path / "exact", n_approx=False), sample_sizes, z_scores
    )
    np.testing.assert_allclose(exact_grid, approximate_grid)
    assert np.isfinite(exact_fdr).all()
    assert ((0.0 <= exact_fdr) & (exact_fdr <= 1.0)).all()
    assert np.max(np.abs(exact_fdr - approximate_fdr)) > 1.0e-12

    parallel_fdr, parallel_grid = mtag.fdr(
        _fdr_args(tmp_path / "parallel", n_approx=False, cores=2),
        sample_sizes,
        z_scores,
    )
    np.testing.assert_allclose(parallel_grid, exact_grid)
    np.testing.assert_allclose(parallel_fdr, exact_fdr)

    fitted_fdr, fitted_grid = mtag.fdr(
        _fdr_args(tmp_path / "fitted", fit_ss=True), sample_sizes, z_scores
    )
    assert len(fitted_grid) > 0
    assert np.isfinite(fitted_fdr).all()
    assert ((0.0 <= fitted_fdr) & (fitted_fdr <= 1.0)).all()


def test_spike_slab_optimizer_uses_inverse_parameter_transform(monkeypatch):
    captured = {}

    def fake_minimize(function, x_0, **kwargs):
        captured["x_0"] = np.asarray(x_0)
        return SimpleNamespace(x=np.asarray(x_0), success=True)

    monkeypatch.setattr(mtag.scipy.optimize, "minimize", fake_minimize)
    pi_null, tau = mtag._optim_ss(
        (
            np.array([0.0]),
            np.array([1.0]),
            (0.2, 0.03),
            {},
        )
    )
    np.testing.assert_allclose(captured["x_0"], [np.log(0.2 / 0.8), -np.log(0.03)])
    assert pi_null == pytest.approx(0.2)
    assert tau == pytest.approx(0.03)


def test_skip_mtag_cli_supports_single_row_grid_and_exact_sample_sizes(tmp_path):
    output_prefix = tmp_path / "prior-results"
    sample_sizes = (
        np.array([8_000.0, 10_000.0, 25_000.0, 10_000.0]),
        np.array([20_000.0, 12_000.0, 9_000.0, 12_000.0]),
    )
    z_scores = (
        np.array([-2.0, -0.5, 1.0, 2.5]),
        np.array([-1.5, 0.25, 1.25, 2.0]),
    )
    for trait in (1, 2):
        pd.DataFrame(
            {"N": sample_sizes[trait - 1], "Z": z_scores[trait - 1]}
        ).to_csv(
            tmp_path / f"prior-results_trait_{trait}.txt",
            sep="\t",
            index=False,
        )
    np.savetxt(tmp_path / "prior-results_omega_hat.txt", OMEGA)
    np.savetxt(tmp_path / "prior-results_sigma_hat.txt", SIGMA)
    grid_path = tmp_path / "one-row-grid.txt"
    np.savetxt(grid_path, [[0.0, 0.5, 0.0, 0.5]])

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "mtag.py"),
            "--skip_mtag",
            "--out",
            output_prefix.name,
            "--grid_file",
            str(grid_path),
            "--no-n-approx",
            "--intervals",
            "2",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    fdr_matrix = np.atleast_2d(np.loadtxt(tmp_path / "prior-results_fdr_mat.txt"))
    probability_grid = np.atleast_2d(
        np.loadtxt(tmp_path / "prior-results_prob_grid.txt")
    )
    assert fdr_matrix.shape == (1, 2)
    np.testing.assert_allclose(probability_grid, [[0.0, 0.5, 0.0, 0.5]])
    assert np.isfinite(fdr_matrix).all()


def test_maxfdr_rejects_invalid_configuration(tmp_path):
    sample_sizes = np.array([[10_000.0, 12_000.0]])
    z_scores = np.zeros_like(sample_sizes)
    with pytest.raises(ValueError, match="number of cores"):
        mtag.fdr(
            _fdr_args(tmp_path / "bad-cores", cores=0),
            sample_sizes,
            z_scores,
        )
    with pytest.raises(ValueError, match="significance threshold"):
        mtag.fdr(
            _fdr_args(tmp_path / "bad-p", p_sig=1.0),
            sample_sizes,
            z_scores,
        )
