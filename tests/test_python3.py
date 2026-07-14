"""Python 3 regression and command-line smoke tests for MTAG."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

import mtag


ROOT = Path(__file__).resolve().parents[1]


def test_mtag_analysis_regression_values():
    z_scores = np.array([[1.0, 2.0], [-1.0, 0.5]])
    sample_sizes = np.array([[100.0, 120.0], [90.0, 110.0]])
    omega = np.array([[0.4, 0.2], [0.2, 0.3]])
    sigma = np.array([[1.0, 0.1], [0.1, 1.2]])

    betas, standard_errors, factors = mtag.mtag_analysis(
        z_scores, sample_sizes, omega, sigma
    )

    np.testing.assert_allclose(
        betas,
        [[0.10256087320314272, 0.18212776624960031],
         [-0.10326158356704489, 0.044616197809091596]],
    )
    np.testing.assert_allclose(
        standard_errors,
        [[0.099604472036958672, 0.099406835258034329],
         [0.10494640879756341, 0.10377587721202022]],
    )
    np.testing.assert_allclose(
        factors,
        [[1.0096582648688821, 1.0068523521618555],
         [1.0106979466032606, 1.0074274163769767]],
    )


def _write_sumstats(path, z_scores, sample_size):
    allele_pairs = [
        ("A", "G"),
        ("C", "A"),
        ("G", "A"),
        ("T", "C"),
        ("A", "C"),
        ("C", "T"),
        ("G", "T"),
        ("T", "G"),
    ]
    rows = len(z_scores)
    data = pd.DataFrame(
        {
            "snpid": [f"rs{i + 1}" for i in range(rows)],
            "chr": np.ones(rows, dtype=int),
            "bpos": np.arange(101, 101 + rows),
            "a1": [allele_pairs[i % len(allele_pairs)][0] for i in range(rows)],
            "a2": [allele_pairs[i % len(allele_pairs)][1] for i in range(rows)],
            "freq": np.linspace(0.2, 0.39, rows),
            "z": z_scores,
            "p": 2 * norm.sf(np.abs(z_scores)),
            "n": np.full(rows, sample_size),
        }
    )
    data.to_csv(path, sep=" ", index=False)


def test_python3_cli_end_to_end_with_supplied_covariances(tmp_path):
    trait_1 = tmp_path / "trait_1.txt"
    trait_2 = tmp_path / "trait_2.txt"
    z_1 = np.linspace(-2.0, 1.8, 20)
    z_2 = np.linspace(-1.5, 2.3, 20)
    _write_sumstats(trait_1, z_1, 10_000)
    _write_sumstats(trait_2, z_2, 12_000)

    omega_path = tmp_path / "omega.txt"
    sigma_path = tmp_path / "sigma.txt"
    np.savetxt(omega_path, [[0.5, 0.2], [0.2, 0.4]])
    np.savetxt(sigma_path, [[1.0, 0.1], [0.1, 1.0]])
    output_prefix = tmp_path / "results"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "mtag.py"),
            "--sumstats",
            f"{trait_1},{trait_2}",
            "--gencov_path",
            str(omega_path),
            "--residcov_path",
            str(sigma_path),
            "--out",
            str(output_prefix),
            "--force",
            "--median_z_cutoff",
            "999",
            "--n_min",
            "0",
            "--maf_min",
            "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    for trait_number in (1, 2):
        output_path = tmp_path / f"results_trait_{trait_number}.txt"
        assert output_path.exists()
        output = pd.read_csv(output_path, sep=r"\s+")
        assert len(output) == 20
        assert {
            "SNP",
            "CHR",
            "BP",
            "A1",
            "A2",
            "Z",
            "N",
            "FRQ",
            "mtag_beta",
            "mtag_se",
            "mtag_z",
            "mtag_pval",
        } == set(output.columns)
        assert np.isfinite(
            output[["mtag_beta", "mtag_se", "mtag_z", "mtag_pval"]]
        ).all().all()
        assert (output["mtag_se"] > 0).all()
        assert output["mtag_pval"].between(0, 1).all()

    assert (tmp_path / "results_omega_hat.txt").exists()
    assert (tmp_path / "results_sigma_hat.txt").exists()
    assert (tmp_path / "results.log").exists()


def test_python3_cli_estimates_covariances_from_bundled_ld_scores(tmp_path):
    ld_scores = pd.read_csv(
        ROOT / "ld_ref_panel/eur_w_ld_chr/1.l2.ldscore.gz",
        sep=r"\s+",
        nrows=1_000,
    )
    hm3_alleles = pd.read_csv(
        ROOT / "ld_ref_panel/eur_w_ld_chr/w_hm3.snplist",
        sep=r"\s+",
        nrows=1_000,
    )
    variants = ld_scores.merge(hm3_alleles, on="SNP").head(300)
    assert len(variants) == 300

    rng = np.random.default_rng(20260713)
    z_1 = rng.normal(size=len(variants)) * 1.3
    z_2 = 0.35 * z_1 + rng.normal(size=len(variants)) * 1.2
    for number, z_scores, sample_size in (
        (1, z_1, 100_000),
        (2, z_2, 120_000),
    ):
        sumstats = pd.DataFrame(
            {
                "snpid": variants["SNP"],
                "chr": variants["CHR"],
                "bpos": variants["BP"],
                "a1": variants["A1"],
                "a2": variants["A2"],
                "freq": variants["MAF"],
                "z": z_scores,
                "p": 2 * norm.sf(np.abs(z_scores)),
                "n": sample_size,
            }
        )
        sumstats.to_csv(tmp_path / f"input_trait_{number}.txt", sep=" ", index=False)

    output_prefix = tmp_path / "estimated"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "mtag.py"),
            "--sumstats",
            f"{tmp_path / 'input_trait_1.txt'},{tmp_path / 'input_trait_2.txt'}",
            "--out",
            str(output_prefix),
            "--force",
            "--median_z_cutoff",
            "999",
            "--n_min",
            "0",
            "--maf_min",
            "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    omega = np.loadtxt(tmp_path / "estimated_omega_hat.txt")
    sigma = np.loadtxt(tmp_path / "estimated_sigma_hat.txt")
    assert omega.shape == sigma.shape == (2, 2)
    assert np.isfinite(omega).all()
    assert np.isfinite(sigma).all()
    for trait_number in (1, 2):
        output = pd.read_csv(
            tmp_path / f"estimated_trait_{trait_number}.txt", sep=r"\s+"
        )
        assert len(output) == 300
        assert np.isfinite(output["mtag_z"]).all()


def test_python3_command_line_entry_points_show_help():
    commands = [
        [sys.executable, str(ROOT / "mtag.py"), "--help"],
        [sys.executable, str(ROOT / "mtag_munge.py"), "--help"],
        [sys.executable, "-m", "ldsc_mod.ldsc", "--help"],
    ]
    for command in commands:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "usage:" in result.stdout.lower()
