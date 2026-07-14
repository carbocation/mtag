#!/usr/bin/env python3
"""Generate deterministic LD Score and summary-statistic test fixtures."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


N_INDIV = 10_000
N_SIMS = 1_000
N_SNP = 1_000
H2_1 = 0.3
H2_2 = 0.6
DEFAULT_SEED = 20_260_713


def _write_values(path, values):
    with path.open("w") as output:
        print("\t".join(map(str, values)), file=output)


def _write_ld_scores(frame, prefix, m_values):
    ldscore_suffix = ".l2.ldscore"
    m_suffix = ".l2.M_5_50"
    frame.to_csv(
        f"{prefix}{ldscore_suffix}", sep="\t", index=False, float_format="%.3f"
    )
    _write_values(Path(f"{prefix}{m_suffix}"), m_values)

    midpoint = len(frame) // 2
    for chromosome, chromosome_frame in (
        (1, frame.iloc[:midpoint, :]),
        (2, frame.iloc[midpoint:, :]),
    ):
        chromosome_frame.to_csv(
            f"{prefix}{chromosome}{ldscore_suffix}",
            sep="\t",
            index=False,
            float_format="%.3f",
        )
        _write_values(
            Path(f"{prefix}{chromosome}{m_suffix}"),
            (value / 2 for value in m_values),
        )


def generate_simulation(
    output_dir,
    *,
    seed=DEFAULT_SEED,
    n_sims=N_SIMS,
    n_snps=N_SNP,
    n_individuals=N_INDIV,
):
    """Generate the fixtures used by the LDSC simulation regression tests."""
    if n_sims < 1:
        raise ValueError("n_sims must be positive")
    if n_snps < 2:
        raise ValueError("n_snps must be at least 2")

    output_dir = Path(output_dir)
    ldscore_dir = output_dir / "ldscore"
    sumstats_dir = output_dir / "sumstats"
    ldscore_dir.mkdir(parents=True, exist_ok=True)
    sumstats_dir.mkdir(parents=True, exist_ok=True)

    # RandomState deliberately preserves the stable MT19937 sequence used by the
    # original NumPy-based generator while avoiding process-global random state.
    random = np.random.RandomState(seed)
    two_ldsc = np.abs(100 * random.normal(size=2 * n_snps)).reshape((n_snps, 2))
    single_ldsc = np.sum(two_ldsc, axis=1)
    m_two = np.sum(two_ldsc, axis=0)
    m_single = np.sum(single_ldsc)
    variants = pd.DataFrame(
        {
            "CHR": np.ones(n_snps, dtype=int),
            "SNP": [f"rs{i}" for i in range(n_snps)],
            "BP": np.arange(n_snps),
        }
    )

    first = variants.copy()
    first["LD"] = two_ldsc[:, 0]
    _write_ld_scores(first, ldscore_dir / "twold_firstfile", [m_two[0]])

    second = variants.copy()
    second["LD"] = two_ldsc[:, 1]
    _write_ld_scores(second, ldscore_dir / "twold_secondfile", [m_two[1]])

    one_ldscore = variants.copy()
    one_ldscore["LD"] = single_ldsc
    _write_ld_scores(one_ldscore, ldscore_dir / "oneld_onefile", [m_single])

    two_ldscores = variants.copy()
    two_ldscores["LD1"] = two_ldsc[:, 0]
    two_ldscores["LD2"] = two_ldsc[:, 1]
    _write_ld_scores(two_ldscores, ldscore_dir / "twold_onefile", m_two)

    weights = variants.copy()
    weights["LD"] = np.ones(n_snps)
    weights.to_csv(
        ldscore_dir / "w.l2.ldscore",
        index=False,
        sep="\t",
        float_format="%.3f",
    )

    sumstats_template = pd.DataFrame(
        {
            "SNP": [f"rs{i}" for i in range(n_snps)],
            "A1": ["A"] * n_snps,
            "A2": ["G"] * n_snps,
            "N": np.full(n_snps, n_individuals),
        }
    )
    z_scale = np.sqrt(
        1
        + n_individuals
        * (
            H2_1 * two_ldsc[:, 0] / float(m_two[0])
            + H2_2 * two_ldsc[:, 1] / float(m_two[1])
        )
    )
    for simulation in range(n_sims):
        frame = sumstats_template.copy()
        frame["Z"] = random.normal(size=n_snps) * z_scale
        frame = frame.reindex(random.permutation(frame.index))
        frame.to_csv(
            sumstats_dir / str(simulation),
            sep="\t",
            index=False,
            float_format="%.3f",
        )

    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory; ldscore/ and sumstats/ are created beneath it.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-sims", type=int, default=N_SIMS)
    parser.add_argument("--n-snps", type=int, default=N_SNP)
    args = parser.parse_args(argv)
    generate_simulation(
        args.out,
        seed=args.seed,
        n_sims=args.n_sims,
        n_snps=args.n_snps,
    )


if __name__ == "__main__":
    main()
