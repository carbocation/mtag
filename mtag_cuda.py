"""Optional CUDA acceleration for the main MTAG estimator.

This module deliberately depends only on NumPy at import time.  CuPy is loaded
only when CUDA execution is requested, so the existing CPU installation and
command-line behavior remain unchanged.
"""

from __future__ import absolute_import
from __future__ import division

import logging
import sys

import numpy as np


def _validate_inputs(Zs, Ns, omega_hat, sigma_LD, batch_size):
    """Validate shapes before allocating host or device result arrays."""
    if batch_size <= 0:
        raise ValueError("CUDA batch size must be a positive integer")

    if Zs.ndim != 2 or Ns.ndim != 2 or Zs.shape != Ns.shape:
        raise ValueError("Zs and Ns must be two-dimensional arrays with matching shapes")

    _, num_traits = Zs.shape
    expected_matrix_shape = (num_traits, num_traits)
    if omega_hat.shape != expected_matrix_shape:
        raise ValueError("omega_hat must have shape {}".format(expected_matrix_shape))
    if sigma_LD.shape != expected_matrix_shape:
        raise ValueError("sigma_LD must have shape {}".format(expected_matrix_shape))


def _mtag_analysis_batch(xp, Zs, Ns, omega_hat, sigma_LD):
    """Run the MTAG estimator for one batch using a NumPy-compatible module."""
    num_snps, num_traits = Zs.shape
    inv_sqrt_n = 1.0 / xp.sqrt(Ns)

    # W_N is diagonal, so W_N^-1 Sigma W_N^-1 can be constructed
    # elementwise without materializing or inverting an M x P x P stack of
    # diagonal matrices.
    sigma_n = (sigma_LD[None, :, :] *
               inv_sqrt_n[:, :, None] *
               inv_sqrt_n[:, None, :])
    w_inv_z = Zs * inv_sqrt_n

    mtag_betas = xp.empty((num_snps, num_traits), dtype=xp.float64)
    mtag_se = xp.empty((num_snps, num_traits), dtype=xp.float64)
    mtag_factor = xp.empty((num_snps, num_traits), dtype=xp.float64)

    for trait in range(num_traits):
        gamma_k = omega_hat[:, trait]
        tau_k_2 = omega_hat[trait, trait]
        omega_conditional = (omega_hat -
                             xp.outer(gamma_k, gamma_k) / tau_k_2)
        xx = omega_conditional[None, :, :] + sigma_n
        yy = gamma_k / tau_k_2

        # The legacy implementation forms yy^T inv(xx). Solve the transposed
        # system for the same vector while avoiding a full batched inverse.
        rhs = xp.broadcast_to(yy, (num_snps, num_traits)).copy()[:, :, None]
        weighted = xp.linalg.solve(xp.swapaxes(xx, 1, 2), rhs)[:, :, 0]
        beta_denom = xp.einsum('mp,p->m', weighted, yy)

        mtag_factor[:, trait] = xp.einsum(
            'mp,m->m', weighted, 1.0 / beta_denom
        )
        mtag_betas[:, trait] = (xp.einsum('mp,mp->m', weighted, w_inv_z) /
                                beta_denom)
        mtag_se[:, trait] = xp.sqrt(1.0 / beta_denom)

    return mtag_betas, mtag_se, mtag_factor


def _mtag_analysis_batched(Zs, Ns, omega_hat, sigma_LD, xp, batch_size,
                           to_numpy):
    """Run chunked MTAG with ``xp`` as either NumPy or CuPy.

    Keeping this adapter backend-neutral lets the CUDA execution path be
    regression-tested with NumPy on systems without a GPU.
    """
    Zs = np.asarray(Zs, dtype=np.float64)
    Ns = np.asarray(Ns, dtype=np.float64)
    omega_hat = np.asarray(omega_hat, dtype=np.float64)
    sigma_LD = np.asarray(sigma_LD, dtype=np.float64)
    _validate_inputs(Zs, Ns, omega_hat, sigma_LD, batch_size)

    num_snps, num_traits = Zs.shape
    mtag_betas = np.empty((num_snps, num_traits), dtype=np.float64)
    mtag_se = np.empty((num_snps, num_traits), dtype=np.float64)
    mtag_factor = np.empty((num_snps, num_traits), dtype=np.float64)

    omega_device = xp.asarray(omega_hat, dtype=xp.float64)
    sigma_device = xp.asarray(sigma_LD, dtype=xp.float64)

    for start in range(0, num_snps, batch_size):
        stop = min(start + batch_size, num_snps)
        batch_results = _mtag_analysis_batch(
            xp,
            xp.asarray(Zs[start:stop], dtype=xp.float64),
            xp.asarray(Ns[start:stop], dtype=xp.float64),
            omega_device,
            sigma_device,
        )
        mtag_betas[start:stop] = to_numpy(batch_results[0])
        mtag_se[start:stop] = to_numpy(batch_results[1])
        mtag_factor[start:stop] = to_numpy(batch_results[2])

    return mtag_betas, mtag_se, mtag_factor


def _load_cupy():
    try:
        import cupy as cp
    except ImportError:
        if sys.version_info[0] < 3:
            install_hint = (
                "Python 2 requires CuPy 6.x and CUDA 10.1 or older "
                "(for example, cupy-cuda101==6.7.0)."
            )
        else:
            install_hint = (
                "Install the CuPy package matching the machine's CUDA "
                "toolkit (for example, cupy-cuda12x)."
            )
        raise RuntimeError(
            "--cuda requires CuPy. {}".format(install_hint)
        )

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as error:
        raise RuntimeError(
            "CuPy is installed, but CUDA could not be initialized: {}".format(error)
        )

    if device_count < 1:
        raise RuntimeError("--cuda was requested, but CuPy found no CUDA devices")
    return cp, device_count


def mtag_analysis_cuda(Zs, Ns, omega_hat, sigma_LD, device=0,
                       batch_size=100000):
    """Run the main MTAG calculation on a CUDA device via CuPy."""
    if device < 0:
        raise ValueError("CUDA device index must be non-negative")

    cp, device_count = _load_cupy()
    if device >= device_count:
        raise RuntimeError(
            "CUDA device {} was requested, but only {} device(s) are visible".format(
                device, device_count
            )
        )

    with cp.cuda.Device(device):
        properties = cp.cuda.runtime.getDeviceProperties(device)
        device_name = properties.get('name', 'unknown')
        if not isinstance(device_name, str):
            device_name = device_name.decode('utf-8')
        logging.info(
            "Using CUDA device {} ({}) for MTAG with batches of {} SNPs.".format(
                device, device_name, batch_size
            )
        )
        return _mtag_analysis_batched(
            Zs, Ns, omega_hat, sigma_LD, cp, batch_size, cp.asnumpy
        )
