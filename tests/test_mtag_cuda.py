from __future__ import absolute_import
from __future__ import division

import unittest
import sys
import types

import numpy as np

from mtag_cuda import _mtag_analysis_batched, mtag_analysis_cuda


class FakeCudaDevice(object):

    def __init__(self, device):
        self.device = device

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False


class FakeCudaRuntime(object):

    @staticmethod
    def getDeviceCount():
        return 1

    @staticmethod
    def getDeviceProperties(device):
        return {'name': b'NumPy test double'}


class FakeCuda(object):
    Device = FakeCudaDevice
    runtime = FakeCudaRuntime()


def fake_cupy_module():
    """Build the small CuPy-compatible surface used by the CUDA adapter."""
    module = types.ModuleType('cupy')
    for name in [
        'asarray', 'broadcast_to', 'einsum', 'empty', 'float64', 'linalg',
        'outer', 'sqrt', 'swapaxes',
    ]:
        setattr(module, name, getattr(np, name))
    module.asnumpy = np.asarray
    module.cuda = FakeCuda()
    return module


def reference_mtag_analysis(Zs, Ns, omega_hat, sigma_LD):
    """The original MTAG implementation, retained as a test oracle."""
    num_snps, num_traits = Zs.shape
    w_n = np.einsum('mp,pq->mpq', np.sqrt(Ns), np.eye(num_traits))
    w_n_inv = np.linalg.inv(w_n)
    sigma_n = np.einsum(
        'mpq,mqr->mpr',
        np.einsum('mpq,qr->mpr', w_n_inv, sigma_LD),
        w_n_inv,
    )

    mtag_betas = np.zeros((num_snps, num_traits))
    mtag_se = np.zeros((num_snps, num_traits))
    mtag_factor = np.zeros((num_snps, num_traits))

    for trait in range(num_traits):
        gamma_k = omega_hat[:, trait]
        tau_k_2 = omega_hat[trait, trait]
        omega_conditional = (omega_hat -
                             np.outer(gamma_k, gamma_k) / tau_k_2)
        inv_xx = np.linalg.inv(omega_conditional + sigma_n)
        yy = gamma_k / tau_k_2
        w_inv_z = np.einsum('mqp,mp->mq', w_n_inv, Zs)

        weighted = np.einsum('q,mqp->mp', yy, inv_xx)
        beta_denom = np.einsum('mp,p->m', weighted, yy)
        mtag_factor[:, trait] = np.einsum(
            'mp,m->m', weighted, 1.0 / beta_denom
        )
        mtag_betas[:, trait] = (
            np.einsum('mp,mp->m', weighted, w_inv_z) / beta_denom
        )
        mtag_se[:, trait] = np.sqrt(1.0 / beta_denom)

    return mtag_betas, mtag_se, mtag_factor


class MtagCudaMathTest(unittest.TestCase):

    def setUp(self):
        random = np.random.RandomState(8128)
        self.Zs = random.normal(size=(23, 3))
        self.Ns = random.randint(50000, 250000, size=(23, 3)).astype(float)
        self.omega = np.array([
            [2.0e-5, 8.0e-6, 5.0e-6],
            [8.0e-6, 2.5e-5, 7.0e-6],
            [5.0e-6, 7.0e-6, 1.8e-5],
        ])
        self.sigma = np.array([
            [1.0, 0.15, 0.08],
            [0.15, 1.0, 0.12],
            [0.08, 0.12, 1.0],
        ])

    def test_chunked_backend_matches_original_calculation(self):
        expected = reference_mtag_analysis(
            self.Zs, self.Ns, self.omega, self.sigma
        )
        actual = _mtag_analysis_batched(
            self.Zs,
            self.Ns,
            self.omega,
            self.sigma,
            np,
            batch_size=7,
            to_numpy=np.asarray,
        )

        for expected_array, actual_array in zip(expected, actual):
            np.testing.assert_allclose(
                actual_array, expected_array, rtol=1.0e-12, atol=1.0e-12
            )

    def test_invalid_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            _mtag_analysis_batched(
                self.Zs,
                self.Ns,
                self.omega,
                self.sigma,
                np,
                batch_size=0,
                to_numpy=np.asarray,
            )

    def test_single_trait_batches_match_original_calculation(self):
        expected = reference_mtag_analysis(
            self.Zs[:, :1], self.Ns[:, :1],
            self.omega[:1, :1], self.sigma[:1, :1]
        )
        actual = _mtag_analysis_batched(
            self.Zs[:, :1],
            self.Ns[:, :1],
            self.omega[:1, :1],
            self.sigma[:1, :1],
            np,
            batch_size=5,
            to_numpy=np.asarray,
        )

        for expected_array, actual_array in zip(expected, actual):
            np.testing.assert_allclose(
                actual_array, expected_array, rtol=1.0e-12, atol=1.0e-12
            )

    def test_public_cuda_adapter_batches_and_returns_numpy(self):
        expected = reference_mtag_analysis(
            self.Zs, self.Ns, self.omega, self.sigma
        )
        previous_cupy = sys.modules.get('cupy')
        sys.modules['cupy'] = fake_cupy_module()
        try:
            actual = mtag_analysis_cuda(
                self.Zs,
                self.Ns,
                self.omega,
                self.sigma,
                device=0,
                batch_size=7,
            )
        finally:
            if previous_cupy is None:
                del sys.modules['cupy']
            else:
                sys.modules['cupy'] = previous_cupy

        for expected_array, actual_array in zip(expected, actual):
            self.assertIsInstance(actual_array, np.ndarray)
            np.testing.assert_allclose(
                actual_array, expected_array, rtol=1.0e-12, atol=1.0e-12
            )


if __name__ == '__main__':
    unittest.main()
