#!/usr/bin/env python
"""Compare the MTAG NumPy and CUDA kernels on synthetic GWAS inputs."""

from __future__ import absolute_import
from __future__ import division

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtag_cuda import _mtag_analysis_batched, mtag_analysis_cuda


def make_inputs(num_snps, num_traits, seed):
    random = np.random.RandomState(seed)
    Zs = random.normal(size=(num_snps, num_traits))
    Ns = random.randint(50000, 250000, size=(num_snps, num_traits)).astype(float)

    raw = random.normal(size=(num_traits, num_traits))
    omega = np.dot(raw, raw.T)
    omega *= 2.0e-5 / np.mean(np.diag(omega))
    omega += np.eye(num_traits) * 1.0e-5

    sigma = np.full((num_traits, num_traits), 0.1)
    np.fill_diagonal(sigma, 1.0)
    return Zs, Ns, omega, sigma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snps', type=int, default=100000)
    parser.add_argument('--traits', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=100000)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=8128)
    parser.add_argument('--skip-cpu', action='store_true')
    args = parser.parse_args()

    inputs = make_inputs(args.snps, args.traits, args.seed)
    expected = None
    if not args.skip_cpu:
        start = time.time()
        expected = _mtag_analysis_batched(
            inputs[0], inputs[1], inputs[2], inputs[3], np,
            args.batch_size, np.asarray
        )
        cpu_seconds = time.time() - start
        print('CPU:  {:.3f} seconds'.format(cpu_seconds))

    # Warm up the CUDA context and kernels with a small batch before timing.
    warmup_size = min(1000, args.snps)
    mtag_analysis_cuda(
        inputs[0][:warmup_size], inputs[1][:warmup_size],
        inputs[2], inputs[3], args.device, args.batch_size
    )

    start = time.time()
    actual = mtag_analysis_cuda(
        inputs[0], inputs[1], inputs[2], inputs[3],
        args.device, args.batch_size
    )
    cuda_seconds = time.time() - start
    print('CUDA: {:.3f} seconds'.format(cuda_seconds))

    if expected is not None:
        print('Speedup: {:.2f}x'.format(cpu_seconds / cuda_seconds))
        max_error = max(
            np.max(np.abs(cpu_array - cuda_array))
            for cpu_array, cuda_array in zip(expected, actual)
        )
        print('Maximum absolute error: {:.3e}'.format(max_error))


if __name__ == '__main__':
    main()
