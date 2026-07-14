from __future__ import division
from ldsc_mod.ldscore import parse as ps
import unittest
import numpy as np
import pandas as pd
import os
import pytest
from pathlib import Path
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_equal

DIR = Path(__file__).parent
assert_raises = pytest.raises
HAS_TEXT_PARSE_FIXTURES = (DIR / 'parse_test/test.cts').exists()
HAS_SINGLE_LDSCORE_FIXTURE = (DIR / 'parse_test/test.l2.ldscore.bz2').exists()
HAS_M_FIXTURES = (DIR / 'parse_test/test.l2.M').exists()
HAS_PLINK_FIXTURES = (DIR / 'plink_test/plink.bim').exists()


def test_series_eq():
    x = pd.Series([1, 2, 3])
    y = pd.Series([1, 2])
    z = pd.Series([1, 2, 4])
    assert ps.series_eq(x, x)
    assert not ps.series_eq(x, y)
    assert not ps.series_eq(x, z)


def test_get_compression():
    assert_equal(ps.get_compression('gz'), 'gzip')
    assert_equal(ps.get_compression('bz2'), 'bz2')
    assert_equal(ps.get_compression('asdf'), None)


@pytest.mark.skipif(not HAS_TEXT_PARSE_FIXTURES, reason='legacy text parsing fixtures are not distributed')
def test_read_cts():
    match_snps = pd.Series(['rs1', 'rs2', 'rs3'])
    assert_array_equal(
        ps.read_cts(os.path.join(DIR, 'parse_test/test.cts'), match_snps), [1, 2, 3])
    assert_raises(ValueError, ps.read_cts, os.path.join(
        DIR, 'parse_test/test.cts'), match_snps[0:2])


@pytest.mark.skipif(not HAS_TEXT_PARSE_FIXTURES, reason='legacy text parsing fixtures are not distributed')
def test_read_sumstats():
    x = ps.sumstats(
        os.path.join(DIR, 'parse_test/test.sumstats'), dropna=True, alleles=True)
    assert_equal(len(x), 1)
    assert_array_equal(x.SNP, 'rs1')
    assert_raises(ValueError, ps.sumstats, os.path.join(
        DIR, 'parse_test/test.l2.ldscore.gz'))


@pytest.mark.skipif(not HAS_TEXT_PARSE_FIXTURES, reason='legacy frequency fixtures are not distributed')
def test_frq_parser():
    x = ps.frq_parser(os.path.join(DIR, 'parse_test/test1.frq'), compression=None)
    assert_array_equal(x.columns, ['SNP', 'FRQ'])
    assert_array_equal(x.SNP, ['rs_' + str(i) for i in range(8)])
    assert_array_equal(x.FRQ, [.01, .1, .7, .2, .2, .2, .99, .03])
    x = ps.frq_parser(os.path.join(DIR, 'parse_test/test2.frq.gz'), compression='gzip')
    assert_array_equal(x.columns, ['SNP', 'FRQ'])
    assert_array_equal(x.SNP, ['rs_' + str(i) for i in range(8)])
    assert_array_equal(x.FRQ, [.01, .1, .3, .2, .2, .2, .01, .03])


class Test_ldscore(unittest.TestCase):

    def test_ldscore(self):
        if not HAS_SINGLE_LDSCORE_FIXTURE:
            self.skipTest('single-file LD Score fixture is not distributed')
        x = ps.ldscore(os.path.join(DIR, 'parse_test/test'))
        assert_equal(list(x['SNP']), ['rs' + str(i) for i in range(1, 23)])
        assert_equal(list(x['AL2']), list(range(1, 23)))
        assert_equal(list(x['BL2']), list(range(2, 46, 2)))

    def test_ldscore_loop(self):
        x = ps.ldscore(os.path.join(DIR, 'parse_test/test'), 2)
        assert_equal(list(x['SNP']), ['rs' + str(i) for i in range(1, 3)])
        assert_equal(list(x['AL2']), list(range(1, 3)))
        assert_equal(list(x['BL2']), list(range(2, 6, 2)))

    def test_ldscore_fromlist(self):
        if not HAS_SINGLE_LDSCORE_FIXTURE:
            self.skipTest('single-file LD Score fixture is not distributed')
        fh = os.path.join(DIR, 'parse_test/test')
        x = ps.ldscore_fromlist([fh, fh])
        assert_array_equal(x.shape, (22, 5))
        y = ps.ldscore(os.path.join(DIR, 'parse_test/test'))
        assert_array_equal(x.iloc[:, 0:3], y)
        assert_array_equal(x.iloc[:, [0, 3, 4]], y)
        assert_raises(
            ValueError, ps.ldscore_fromlist, [fh, os.path.join(DIR, 'parse_test/test2')])


@unittest.skipUnless(HAS_M_FIXTURES, 'LD Score M fixtures are not distributed')
class Test_M(unittest.TestCase):

    def test_bad_M(self):
        assert_raises(
            ValueError, ps.M, os.path.join(DIR, 'parse_test/test_bad'))

    def test_M(self):
        x = ps.M(os.path.join(DIR, 'parse_test/test'))
        assert_array_equal(x.shape, (1, 3))
        assert_array_equal(x, [[1000, 2000, 3000]])

    def test_M_loop(self):
        x = ps.M(os.path.join(DIR, 'parse_test/test'), 2)
        assert_array_equal(x.shape, (1, 2))
        assert_array_equal(x, [[3, 6]])

    def test_M_fromlist(self):
        fh = os.path.join(DIR, 'parse_test/test')
        x = ps.M_fromlist([fh, fh])
        assert_array_equal(x.shape, (1, 6))
        assert_array_equal(x, np.hstack((ps.M(fh), ps.M(fh))))


@unittest.skipUnless(HAS_PLINK_FIXTURES, 'PLINK fixtures are not distributed')
class Test_Fam(unittest.TestCase):

    def test_fam(self):
        fam = ps.PlinkFAMFile(os.path.join(DIR, 'plink_test/plink.fam'))
        assert_equal(fam.n, 5)
        correct = np.array(['per0', 'per1', 'per2', 'per3', 'per4'])
        assert_array_equal(fam.IDList.values.reshape((5,)), correct)

    def test_bad_filename(self):
        assert_raises(
            ValueError, ps.PlinkFAMFile, os.path.join(DIR, 'plink_test/plink.bim'))


@unittest.skipUnless(HAS_PLINK_FIXTURES, 'PLINK fixtures are not distributed')
class Test_Bim(unittest.TestCase):

    def test_bim(self):
        bim = ps.PlinkBIMFile(os.path.join(DIR, 'plink_test/plink.bim'))
        assert_equal(bim.n, 8)
        correct = np.array(
            ['rs_0', 'rs_1', 'rs_2', 'rs_3', 'rs_4', 'rs_5', 'rs_6', 'rs_7'])
        assert_array_equal(bim.IDList.values.reshape(8), correct)

    def test_bad_filename(self):
        assert_raises(
            ValueError, ps.PlinkBIMFile, os.path.join(DIR, 'plink_test/plink.fam'))
