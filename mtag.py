#!/usr/bin/env python3
'''
'''

from __future__ import division
from __future__ import absolute_import
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special
import argparse
import itertools
import time
import os, re
import joblib
import sys, gzip, bz2
import logging
from argparse import Namespace

from ldsc_mod.ldscore import sumstats as sumstats_sig
from ldsc_mod.ldscore import allele_info
from ldsc_mod.ldscore import regressions as ldsc_reg

import mtag_munge as munge_sumstats
import warnings
from functools import reduce
warnings.filterwarnings("ignore")

__version__ = '1.0.8'

borderline = "<><><<>><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"

header ="\n"
header += borderline +"\n"
header += "<>\n"
header += "<> MTAG: Multi-trait Analysis of GWAS \n"
header += "<> Version: {}\n".format(str(__version__))
header += "<> (C) 2017 Omeed Maghzian, Raymond Walters, and Patrick Turley\n"
header += "<> Harvard University Department of Economics / Broad Institute of MIT and Harvard\n"
header += "<> GNU General Public License v3\n"
header += borderline + "\n"
header += "<> Note:  It is recommended to run your own QC on the input before using this program. \n"
header += "<> Software-related correspondence: jjala.ssgac@gmail.com \n"
header += "<> All other correspondence: paturley@broadinstitute.org \n"
header += borderline +"\n"
header += "\n\n"

pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 800)
pd.set_option('display.precision', 12)
pd.set_option('max_colwidth', 800)
pd.set_option('colheader_justify', 'left')

np.set_printoptions(linewidth=800)
np.set_printoptions(precision=3)

DEFAULT_MEDIAN_Z_THRESHOLD = 0.1


## General helper functions
def safely_create_folder(folder_path):
    try:
        os.makedirs(folder_path)
    except OSError:
        if not os.path.isdir(folder_path):
            raise

class DisableLogger():
    '''
    For disabling the logging module when calling munge_sumstats
    '''
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, a, b, c):
       logging.disable(logging.NOTSET)

## Read / Write functions
def _read_SNPlist(file_path, SNP_index):

    # TODO Add more possible ways of reading SNPlists
    snplist = pd.read_csv(file_path, header=0, index_col=False)
    if SNP_index not in snplist.columns:
        raise ValueError("SNPlist read from {} does not include --snp_name {} in its columns.".format(file_path, SNP_index))
    return snplist

def _read_GWAS_sumstats(GWAS_file_name, chunksize):
    '''
    read GWAS summary statistics from file that is in one of the acceptable formats.
    '''
    # TODO read more file types
    (openfunc, compression) = munge_sumstats.get_compression(GWAS_file_name)
    dat_gen = pd.read_csv(GWAS_file_name, index_col=False, header=0, sep=r'\s+', compression=compression, na_values=['.','NA'],
        iterator=True, chunksize=chunksize)
    dat_gen = list(dat_gen)
    dat_gen_unfiltered = pd.concat(dat_gen, axis=0).reset_index(drop=True)

    return  dat_gen_unfiltered, dat_gen

def _read_matrix(file_path):
    '''
    For reading 2-dimensional matrices. These files must be in .npy form or whitespace delimited .csv files
    '''
    ext = file_path[-4:]
    if ext == '.npy':
        return np.load(file_path)
    if ext == '.txt':
        return np.loadtxt(file_path)
    else:
        raise ValueError('{} is not one of the acceptable file paths for reading in matrix-valued objects.'.format(ext))

## LDSC related functions
def sec_to_str(t):
    '''Convert seconds to days:hours:minutes:seconds'''
    [d, h, m, s, n] = reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=d)
    if h > 0:
        f += '{H}h:'.format(H=h)
    if m > 0:
        f += '{M}m:'.format(M=m)

    f += '{S}s'.format(S=s)
    return f

class Logger_to_Logging(object):
    """
    Logger class that write uses logging module and is needed to use munge_sumstats or ldsc from the LD score package.
    """
    def __init__(self):
        logging.info('created Logger instance to pass through ldsc.')
        super(Logger_to_Logging, self).__init__()


    def log(self,x):
        logging.info(x)

def _perform_munge(args, GWAS_df, GWAS_dat_gen,p):

    '''
    Use the modified LDSC munging to clean sumstats
    '''
    original_cols = GWAS_df.columns
    merge_alleles = None
    out = None
    ignore_list = ""

    if args.info_min is None:
        ignore_list += "info"

    a1_munge = None if args.a1_name == "a1" else args.a1_name
    a2_munge = None if args.a2_name == "a2" else args.a2_name
    eaf_munge = None if args.eaf_name == "freq" else args.eaf_name
    p_munge = None if args.p_name == "p" else args.p_name
    beta_munge = args.beta_name if args.beta_name is not None else 'beta'
    z_munge = args.z_name if args.z_name is not None else 'z'
    n_add = args.n_list[p] if args.n_value is not None else None

    if args.use_beta_se:
        argnames = Namespace(sumstats=None,N=None,N_cas=None,N_con=None,out=out,maf_min=args.maf_min_list[p], info_min =args.info_min_list[p],daner=False, no_alleles=False, merge_alleles=merge_alleles,n_min=args.n_min_list[p],chunksize=args.chunksize, snp=args.snp_name,N_col=args.n_name, N_cas_col=None, N_con_col = None, a1=a1_munge, a2=a2_munge, p=p_munge, frq=eaf_munge,signed_sumstats=beta_munge+',0', keep_beta=True, keep_se=True, info=None,info_list=None, nstudy=None,nstudy_min=None,ignore=ignore_list,a1_inc=False, keep_maf=True, daner_n=False, keep_str_ambig=True, input_datgen=GWAS_dat_gen, cnames=list(original_cols), n_value=n_add)
    else:
        argnames = Namespace(sumstats=None,N=None,N_cas=None,N_con=None,out=out,maf_min=args.maf_min_list[p], info_min =args.info_min_list[p],daner=False, no_alleles=False, merge_alleles=merge_alleles,n_min=args.n_min_list[p],chunksize=args.chunksize, snp=args.snp_name,N_col=args.n_name, N_cas_col=None, N_con_col = None, a1=a1_munge, a2=a2_munge, p=p_munge,frq=eaf_munge,signed_sumstats=z_munge+',0', keep_beta=False, keep_se=False, info=None,info_list=None, nstudy=None,nstudy_min=None,ignore=ignore_list,a1_inc=False, keep_maf=True, daner_n=False, keep_str_ambig=True, input_datgen=GWAS_dat_gen, cnames=list(original_cols), n_value=n_add)

    logging.info(borderline)
    logging.info('Munging Trait {}  {}'.format(p+1,borderline[:-17]))
    logging.info(borderline)

    argnames.median_z_cutoff = args.median_z_cutoff
    munged_results = munge_sumstats.munge_sumstats(argnames, write_out=False, new_log=False)
    GWAS_df = GWAS_df.merge(munged_results, how='inner',left_on =args.snp_name,right_on='SNP',suffixes=('','_ss'))

    if args.n_value is not None:
        GWAS_df = GWAS_df[list(original_cols) + ["N"]]
    else:
        GWAS_df = GWAS_df[original_cols]

    logging.info(borderline)
    logging.info('Munging of Trait {} complete. SNPs remaining:\t {}'.format(p+1, len(GWAS_df)))
    logging.info(borderline+'\n')

    return GWAS_df, munged_results

def _quick_mode(ndarray,axis=0):
    '''
    From stackoverflow: Efficient calculation of the mode of an array. Scipy.stats.mode is way too slow
    '''
    if ndarray.size == 1:
        return (ndarray[0],1)
    elif ndarray.size == 0:
        raise Exception('Attempted to find mode on an empty array!')
    try:
        axis = [i for i in range(ndarray.ndim)][axis]
    except IndexError:
        raise Exception('Axis %i out of range for array with %i dimension(s)' % (axis,ndarray.ndim))
    srt = np.sort(ndarray, axis=axis)
    dif = np.diff(srt, axis=axis)
    shape = [i for i in dif.shape]
    shape[axis] += 2
    indices = np.indices(shape)[axis]
    index = tuple([slice(None) if i != axis else slice(1,-1) for i in range(dif.ndim)])
    indices[index][dif == 0] = 0
    indices.sort(axis=axis)
    bins = np.diff(indices, axis=axis)
    location = np.argmax(bins, axis=axis)
    mesh = np.indices(bins.shape)
    index = tuple([slice(None) if i != axis else 0 for i in range(dif.ndim)])
    index = [mesh[i][index].ravel() if i != axis else location.ravel() for i in range(bins.ndim)]
    counts = bins[tuple(index)].reshape(location.shape)
    index[axis] = indices[tuple(index)]
    modals = srt[tuple(index)].reshape(location.shape)
    return (modals, counts)

def set_default_cnames(args):
    return{args.snp_name: 'SNP',
            args.z_name: 'Z',
            args.n_name: 'N',
            args.beta_name: 'BETA',
            args.se_name: 'SE',
            args.eaf_name: 'FRQ',
            args.chr_name: 'CHR',
            args.bpos_name: 'BP',
            args.a1_name: 'A1',
            args.a2_name: 'A2',
            args.p_name: "P"}


def _internal_column_renames(columns, args):
    """Choose at most one input column for each canonical MTAG name.

    The legacy alias map renames every recognized spelling.  If a file has,
    for example, both ``P_BOLT_LMM_INF`` and ``P_BOLT_LMM``, that produces two
    columns named ``P``.  Older pandas versions allowed the duplicate labels,
    while current pandas refuses to reindex any such frame.  Prefer columns
    named explicitly by the user and use the broad alias map only as a
    fallback, leaving unselected aliases untouched.
    """
    columns = list(columns)
    broad_renames = munge_sumstats.set_default_cnames(args)
    requested_renames = set_default_cnames(args)
    selected_sources = {}

    for source, target in requested_renames.items():
        if source is None or source not in columns:
            continue
        previous = selected_sources.get(target)
        if previous is not None and previous != source:
            raise ValueError(
                'Input columns {} and {} were both selected for internal '
                'column {}.'.format(previous, source, target)
            )
        selected_sources[target] = source

    alias_targets = {
        broad_renames[source]
        for source in columns
        if source in broad_renames
    }
    for target in alias_targets:
        if target in selected_sources:
            continue
        if target in columns:
            selected_sources[target] = target
            continue
        selected_sources[target] = next(
            source for source in columns if broad_renames.get(source) == target
        )

    renames = {}
    for target, source in selected_sources.items():
        if source != target:
            renames[source] = target
            if target in columns:
                replacement = target + '_unselected'
                suffix = 2
                while replacement in columns or replacement in renames.values():
                    replacement = '{}_unselected_{}'.format(target, suffix)
                    suffix += 1
                renames[target] = replacement

    return renames


def _resolved_input_columns(columns, args):
    """Map the one selected source column for each MTAG field."""
    renames = _internal_column_renames(columns, args)
    resolved = {}
    recognized = {
        'SNP', 'Z', 'N', 'FRQ', 'CHR', 'BP', 'A1', 'A2', 'P',
        'INFO', 'SE',
    }
    for source in columns:
        target = renames.get(source, source)
        if target in recognized and target not in resolved:
            resolved[target] = source
    return resolved


def _update_filter_counts(mask, condition, drops, key):
    """Apply one legacy munge filter and record its incremental drop count."""
    old_count = int(mask.sum())
    mask &= np.asarray(condition, dtype=bool)
    drops[key] += old_count - int(mask.sum())


def _uppercase_by_unique_value(series):
    """Uppercase a low-cardinality string Series without per-row callbacks."""
    values = series.dropna().unique()
    mapping = {value: value.upper() for value in values}
    return series.map(mapping)


def _trait_needs_position_columns(args, trait_index):
    return (
        not args.no_chr_data
        and (trait_index == 0 or args.meta_format)
    )


def _load_trait_fast(args, file_path, trait_index):
    """Read, QC, and retain one trait without a secondary munged dataframe.

    This follows the filtering order used by ``mtag_munge.parse_dat``.  A
    narrow copy of all raw core rows is retained alongside accepted SNP/N/Z
    keys so the historical duplicate-SNP and raw-row selection semantics are
    unchanged.
    """
    start_time = time.time()
    (_, compression) = munge_sumstats.get_compression(file_path)
    header = list(pd.read_csv(
        file_path,
        index_col=False,
        header=0,
        sep=r'\s+',
        compression=compression,
        nrows=0,
    ).columns)
    resolved = _resolved_input_columns(header, args)

    required = ['SNP', 'Z', 'FRQ', 'A1', 'A2', 'P']
    if args.n_value is None:
        required.append('N')
    if _trait_needs_position_columns(args, trait_index):
        required.extend(['CHR', 'BP'])
    if args.info_min_list[trait_index] is not None:
        required.append('INFO')
    missing = [field for field in required if field not in resolved]
    if missing:
        raise ValueError(
            'Could not find required input columns for fast loading: {}'
            .format(', '.join(missing))
        )

    read_targets = list(required)
    if 'SE' in resolved:
        read_targets.append('SE')
    if 'N' in resolved and 'N' not in read_targets:
        # Legacy munging considers an input N column when checking missing
        # values even if --n_value later replaces it.
        read_targets.append('N')
    read_targets = list(dict.fromkeys(read_targets))
    source_to_target = {
        resolved[target]: target for target in read_targets
    }
    usecols = list(source_to_target)

    logging.info(
        'Read in Trait {} summary statistics from {} using the fused loader ...'
        .format(trait_index + 1, file_path)
    )
    logging.info(borderline)
    logging.info(
        'Munging Trait {}  {}'.format(
            trait_index + 1, borderline[:-17]
        )
    )
    logging.info(borderline)
    logging.info('Interpreting column names as follows:')
    logging.info('\n'.join(
        '{}:\t{}'.format(source, target)
        for source, target in source_to_target.items()
    ) + '\n')

    filter_args = Namespace(
        maf_min=args.maf_min_list[trait_index],
        info_min=args.info_min_list[trait_index],
    )
    raw_columns = ['SNP', 'Z', 'FRQ', 'A1', 'A2']
    if args.n_value is None:
        raw_columns.append('N')
    if _trait_needs_position_columns(args, trait_index):
        raw_columns.extend(['CHR', 'BP'])

    raw_chunks = []
    accepted_masks = []
    total_rows = 0
    drops = {
        'NA': 0, 'P': 0, 'INFO': 0, 'FRQ': 0,
        'A': 0, 'SE': 0,
    }
    reader = pd.read_csv(
        file_path,
        index_col=False,
        header=0,
        sep=r'\s+',
        compression=compression,
        usecols=usecols,
        na_values=['.', 'NA', 'NaN'],
        iterator=True,
        chunksize=int(args.chunksize),
        dtype={resolved['Z']: np.float64},
    )
    for source_chunk in reader:
        total_rows += len(source_chunk)
        data = source_chunk.rename(columns=source_to_target)
        data['A1'] = _uppercase_by_unique_value(data['A1'])
        data['A2'] = _uppercase_by_unique_value(data['A2'])
        raw_chunks.append(data[raw_columns].copy())

        qc_columns = ['SNP', 'Z', 'FRQ', 'A1', 'A2', 'P']
        if 'N' in data.columns:
            qc_columns.append('N')
        if 'SE' in data.columns:
            qc_columns.append('SE')
        mask = np.ones(len(data), dtype=bool)
        for column in qc_columns:
            mask &= data[column].notna().to_numpy()
        drops['NA'] += len(data) - int(mask.sum())

        if 'INFO' in data.columns:
            _update_filter_counts(
                mask,
                munge_sumstats.filter_info(data['INFO'], filter_args),
                drops,
                'INFO',
            )
        _update_filter_counts(
            mask,
            munge_sumstats.filter_frq(data['FRQ'], filter_args),
            drops,
            'FRQ',
        )
        if 'SE' in data.columns:
            _update_filter_counts(
                mask,
                munge_sumstats.filter_se(data['SE'], filter_args),
                drops,
                'SE',
            )
        _update_filter_counts(
            mask,
            munge_sumstats.filter_pvals(data['P'], filter_args),
            drops,
            'P',
        )
        allele_pairs = data['A1'] + data['A2']
        _update_filter_counts(
            mask,
            munge_sumstats.filter_alleles(allele_pairs, True),
            drops,
            'A',
        )

        accepted_masks.append(mask)

    raw = pd.concat(raw_chunks, axis=0, ignore_index=True)
    accepted_mask = np.concatenate(accepted_masks)
    accepted_before_dedup = int(accepted_mask.sum())
    logging.info(
        'Read {} SNPs from --sumstats file.\n'
        'Removed {} SNPs with missing values.\n'
        'Removed {} SNPs with INFO <= {}.\n'
        'Removed {} SNPs with MAF <= {}.\n'
        'Removed {} SNPs with SE <0 or NaN values.\n'
        'Removed {} SNPs with out-of-bounds p-values.\n'
        'Removed {} variants that were not SNPs. Note: strand ambiguous '
        'SNPs were not dropped.\n'
        '{} SNPs remain.'.format(
            total_rows,
            drops['NA'],
            drops['INFO'],
            args.info_min_list[trait_index],
            drops['FRQ'],
            args.maf_min_list[trait_index],
            drops['SE'],
            drops['P'],
            drops['A'],
            accepted_before_dedup,
        )
    )

    if args.n_value is not None:
        sample_size = args.n_list[trait_index]
        logging.info(
            'Adding uniform sample size {} to summary statistics.'.format(
                sample_size
            )
        )
        raw['N'] = sample_size

    # Factorize SNPs once and use integer codes for both legacy duplicate
    # decisions: the first QC-passing occurrence controls N filtering, while
    # the first raw occurrence is retained in MTAG's output frame.
    snp_codes, unique_snps = pd.factorize(raw['SNP'], sort=False)
    position = np.arange(len(raw), dtype=np.int64)
    valid_codes = snp_codes >= 0
    accepted_position = np.full(len(unique_snps), len(raw), dtype=np.int64)
    accepted_rows = accepted_mask & valid_codes
    np.minimum.at(
        accepted_position,
        snp_codes[accepted_rows],
        position[accepted_rows],
    )
    accepted_codes = accepted_position < len(raw)
    accepted = raw.iloc[accepted_position[accepted_codes]][
        ['SNP', 'Z', 'N']
    ].reset_index(drop=True)
    logging.info(
        'Removed {} SNPs with duplicated rs numbers ({} SNPs remain).'
        .format(accepted_before_dedup - len(accepted), len(accepted))
    )
    n_min = args.n_min_list[trait_index]
    if n_min is None:
        n_min = accepted['N'].quantile(0.9) / 1.5
    old_count = len(accepted)
    n_pass = accepted['N'].to_numpy() >= n_min
    accepted_code_indices = np.flatnonzero(accepted_codes)
    kept_code_indices = accepted_code_indices[n_pass]
    accepted = accepted.loc[n_pass].reset_index(drop=True)
    logging.info(
        'Removed {} SNPs with N < {} ({} SNPs remain).'.format(
            old_count - len(accepted), n_min, len(accepted)
        )
    )
    logging.info(munge_sumstats.check_median(
        accepted['Z'], 0.0, args.median_z_cutoff, 'SIGNED_SUMSTAT'
    ))
    logging.info('Dropping snps with null values')
    chi_squared = np.square(accepted['Z'])
    logging.info(
        '\nMetadata:\nMean chi^2 = {}\nLambda GC = {}\nMax chi^2 = {}\n'
        '{} Genome-wide significant SNPs (some may have been removed by '
        'filtering).'.format(
            round(chi_squared.mean(), 3),
            round(chi_squared.median() / 0.4549, 3),
            round(chi_squared.max(), 3),
            int((chi_squared > 29).sum()),
        )
    )

    kept_codes = np.zeros(len(unique_snps), dtype=bool)
    kept_codes[kept_code_indices] = True
    membership_mask = valid_codes & kept_codes[np.maximum(snp_codes, 0)]
    z_checker = np.mean(np.square(raw.loc[membership_mask, 'Z']))
    membership_count = int(membership_mask.sum())
    first_raw_position = np.full(len(unique_snps), len(raw), dtype=np.int64)
    np.minimum.at(
        first_raw_position,
        snp_codes[valid_codes],
        position[valid_codes],
    )
    raw = raw.iloc[first_raw_position[kept_codes]].reset_index(drop=True)
    logging.info(
        '{}\nMunging of Trait {} complete. SNPs remaining:\t {}\n{}\n'
        .format(borderline, trait_index + 1, membership_count, borderline)
    )
    duplicate_raw_count = membership_count - len(raw)
    if duplicate_raw_count > 0:
        logging.info(
            'Trait {}: Dropped {} SNPs for duplicate values in the '
            '"snp_name" column'.format(
                trait_index + 1, duplicate_raw_count
            )
        )
    logging.info(
        'Fused loading and munging of Trait {} took {:.3f} seconds.'.format(
            trait_index + 1, time.time() - start_time
        )
    )
    return raw, total_rows, z_checker


def _polars_module():
    """Import the Rust-backed I/O engine with a single-thread default."""
    os.environ.setdefault('POLARS_MAX_THREADS', '1')
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            'MTAG requires Polars; install the dependencies in '
            'requirements.txt'
        ) from exc
    return pl


def _has_tab_delimited_header(file_path):
    """Return whether an input has the true TSV header Polars requires."""
    return '\t' in _read_input_header(file_path)


def _is_polars_input_compatible(file_path):
    """Return whether Polars can directly parse this MTAG input."""
    (_, compression) = munge_sumstats.get_compression(file_path)
    return compression != 'bz2' and _has_tab_delimited_header(file_path)


def _read_input_header(file_path):
    """Read one header line as text from supported compression formats."""
    (openfunc, _) = munge_sumstats.get_compression(file_path)
    with openfunc(file_path, mode='r') as input_file:
        header = input_file.readline()
    if isinstance(header, bytes):
        header = header.decode('utf-8')
    return header


def _polars_frame_to_pandas(frame):
    """Convert a narrow Polars result without requiring PyArrow."""
    values = {column: frame[column].to_numpy() for column in frame.columns}
    chromosome_columns = [
        column for column in values
        if column == 'CHR' or re.fullmatch(r'CHR\d+', column)
    ]
    for column in chromosome_columns:
        chromosomes = values[column].astype(object)
        numeric = np.fromiter(
            (
                value is not None and value.lstrip('+-').isdigit()
                for value in chromosomes
            ),
            dtype=bool,
            count=len(chromosomes),
        )
        chromosomes[numeric] = np.asarray(
            chromosomes[numeric], dtype=np.int64
        )
        values[column] = chromosomes
    return pd.DataFrame(values)


def _write_output_frame(args, frame, path, na_rep=''):
    """Write a result table with the selected pandas or Polars backend."""
    output_backend = getattr(args, 'output_backend', 'polars')
    if getattr(args, 'legacy_loader', False):
        output_backend = 'pandas'
    if output_backend == 'pandas':
        frame.to_csv(path, sep='\t', index=False, na_rep=na_rep)
        return

    pl = _polars_module()
    values = {}
    for column in frame.columns:
        series = frame[column]
        if series.dtype == object or isinstance(
            series.dtype, pd.StringDtype
        ):
            # In particular, CHR can contain both integer autosomes and X/Y.
            # Converting object columns to nullable strings avoids PyArrow and
            # produces the same textual values as pandas' CSV writer.
            values[column] = series.astype('string').to_numpy(
                dtype=object, na_value=None
            )
        else:
            values[column] = series.to_numpy()
    output = pl.DataFrame(values)
    float_columns = [
        column for column, dtype in output.schema.items()
        if dtype in (pl.Float32, pl.Float64)
    ]
    if float_columns:
        # Polars distinguishes NaN from null. pandas to_csv applies na_rep to
        # both, so normalize NaNs before writing for matching missing values.
        output = output.with_columns([
            pl.col(column).fill_nan(None) for column in float_columns
        ])
    output.write_csv(path, separator='\t', null_value=na_rep)


def _load_trait_polars(args, file_path, trait_index, return_polars=False):
    """Rust-backed TSV loader matching the fused pandas QC semantics."""
    pl = _polars_module()
    start_time = time.time()
    header_line = _read_input_header(file_path)
    if '\t' not in header_line:
        raise ValueError(
            '--load-backend polars currently requires tab-delimited input; '
            'use --load-backend pandas for arbitrary whitespace delimiters'
        )
    header = header_line.rstrip('\r\n').split('\t')
    resolved = _resolved_input_columns(header, args)

    required = ['SNP', 'Z', 'FRQ', 'A1', 'A2', 'P']
    if args.n_value is None:
        required.append('N')
    if _trait_needs_position_columns(args, trait_index):
        required.extend(['CHR', 'BP'])
    if args.info_min_list[trait_index] is not None:
        required.append('INFO')
    missing = [field for field in required if field not in resolved]
    if missing:
        raise ValueError(
            'Could not find required input columns for Polars loading: {}'
            .format(', '.join(missing))
        )

    read_targets = list(required)
    if 'SE' in resolved:
        read_targets.append('SE')
    if 'N' in resolved and 'N' not in read_targets:
        read_targets.append('N')
    read_targets = list(dict.fromkeys(read_targets))
    source_to_target = {
        resolved[target]: target for target in read_targets
    }
    usecols = list(source_to_target)
    schema = {}
    for source, target in source_to_target.items():
        if target in {'SNP', 'A1', 'A2', 'CHR'}:
            schema[source] = pl.String
        elif target == 'BP':
            schema[source] = pl.Int64
        elif target == 'N':
            # Preserve integer sample-size columns when Polars can infer them;
            # decimal effective-N columns continue to infer as Float64.
            continue
        else:
            schema[source] = pl.Float64

    logging.info(
        'Read in Trait {} summary statistics from {} using Polars ...'.format(
            trait_index + 1, file_path
        )
    )
    data = pl.read_csv(
        file_path,
        separator='\t',
        columns=usecols,
        null_values=['.', 'NA', 'NaN'],
        schema_overrides=schema,
        infer_schema_length=10000,
        rechunk=True,
        n_threads=1,
    ).rename(source_to_target).with_columns(
        pl.col('A1').str.to_uppercase(),
        pl.col('A2').str.to_uppercase(),
    )
    total_rows = len(data)
    if args.n_value is not None:
        sample_size = args.n_list[trait_index]
        data = data.with_columns(
            pl.lit(sample_size, dtype=pl.Int64).alias('N')
        )
        logging.info(
            'Adding uniform sample size {} to summary statistics.'.format(
                sample_size
            )
        )

    raw_columns = ['SNP', 'Z', 'FRQ', 'A1', 'A2', 'N']
    if _trait_needs_position_columns(args, trait_index):
        raw_columns.extend(['CHR', 'BP'])
    raw = data.select(raw_columns)

    qc_columns = ['SNP', 'Z', 'FRQ', 'A1', 'A2', 'P']
    if args.n_value is None and 'N' in data.columns:
        qc_columns.append('N')
    if 'SE' in data.columns:
        qc_columns.append('SE')
    conditions = [
        pl.all_horizontal([
            pl.col(column).is_not_null() for column in qc_columns
        ])
    ]
    drop_keys = ['NA']
    if 'INFO' in data.columns:
        conditions.append(
            pl.col('INFO') >= args.info_min_list[trait_index]
        )
        drop_keys.append('INFO')
    frequency = pl.col('FRQ')
    conditions.append(
        (frequency > 0)
        & (frequency < 1)
        & (
            pl.min_horizontal(frequency, 1 - frequency)
            > args.maf_min_list[trait_index]
        )
    )
    drop_keys.append('FRQ')
    if 'SE' in data.columns:
        conditions.append(pl.col('SE') >= 0)
        drop_keys.append('SE')
    conditions.append((pl.col('P') > 0) & (pl.col('P') <= 1))
    drop_keys.append('P')
    conditions.append(
        pl.concat_str(['A1', 'A2']).is_in(
            list(allele_info.VALID_andSA_SNPS)
        )
    )
    drop_keys.append('A')

    cumulative = conditions[0]
    cumulative_masks = [cumulative]
    for condition in conditions[1:]:
        cumulative = cumulative & condition
        cumulative_masks.append(cumulative)
    counts = data.select([
        mask.sum().alias('filter_{}'.format(index))
        for index, mask in enumerate(cumulative_masks)
    ]).row(0)
    drops = {}
    previous_count = total_rows
    for key, count in zip(drop_keys, counts):
        drops[key] = previous_count - count
        previous_count = count
    for key in ['INFO', 'SE']:
        drops.setdefault(key, 0)

    accepted_before_dedup = counts[-1]
    accepted = data.filter(cumulative_masks[-1]).unique(
        subset='SNP', keep='first', maintain_order=True
    )
    duplicate_accepted_count = accepted_before_dedup - len(accepted)
    n_min = args.n_min_list[trait_index]
    if n_min is None:
        n_min = accepted.select(
            pl.col('N').quantile(0.9, interpolation='linear') / 1.5
        ).item()
    accepted_before_n = len(accepted)
    accepted = accepted.filter(pl.col('N') >= n_min)

    logging.info(
        'Read {} SNPs from --sumstats file.\n'
        'Removed {} SNPs with missing values.\n'
        'Removed {} SNPs with INFO <= {}.\n'
        'Removed {} SNPs with MAF <= {}.\n'
        'Removed {} SNPs with SE <0 or NaN values.\n'
        'Removed {} SNPs with out-of-bounds p-values.\n'
        'Removed {} variants that were not SNPs. Note: strand ambiguous '
        'SNPs were not dropped.\n'
        '{} SNPs remain.\n'
        'Removed {} SNPs with duplicated rs numbers ({} SNPs remain).\n'
        'Removed {} SNPs with N < {} ({} SNPs remain).'.format(
            total_rows,
            drops['NA'],
            drops['INFO'],
            args.info_min_list[trait_index],
            drops['FRQ'],
            args.maf_min_list[trait_index],
            drops['SE'],
            drops['P'],
            drops['A'],
            accepted_before_dedup,
            duplicate_accepted_count,
            accepted_before_n,
            accepted_before_n - len(accepted),
            n_min,
            len(accepted),
        )
    )
    signed_z = accepted['Z'].to_numpy()
    logging.info(munge_sumstats.check_median(
        signed_z, 0.0, args.median_z_cutoff, 'SIGNED_SUMSTAT'
    ))
    chi_squared = np.square(signed_z)
    logging.info(
        '\nMetadata:\nMean chi^2 = {}\nLambda GC = {}\nMax chi^2 = {}\n'
        '{} Genome-wide significant SNPs (some may have been removed by '
        'filtering).'.format(
            round(np.mean(chi_squared), 3),
            round(np.median(chi_squared) / 0.4549, 3),
            round(np.max(chi_squared), 3),
            int(np.sum(chi_squared > 29)),
        )
    )

    raw_membership = raw.join(
        accepted.select('SNP'),
        on='SNP',
        how='semi',
        maintain_order='left',
    )
    z_checker = raw_membership.select(
        (pl.col('Z') * pl.col('Z')).mean()
    ).item()
    membership_count = len(raw_membership)
    raw = raw_membership.unique(
        subset='SNP', keep='first', maintain_order=True
    )
    duplicate_raw_count = membership_count - len(raw)
    if duplicate_raw_count > 0:
        logging.info(
            'Trait {}: Dropped {} SNPs for duplicate values in the '
            '"snp_name" column'.format(
                trait_index + 1, duplicate_raw_count
            )
        )
    logging.info(
        'Polars loading and munging of Trait {} took {:.3f} seconds.'.format(
            trait_index + 1, time.time() - start_time
        )
    )
    if return_polars:
        return raw, total_rows, z_checker
    return _polars_frame_to_pandas(raw), total_rows, z_checker


def _load_and_merge_polars_standard(args, input_files):
    """Keep standard multi-trait harmonization in Polars until final output."""
    pl = _polars_module()
    trait_frames = []
    for trait_index, file_path in enumerate(input_files):
        frame, _, z_checker = _load_trait_polars(
            args, file_path, trait_index, return_polars=True
        )
        if z_checker < 1.02 and not args.force:
            raise ValueError(
                'The mean chi2 statistic of trait {} is less than 1.02, '
                'which may lead to unstable estimates. To perform MTAG on '
                'your results anyways, include the --force option, though '
                'the estimates should be interpreted cautiously.'.format(
                    trait_index + 1
                )
            )
        if z_checker < 1.02:
            logging.info(
                'Warning: The mean chi2 statistic of trait {} is less 1.02 '
                '- MTAG estimates may be unstable.'.format(trait_index + 1)
            )
        frame = frame.rename({
            column: '{}{}'.format(column, trait_index)
            for column in frame.columns
            if column != 'SNP'
        })
        trait_frames.append(frame)

    strand_ambiguous = [
        pair for pair, is_ambiguous in allele_info.STRAND_AMBIGUOUS.items()
        if is_ambiguous
    ]
    combined = trait_frames[0].with_columns(
        pl.concat_str(['A10', 'A20'])
        .is_in(strand_ambiguous)
        .alias('strand_ambig')
    )
    if args.incld_ambig_snps:
        logging.info(
            '{} strand ambiguous SNPs in Trait 1 are included.'.format(
                combined['strand_ambig'].sum()
            )
        )
    else:
        old_count = len(combined)
        combined = combined.filter(~pl.col('strand_ambig'))
        logging.info(
            'Dropped {} SNPs due to strand ambiguity, {} SNPs remain in '
            'intersection after merging trait1'.format(
                old_count - len(combined), len(combined)
            )
        )

    for trait_index in range(1, len(trait_frames)):
        combined = combined.join(
            trait_frames[trait_index],
            on='SNP',
            how='inner',
            maintain_order='left',
        )
        old_count = len(combined)
        same_order = (
            (pl.col('A10') == pl.col('A1{}'.format(trait_index)))
            & (pl.col('A20') == pl.col('A2{}'.format(trait_index)))
        )
        flipped_order = (
            (pl.col('A10') == pl.col('A2{}'.format(trait_index)))
            & (pl.col('A20') == pl.col('A1{}'.format(trait_index)))
        )
        combined = combined.with_columns(
            flipped_order.alias('flip_snps{}'.format(trait_index))
        ).filter(same_order | flipped_order)
        if len(combined) < old_count:
            logging.info(
                'Dropped {} SNPs due to inconsistent allele pairs from '
                'phenotype {}. {} SNPs remain.'.format(
                    old_count - len(combined),
                    trait_index + 1,
                    len(combined),
                )
            )
        flip_column = 'flip_snps{}'.format(trait_index)
        flip_count = combined[flip_column].sum()
        if flip_count > 0:
            a1_column = 'A1{}'.format(trait_index)
            a2_column = 'A2{}'.format(trait_index)
            combined = combined.with_columns(
                pl.when(pl.col(flip_column))
                .then(-pl.col('Z{}'.format(trait_index)))
                .otherwise(pl.col('Z{}'.format(trait_index)))
                .alias('Z{}'.format(trait_index)),
                pl.when(pl.col(flip_column))
                .then(1.0 - pl.col('FRQ{}'.format(trait_index)))
                .otherwise(pl.col('FRQ{}'.format(trait_index)))
                .alias('FRQ{}'.format(trait_index)),
                pl.when(pl.col(flip_column))
                .then(pl.col(a2_column))
                .otherwise(pl.col(a1_column))
                .alias(a1_column),
                pl.when(pl.col(flip_column))
                .then(pl.col(a1_column))
                .otherwise(pl.col(a2_column))
                .alias(a2_column),
            )
            logging.info(
                'Flipped the signs of of {} SNPs to make them consistent '
                'with the effect allele orderings of the first trait.'.format(
                    flip_count
                )
            )

    if args.only_chr is not None and not args.no_chr_data:
        chromosomes = args.only_chr.split(',')
        combined = combined.filter(
            pl.col('CHR0').cast(pl.String).is_in(chromosomes)
        )
    logging.info(
        '... Merge of GWAS summary statistics complete. Number of SNPs:\t {}'
        .format(len(combined))
    )
    combined = _polars_frame_to_pandas(combined)
    args.P = len(input_files)
    return combined, combined, args

def load_and_merge_data(args):
    '''
    Parses file names from MTAG command line arguments and returns the relevant used for method.

    The output DATA has internal column names!
    '''

    #=====================
    # Parse inputs + filters
    #=====================
    GWAS_input_files = args.sumstats.split(',')
    P = len(GWAS_input_files)  # of phenotypes/traits
    if args.n_min is not None:
        args.n_min_list = [float(x) for x in args.n_min.split(',')]
        if len(args.n_min_list) == 1:
            args.n_min_list = args.n_min_list * P
    else:
        args.n_min_list = [None]*P

    if args.maf_min is not None:
        args.maf_min_list = [float(x) for x in args.maf_min.split(',')]
        if len(args.maf_min_list) == 1:
            args.maf_min_list = args.maf_min_list * P
    else:
        args.maf_min_list = [None]*P

    if args.info_min is not None:
        args.info_min_list = [float(x) for x in args.info_min.split(',')]
        if len(args.info_min_list) == 1:
            args.info_min_list = args.info_min_list * P
    else:
        args.info_min_list = [None]*P

    if args.n_value is not None:
        args.n_list = [int(x) for x in args.n_value.split(',')]
        assert P == len(args.n_list), "Mismatch of length of --n_value and number of summary statistics."

    load_backend = getattr(args, 'load_backend', 'polars')
    if (
        load_backend == 'polars'
        and not getattr(args, 'legacy_loader', False)
        and not all(
            _is_polars_input_compatible(file_path)
            for file_path in GWAS_input_files
        )
    ):
        logging.info(
            'At least one input is not a directly readable tab-delimited '
            'Polars input; using the fused pandas loader for this analysis.'
        )
        load_backend = 'pandas'

    use_polars_standard_path = (
        load_backend == 'polars'
        and not getattr(args, 'legacy_loader', False)
        and not args.use_beta_se
        and not args.meta_format
        and args.include is None
        and args.exclude is None
    )
    if use_polars_standard_path:
        return _load_and_merge_polars_standard(args, GWAS_input_files)

    #=====================
    # Reading sumstats
    #=====================

    GWAS_d = dict()
    sumstats_format = dict()
    use_fused_loader = (
        not getattr(args, 'legacy_loader', False)
        and not args.use_beta_se
    )
    for p, GWAS_input in enumerate(GWAS_input_files):

        if use_fused_loader:
            if load_backend == 'polars':
                loaded_trait = _load_trait_polars(args, GWAS_input, p)
            else:
                loaded_trait = _load_trait_fast(args, GWAS_input, p)
            GWAS_d[p], input_row_count, fused_z_checker = loaded_trait
            sumstats_format[p] = None
        else:
            # read sumstats and add suffix
            GWAS_d[p], gwas_dat_gen = _read_GWAS_sumstats(GWAS_input, args.chunksize)
            logging.info('Read in Trait {} summary statistics ({} SNPs) from {} ...'.format(p+1,len(GWAS_d[p]), GWAS_input))

            # perform munge sumstats
            GWAS_d[p], sumstats_format[p] = _perform_munge(args, GWAS_d[p], gwas_dat_gen, p)

        # generate Z checker:
        if args.use_beta_se:
            GWAS_d[p]['Z'] = GWAS_d[p][args.beta_name] / GWAS_d[p][args.se_name]
            z_checker = np.mean(np.square(GWAS_d[p]['Z']))
        elif use_fused_loader:
            z_checker = fused_z_checker
        else:
            z_checker = np.mean(np.square(GWAS_d[p][args.z_name]))

        # checker of chi2 --> error if sumstats has very low chi2
        if z_checker < 1.02 and not args.force:
            raise ValueError("The mean chi2 statistic of trait {} is less than 1.02, which may lead to unstable estimates. To perform MTAG on your results anyways, include the --force option, though the estimates should be interpreted cautiously.".format(p+1))

        # conform column names internally
        else:
            if z_checker < 1.02:
                logging.info("Warning: The mean chi2 statistic of trait {} is less 1.02 - MTAG estimates may be unstable.".format(p+1))
            if not use_fused_loader:
                GWAS_d[p].rename(
                    columns=_internal_column_renames(GWAS_d[p].columns, args),
                    inplace=True,
                )

            if not getattr(args, 'legacy_loader', False):
                core_columns = ['SNP', 'Z', 'N', 'FRQ', 'A1', 'A2']
                if _trait_needs_position_columns(args, p):
                    core_columns.extend(['CHR', 'BP'])
                if args.use_beta_se:
                    core_columns.extend(['BETA', 'SE'])
                missing_core = [
                    column for column in core_columns
                    if column not in GWAS_d[p].columns
                ]
                if missing_core:
                    raise ValueError(
                        'Missing required MTAG columns after input munging: {}'
                        .format(', '.join(missing_core))
                    )
                GWAS_d[p] = GWAS_d[p][core_columns].copy()

            GWAS_d[p] = GWAS_d[p].add_suffix(p)

        # flag inconsistency change
        if (
            not use_fused_loader
            and args.info_min_list[p] is not None
            and "INFO{}".format(p) not in GWAS_d[p].columns
        ):
            raise IOError("--info_min is specified but info column is not present in sumstats {}".format(p+1))
        if args.maf_min_list[p] is not None and "FRQ{}".format(p) not in GWAS_d[p].columns:
            raise IOError("--maf_min is specified but maf column is not present in sumstats {}".format(p+1))
        if args.n_min_list[p] is not None and "N{}".format(p) not in GWAS_d[p].columns:
            raise IOError("--n_min is specified but n column is not present in sumstats {}".format(p+1))

        # The fused loader has already normalized alleles once while applying
        # its chunk filters.
        if not use_fused_loader:
            for col in [col+str(p) for col in ['A1','A2']]:
                GWAS_d[p][col] = GWAS_d[p][col].str.upper()

        GWAS_d[p] = GWAS_d[p].rename(columns={x+str(p):x for x in GWAS_d[p].columns})
        GWAS_d[p] = GWAS_d[p].rename(columns={'SNP'+str(p):'SNP'})

        if not use_fused_loader:
            # Drop SNPs that are missing
            missing_snps = GWAS_d[p]['SNP'].isin(['NA','.'])
            M0 = len(GWAS_d[p])
            GWAS_d[p] = GWAS_d[p][np.logical_not(missing_snps)]
            if M0-len(GWAS_d[p]) > 0:
                logging.info('Trait {}: Dropped {} SNPs for missing values in the "snp_name" column'.format(p+1, M0-len(GWAS_d[p])))

            # Drop snps that are duplicated
            M0 = len(GWAS_d[p])
            GWAS_d[p] = GWAS_d[p].drop_duplicates(subset='SNP', keep='first')
            if M0-len(GWAS_d[p]) > 0:
                logging.info('Trait {}: Dropped {} SNPs for duplicate values in the "snp_name" column'.format(p+1, M0-len(GWAS_d[p])))

    #=====================
    # merge sumstats
    # intersection/union
    #=====================
    for p in range(P):
        if p == 0:
            GWAS_all = GWAS_d[p]
            GWAS_int = GWAS_all.copy()
            if args.meta_format:
                GWAS_all['Trait0'] = 1
        else:
            if args.meta_format:
                # add trait tags for all SNPs if meta
                GWAS_all = GWAS_all.merge(GWAS_d[p], how='outer', on='SNP', indicator=True)
                GWAS_all.loc[np.logical_or(GWAS_all._merge=='both',GWAS_all._merge=='right_only'),'Trait{}'.format(p)] = 1
                GWAS_all.loc[GWAS_all._merge=='left_only','Trait{}'.format(p)] = 0
                GWAS_all.loc[GWAS_all._merge=='right_only', 'Trait{}'.format(p-1)] = 0
                GWAS_all.drop(['_merge'], axis=1, inplace=True)

            # intersection only
            GWAS_int = GWAS_int.merge(GWAS_d[p], how='inner', on='SNP')
            M_0 = len(GWAS_int)
            snps_to_flip = np.logical_and(GWAS_int['A1'+str(0)] == GWAS_int['A2'+str(p)], GWAS_int['A2'+str(0)] == GWAS_int['A1'+str(p)])
            GWAS_int['flip_snps'+str(p)]= snps_to_flip

            snps_to_keep = np.logical_or(np.logical_and(GWAS_int['A1'+str(0)]==GWAS_int['A1'+str(p)], GWAS_int['A2'+str(0)]==GWAS_int['A2'+str(p)]), snps_to_flip)

            GWAS_int = GWAS_int[snps_to_keep]
            if len(GWAS_int) < M_0:
                logging.info('Dropped {} SNPs due to inconsistent allele pairs from phenotype {}. {} SNPs remain.'.format(M_0 - len(GWAS_int),p+1, len(GWAS_int)))

            if np.sum(snps_to_flip) > 0:
                zz = 'Z'
                freq_name = 'FRQ'
                GWAS_int.loc[snps_to_flip, zz+str(p)] = -1*GWAS_int.loc[snps_to_flip, zz+str(p)]
                GWAS_int.loc[snps_to_flip, freq_name + str(p)] = 1. - GWAS_int.loc[snps_to_flip, freq_name + str(p)]
                store_allele = GWAS_int.loc[snps_to_flip, 'A1'+str(p)]
                GWAS_int.loc[snps_to_flip, 'A1'+str(p)] = GWAS_int.loc[snps_to_flip, 'A2'+str(p)]
                GWAS_int.loc[snps_to_flip, 'A2'+str(p)] = store_allele
                logging.info('Flipped the signs of of {} SNPs to make them consistent with the effect allele orderings of the first trait.'.format(np.sum(snps_to_flip)))

        STRAND_AMBIGUOUS_SET = [x for x in allele_info.STRAND_AMBIGUOUS if allele_info.STRAND_AMBIGUOUS[x]]

        if p == 0 or getattr(args, 'legacy_loader', False):
            allele_pairs = GWAS_int['A1'+str(0)] + GWAS_int['A2'+str(0)]
            GWAS_int['strand_ambig'] = allele_pairs.isin(STRAND_AMBIGUOUS_SET)

        if not args.incld_ambig_snps:
            M_0 = len(GWAS_int)
            GWAS_int = GWAS_int[np.logical_not(GWAS_int['strand_ambig'])]
            logging.info('Dropped {} SNPs due to strand ambiguity, {} SNPs remain in intersection after merging trait{}'.format(M_0-len(GWAS_int),len(GWAS_int), p+1))
        else:
            logging.info('{} strand ambiguous SNPs in Trait {} are included.'.format(np.sum(GWAS_int['strand_ambig']), p+1))

    logging.info('... Merge of GWAS summary statistics complete. Number of SNPs:\t {}'.format(len(GWAS_int)))

    GWAS_orig_cols = GWAS_all.columns

    ## Parses include files
    if args.include is not None:
        for j, include_file in enumerate(args.include.split(',')):
            if j == 0:
                snps_include = _read_SNPlist(include_file, 'SNP')
            else:
                snps_include = snps_include.merge(_read_SNPlist(include_file,'SNP'),how='outer', on='SNP')
        GWAS_all = GWAS_all.merge(snps_include, how="left", on = 'SNP',  indicator="included_merge", suffixes=('','_incl'))
        GWAS_all = GWAS_all.loc[GWAS_all['included_merge']=='both']
        GWAS_all = GWAS_all.loc[:,GWAS_orig_cols]
        logging.info('(--include) Number of SNPs remaining after restricting to SNPs in the union of  {include_path}: \t {M} remain'.format(include_path=args.include,M=len(GWAS_all)))

    ## Parses exclude files
    if args.exclude is not None:
        for exclude_file in args.exclude.split(','):
            snps_exclude = _read_SNPlist(exclude_file, 'SNP')
            GWAS_all = GWAS_all.merge(snps_exclude, how="left", on = 'SNP',  indicator="excluded_merge", suffixes=('','_incl'))
            GWAS_all = GWAS_all.loc[GWAS_all['excluded_merge']=='left_only']
            GWAS_all = GWAS_all.loc[:,GWAS_orig_cols]
            logging.info('(-exclude) Number of SNPs remaining after excluding to SNPs in {exclude_path}: \t {M} remain'.format(exclude_path=exclude_file,M=len(GWAS_all)))

    ## Parse chromosomes
    if args.only_chr is not None and not args.no_chr_data:
        chr_toInclude = args.only_chr.split(',')
        chr_toInclude = [int(c) for c in chr_toInclude]
        GWAS_all = GWAS_all[GWAS_all['CHR'+str(0)].isin(chr_toInclude)]

    ## conform GWAS_int back to intersection when GWAS_all was filtered
    gwas_all_was_filtered = (
        args.include is not None
        or args.exclude is not None
        or (args.only_chr is not None and not args.no_chr_data)
    )
    if getattr(args, 'legacy_loader', False) or gwas_all_was_filtered:
        GWAS_int = GWAS_int.merge(GWAS_all[['SNP']],how='inner',on='SNP')

    ## add information to Namespace
    args.P = P

    return GWAS_all, GWAS_int, args

def ldsc_matrix_formatter(result_rg, output_var):
    ''' 
    Key Arguments:
    result_rg - matrix w/ RG objects obtained from estimate_rg (w/ None's on the diagonal)
    output_var - interested variable in the form of '.[VAR_NAME]'
    '''
    output_mat = np.empty_like(result_rg, dtype=float)
    (nrow, ncol) = result_rg.shape
    for i in range(nrow):
        for j in range(ncol):
            if result_rg[i, j] is None:
                output_mat[i, j] = None
            else:
                exec('output_mat[i, j] = result_rg[i, j]{}'.format(output_var))
    return(output_mat)

def _sigma_ldsc_args(args, sumstats_frames=None):
    """Construct the LDSC options shared by both Sigma implementations."""
    h2_files = None
    rg_files = args.sumstats
    rg_out = '{}_rg_misc'.format(args.out)
    return Namespace(
        out=rg_out, bfile=None, l2=None, extract=None, keep=None,
        ld_wind_snps=None, ld_wind_kb=None, ld_wind_cm=None,
        print_snps=None, annot=None, thin_annot=False, cts_bin=None,
        cts_break=None, cts_names=None, per_allele=False, pq_exp=None,
        no_print_annot=False, maf=None, h2=h2_files, rg=rg_files,
        ref_ld=None, ref_ld_chr=args.ld_ref_panel, w_ld=None,
        w_ld_chr=args.ld_ref_panel, overlap_annot=False,
        no_intercept=False, intercept_h2=None, intercept_gencov=None,
        M=None, two_step=None, chisq_max=None, print_cov=False,
        print_delete_vals=False, chunk_size=50, pickle=False,
        invert_anyway=False, yes_really=False, n_blocks=200,
        not_M_5_50=False, return_silly_things=False,
        no_check_alleles=False, print_coefficients=False,
        samp_prev=None, pop_prev=None, frqfile=None, h2_cts=None,
        frqfile_chr=None, print_all_cts=False,
        sumstats_frames=sumstats_frames, rg_mat=True,
    )


def _estimate_sigma_legacy(data_df, args):
    """Original pairwise pandas/LDSC wrapper retained for validation."""
    sigma_hat = np.empty((args.P,args.P))

    args.munge_out = args.out+'_ldsc_temp/'
    # Creates data files for munging
    # Munge data
    ignore_list = ""
    if args.info_min is None:
        ignore_list += "info"

    gwas_ss_df = dict()

    for p in range(args.P):
        logging.info('Preparing phenotype {} to estimate sigma'.format(p))

        ld_ss_name = {'SNP':'SNP',
                      'A1' + str(p):  'A1',
                      'A2' + str(p):  'A2',
                      'Z' + str(p):   'Z',
                      'N' + str(p):   'N',
                      'FRQ' + str(p): 'FRQ'}
        if args.use_beta_se:
            ld_ss_name['BETA' + str(p)] = 'BETA'
            ld_ss_name['SE' + str(p)] = 'SE'

        gwas_ss_df[p] = data_df[list(ld_ss_name)].copy()
        gwas_ss_df[p] = gwas_ss_df[p].rename(columns=ld_ss_name)

    # run ldsc
    args_ldsc_rg = _sigma_ldsc_args(
        args, [gwas_ss_df[i] for i in range(args.P)]
    )

    if args.no_overlap:
        sigma_hat = np.zeros((args.P, args.P))
        for t in range(args.P):
            args_ldsc_rg.sumstats_frames = [gwas_ss_df[t]]
            rg_results_t = sumstats_sig.estimate_rg(args_ldsc_rg, Logger_to_Logging())
            sigma_hat[t,t] =  ldsc_matrix_formatter(rg_results_t, '.gencov.intercept')[0]
    else:
        rg_results =  sumstats_sig.estimate_rg(args_ldsc_rg, Logger_to_Logging())

        sigma_hat = ldsc_matrix_formatter(rg_results, '.gencov.intercept')

    # if args.no_overlap:
    #     T = sigma_hat.shape[0]
    #     sigma_hat = sigma_hat * np.eye(T)

    # logging.info(type(sigma_hat))
    logging.info(sigma_hat)

    return sigma_hat


def _as_ldsc_column(values):
    """Return one contiguous float column in LDSC's expected shape."""
    return np.ascontiguousarray(values, dtype=float).reshape((-1, 1))


def estimate_sigma(data_df, args):
    """Estimate Sigma after aligning to LDSC once for all trait pairs."""
    if getattr(args, 'legacy_loader', False):
        return _estimate_sigma_legacy(data_df, args)

    start_time = time.time()
    args_ldsc_rg = _sigma_ldsc_args(args)
    trait_columns = []
    for trait_index in range(args.P):
        trait_columns.extend([
            'Z{}'.format(trait_index), 'N{}'.format(trait_index)
        ])
    wide_sumstats = data_df[['SNP'] + trait_columns]
    logger = Logger_to_Logging()
    (
        M_annot,
        w_ld_cname,
        ref_ld_cnames,
        aligned,
        _,
    ) = sumstats_sig._read_ld_sumstats(
        args_ldsc_rg,
        logger,
        None,
        alleles=False,
        dropna=True,
        sumstats=wide_sumstats,
    )
    logging.info(
        'Aligned all traits to {} LDSC regression SNPs once.'.format(
            len(aligned)
        )
    )

    ref_ld = np.ascontiguousarray(
        aligned.loc[:, ref_ld_cnames].to_numpy(dtype=float)
    )
    w_ld = _as_ldsc_column(aligned[w_ld_cname].to_numpy())
    z_columns = [
        _as_ldsc_column(aligned['Z{}'.format(p)].to_numpy())
        for p in range(args.P)
    ]
    n_columns = [
        _as_ldsc_column(aligned['N{}'.format(p)].to_numpy())
        for p in range(args.P)
    ]
    del aligned, wide_sumstats

    n_blocks = min(args_ldsc_rg.n_blocks, len(ref_ld))
    # estimate_rg converts intercept_h2 to a list before checking whether it
    # is None, so its historical automatic two-step condition is unreachable.
    # Preserve the one-step behavior MTAG actually executes.
    twostep = args_ldsc_rg.two_step
    intercept_h2 = [None] * args.P
    intercept_gencov = [None] * args.P

    # The original RG wrapper recomputed both Hsq models for every pair.
    # With MTAG's already-intersected rows, each trait has the same LDSC SNP
    # set, so one regression per trait is identical to the repeated versions.
    hsq = []
    for trait_index in range(args.P):
        logging.info(
            'Estimating LDSC heritability for phenotype {}.'.format(
                trait_index + 1
            )
        )
        hsq.append(ldsc_reg.Hsq(
            np.square(z_columns[trait_index]),
            ref_ld,
            w_ld,
            n_columns[trait_index],
            M_annot,
            n_blocks=n_blocks,
            intercept=intercept_h2[trait_index],
            twostep=twostep,
        ))

    sigma_hat = np.zeros((args.P, args.P), dtype=float)
    for first in range(args.P):
        last = first + 1 if args.no_overlap else args.P
        for second in range(first, last):
            logging.info(
                'Estimating LDSC residual covariance for phenotypes {} and '
                '{}.'.format(first + 1, second + 1)
            )
            gencov = ldsc_reg.Gencov(
                z_columns[first],
                z_columns[second],
                ref_ld,
                w_ld,
                n_columns[first],
                n_columns[second],
                M_annot,
                hsq[first].tot,
                hsq[second].tot,
                hsq[first].intercept,
                hsq[second].intercept,
                n_blocks=n_blocks,
                intercept_gencov=intercept_gencov[second],
                twostep=twostep,
            )
            covariance = float(np.asarray(gencov.intercept).item())
            sigma_hat[first, second] = covariance
            sigma_hat[second, first] = covariance

    logging.info(
        'Shared-alignment Sigma estimation took {:.3f} seconds.'.format(
            time.time() - start_time
        )
    )
    logging.info(sigma_hat)
    return sigma_hat

def _posDef_adjustment(mat, scaling_factor=0.99,max_it=1000):
    '''
    Checks whether the provided is pos semidefinite. If it is not, then it performs the the adjustment procedure descried in 1.2.2 of the Supplementary Note

    scaling_factor: the multiplicative factor that all off-diagonal elements of the matrix are scaled by in the second step of the procedure.
    max_it: max number of iterations set so that
    '''
    logging.info('Checking for positive definiteness ..')
    assert mat.ndim == 2
    assert mat.shape[0] == mat.shape[1]
    is_pos_semidef = lambda m: np.all(np.linalg.eigvals(m) >= 0)
    if is_pos_semidef(mat):
        return mat
    else:
        logging.info('matrix is not positive definite, performing adjustment..')
        P = mat.shape[0]
        for i in range(P):
            for j in range(i,P):
                if np.abs(mat[i,j]) > np.sqrt(mat[i,i] * mat[j,j]):
                    mat[i,j] = scaling_factor*np.sign(mat[i,j])*np.sqrt(mat[i,i] * mat[j,j])
                    mat[j,i] = mat[i,j]
        n=0
        while not is_pos_semidef(mat) and n < max_it:
            dg = np.diag(mat)
            mat = scaling_factor * mat
            mat[np.diag_indices(P)] = dg
            n += 1
        if n == max_it:
            logging.info('Warning: max number of iterations reached in adjustment procedure. Sigma matrix used is still non-positive-definite.')
        else:
            logging.info('Completed in {} iterations'.format(n))
        return mat

def extract_gwas_sumstats(DATA, args, t0):
    '''

    Output:
    -------
    All matrices are of the shape MxP, where M is the number of SNPs used in MTAG and P is the number of summary statistics results used. Columns are ordered according to the initial ordering of GWAS input files.
    results_template = pd.Dataframe of snp_name chr bpos a1 a2
    Zs: matrix of Z scores
    Ns: matrix of sample sizes
    Fs: matrix of allele frequencies
    '''
    n_cols = ['N' +str(p) for p in t0]
    Ns = DATA.filter(items=n_cols).to_numpy()

    # Apply sample-size specific filters

    N_passFilter = np.ones(len(Ns), dtype=bool)

    N_nearMode = np.ones_like(Ns, dtype=bool)
    if args.homogNs_frac is not None or args.homogNs_dist is not None:
        N_modes, _ = _quick_mode(Ns)
        assert len(N_modes) == Ns.shape[1]
        if args.homogNs_frac is not None:
            logging.info('--homogNs_frac {} is on, filtering SNPs ...'.format(args.homogNs_frac))
            assert args.homogNs_frac >= 0.
            homogNs_frac_list = [float(x) for x in args.homogNs_frac.split(',')]
            if len(homogNs_frac_list) == 1:
                homogNs_frac_list = homogNs_frac_list*args.P
            for p in t0:
                N_nearMode[:,p] = np.abs((Ns[:,p] - N_modes[p])) / N_modes[p] <= homogNs_frac_list[p]
        elif args.homogNs_dist is not None:
            logging.info('--homogNs_dist {} is on, filtering SNPs ...'.format(args.homogNs_dist))
            homogNs_dist_list = [float(x) for x in args.homogNs_dist.split(',')]
            if len(homogNs_dist_list) == 1:
                homogNs_dist_list = homogNs_dist_list*args.P

            assert np.all(np.array(homogNs_dist_list) >=0)
            for p in t0:
                N_nearMode[:,p] =  np.abs(Ns[:,p] - N_modes[p]) <= homogNs_dist_list[p]
        else:
            raise ValueError('Cannot specify both --homogNs_frac and --homogNs_dist at the same time.')

        # report restrictions
        mode_restrictions = 'Sample size restrictions close to mode:\n'
        for p in range(Ns.shape[1]):
            mode_restrictions +="Phenotype {}: \t {} SNPs pass modal sample size filter \n".format(p+1,np.sum(N_nearMode[:,p]))

        mode_restrictions+="Intersection of SNPs that pass modal sample size filter for all traits:\t {}".format(np.sum(np.all(N_nearMode, axis=1)))
        logging.info(mode_restrictions)
        N_passFilter = np.logical_and(N_passFilter, np.all(N_nearMode,axis=1))

    if args.n_max is not None:
        n_max_restrictions = "--n_max used, removing SNPs with sample size greater than  {}".format(args.n_max)
        N_passMax = Ns <= args.n_max
        for p in range(Ns.shape[1]):
            n_max_restrictions +=  "Phenotype {}: \t {} SNPs pass modal sample size filter".format(p+1,np.sum(N_passMax[:,p]))
        n_max_restrictions += "Intersection of SNPs that pass maximum sample size filter for all traits:\t {}".format(np.sum(np.all(N_passMax, axis=1)))
        logging.info(n_max_restrictions)
        N_passFilter = np.logical_and(N_passFilter, np.all(N_passMax,axis=1))

    Ns = Ns[N_passFilter]
    DATA = DATA[N_passFilter].reset_index()
    N_raw = np.copy(Ns)
    f_cols = ['FRQ'+ str(p) for p in t0]
    Fs = DATA.filter(items=f_cols).to_numpy()

    if args.use_beta_se:
        beta_cols = ['BETA'+str(p) for p in t0]
        se_cols = ['SE'+str(p) for p in t0]
        BETAs = DATA.filter(items=beta_cols).to_numpy()
        SEs = DATA.filter(items=se_cols).to_numpy()

        # standardizing factor
        std_factor = np.sqrt(2*Fs*(1-Fs))
        Zs = BETAs / SEs
        SEs = np.multiply(SEs, std_factor)
        Ns = 1 / np.square(SEs)
    else:
        z_cols = ['Z'+str(p) for p in t0]
        Zs = DATA.filter(items=z_cols).to_numpy()

    assert Zs.shape[1] == Ns.shape[1] == Fs.shape[1]

    results_template = DATA[['SNP']].copy()

    if args.no_chr_data:
        for col in ['A1','A2']:
           results_template.loc[:,col] = DATA[col+str(t0[0])]
    else:
        for col in ['CHR','BP','A1','A2']:
            results_template.loc[:,col] = DATA[col+str(t0[0])]

    # TODO: non-error form of integer conversion
    # results_template[args.chr_name] = results_template[args.chr_name].astype(int)
    # results_template[args.bpos_name] = results_template[args.bpos_name].astype(int)

    return Zs, Ns, Fs, results_template, DATA, N_raw

###########################################
## OMEGA ESTIMATION
##########################################

def jointEffect_probability(Z_score, omega_hat, sigma_hat,N_mats, S=None):
    ''' For each SNP m in each state s , computes the evaluates the multivariate normal distribution at the observed row of Z-scores
    Calculate the distribution of (Z_m | s ) for all s in S, m in M. --> M  x|S| matrix
    The output is a M x n_S matrix of joint probabilities
    '''

    DTYPE = np.float64
    (M,P) = Z_score.shape
    if S is None: # 2D dimensional form
        assert omega_hat.ndim == 2
        omega_hat = omega_hat.reshape(1,P,P)
        S = np.ones((1,P),dtype=bool)

    (n_S,_) = S.shape
    jointProbs = np.empty((M,n_S))

    xRinvs = np.zeros([M,n_S,P], dtype=DTYPE)
    logSqrtDetSigmas = np.zeros([M,n_S], dtype=DTYPE)
    Ls = np.zeros([M,n_S,P,P], dtype=DTYPE)
    cov_s = np.zeros([M,n_S,P,P], dtype=DTYPE)

    Zs_rep = np.einsum('mp,s->msp',Z_score,np.ones(n_S))  # functionally equivalent to repmat
    cov_s = np.einsum('mpq,spq->mspq',N_mats,omega_hat) + sigma_hat

    Ls = np.linalg.cholesky(cov_s)
    Rs = np.transpose(Ls, axes=(0,1,3,2))

    xRinvs = np.linalg.solve(Ls, Zs_rep)

    logSqrtDetSigmas = np.sum(np.log(np.diagonal(Rs,axis1=2,axis2=3)),axis=2).reshape(M,n_S)

    quadforms = np.sum(xRinvs**2,axis=2).reshape(M,n_S)
    jointProbs = np.exp(-0.5 * quadforms - logSqrtDetSigmas - P * np.log(2 * np.pi) / 2)

    if n_S == 1:
        jointProbs = jointProbs.flatten()

    return jointProbs

def gmm_omega(Zs, Ns, sigma_LD):
    logging.info('Using GMM estimator of Omega ..')
    N_mats = np.sqrt(np.einsum('mp,mq->mpq', Ns,Ns))
    Z_outer = np.einsum('mp,mq->mpq',Zs, Zs)
    return np.mean((Z_outer - sigma_LD) / N_mats, axis=0)

def numerical_omega(args, Zs,N_mats,sigma_LD,omega_start):
    M,P = Zs.shape
    solver_options = dict()
    solver_options['fatol'] = 1.0e-8
    solver_options['xatol'] = args.tol
    solver_options['disp'] = False
    solver_options['maxiter'] = P*250 if args.perfect_gencov else P*(P+1)*500
    if args.perfect_gencov:
        x_start = np.log(np.diag(omega_start))
    else:
        x_start = flatten_out_omega(omega_start)

    opt_results = scipy.optimize.minimize(_omega_neglogL,x_start,args=(Zs,N_mats,sigma_LD,args),method='Nelder-Mead',options=solver_options)

    if args.perfect_gencov:
        return np.sqrt(np.outer(np.exp(opt_results.x), np.exp(opt_results.x))), opt_results
    else:
        return rebuild_omega(opt_results.x), opt_results

def _omega_neglogL(x,Zs,N_mats,sigma_LD,args):
    if args.perfect_gencov:
        omega_it = np.sqrt(np.outer(np.exp(x),np.exp(x)))
    else:
        omega_it = rebuild_omega(x)
    joint_prob = jointEffect_probability(Zs,omega_it,sigma_LD,N_mats)
    return - np.sum(np.log(joint_prob))

def flatten_out_omega(omega_est):
    # stacks the lower part of the cholesky decomposition ROW_WISE [(0,0) (1,0) (1,1) (2,0) (2,1) (2,2) ...]
    P_c = len(omega_est)
    x_chol = np.linalg.cholesky(omega_est)

    # transform components of cholesky decomposition for better optimization
    lowTr_ind = np.tril_indices(P_c)
    x_chol_trf = np.zeros((P_c,P_c))
    for i in range(P_c):
        for j in range(i): # fill in lower triangular components not on diagonal
            x_chol_trf[i,j] = x_chol[i,j]/np.sqrt(x_chol[i,i]*x_chol[j,j])
    x_chol_trf[np.diag_indices(P_c)] = np.log(np.diag(x_chol))  # replace with log transformation on the diagonal
    return tuple(x_chol_trf[lowTr_ind])

def rebuild_omega(chol_elems, s=None):
    '''Rebuild state-dependent Omega given combination of causal states
       cholX_elements are the elements (entered row-wise) of the lower triangular cholesky decomposition of Omega_s

    '''
    if s is None:
        P = int((-1 + np.sqrt(1.+ 8.*len(chol_elems)))/2.)
        s = np.ones(P,dtype=bool)
        P_c = P
    else:
        P_c = int(np.sum(s))
        P = s.shape[1] if s.ndim == 2 else len(s)
    cholL = np.zeros((P_c,P_c))

    cholL[np.tril_indices(P_c)] = np.array(chol_elems)
    cholL[np.diag_indices(P_c)] = np.exp(np.diag(cholL))  # exponentiate the diagnoal so cholL unique
    for i in range(P_c):
        for j in range(i): # multiply by exponentiated diags
            cholL[i,j] = cholL[i,j]*np.sqrt(cholL[i,i]*cholL[j,j])

    omega_c = np.dot(cholL, cholL.T)

    # Expand to include zeros of matrix
    omega = np.zeros((P,P))
    s_caus_ind = np.argwhere(np.outer(s, s))
    omega[(s_caus_ind[:,0],s_caus_ind[:,1])] = omega_c.flatten()
    return omega

def estimate_omega(args,Zs,Ns,sigma_LD, omega_in=None):


    # start_time =time.time()
    logging.info('Beginning estimation of Omega ...')

    M,P = Zs.shape
    N_mats = np.sqrt(np.einsum('mp, mq -> mpq',Ns, Ns))


    if args.perfect_gencov and args.equal_h2:
        logging.info('--perfect_gencov and --equal_h2 option used')
        return np.ones((P,P))

    if args.numerical_omega:
        if omega_in is None: # omega_in serves as starting point
            omega_in = np.zeros((P,P))
            omega_in[np.diag_indices(P)] = np.diag(gmm_omega(Zs,Ns,sigma_LD))

        omega_hat = omega_in

        omega_hat, opt_results = numerical_omega(args, Zs,N_mats, sigma_LD,omega_hat)
        numerical_msg = "\n Numerical optimization of Omega complete:"
        numerical_msg += "\nSuccessful termination? {}".format("Yes" if opt_results.success else "No")
        numerical_msg += "\nTermination message:\t{}".format(opt_results.message)
        numerical_msg += "\nCompleted in {} iterations".format(opt_results.nit)
        logging.info(numerical_msg)
        return omega_hat


    if args.perfect_gencov:
        omega_hat = _posDef_adjustment(gmm_omega(Zs,Ns,sigma_LD))
        return np.sqrt(np.outer(np.diag(omega_hat), np.diag(omega_hat)))

    # else: gmm_omega (default)
    return _posDef_adjustment(gmm_omega(Zs,Ns,sigma_LD))

def cov2corr(cov, return_std=False):
    '''
    convert covariance matrix to correlation matrix
    '''
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    return corr

########################
## MTAG CALCULATION ####
########################

def _sample_size_adjusted_covariance(Ns, covariance):
    """Scale a trait covariance matrix by per-row sample sizes."""
    inv_sqrt_n = 1.0 / np.sqrt(Ns)
    return (
        covariance[None, :, :]
        * inv_sqrt_n[:, :, None]
        * inv_sqrt_n[:, None, :]
    )


def _mtag_analysis_batch(Zs, Ns, omega_hat, sigma_LD):
    """Run the main MTAG estimator for one in-memory SNP batch."""
    M, P = Zs.shape
    Sigma_N = _sample_size_adjusted_covariance(Ns, sigma_LD)
    W_inv_Z = Zs / np.sqrt(Ns)

    mtag_betas = np.empty((M, P), dtype=float)
    mtag_se = np.empty((M, P), dtype=float)
    mtag_factor = np.empty((M, P), dtype=float)

    for p in range(P):
        # Note that in the code, what I call "gamma" should really be omega,
        # but avoid the latter term due to possible confusion with big Omega.
        gamma_k = omega_hat[:, p]
        tau_k_2 = omega_hat[p, p]
        om_min_gam = omega_hat - np.outer(gamma_k, gamma_k) / tau_k_2
        xx = om_min_gam[None, :, :] + Sigma_N
        yy = gamma_k / tau_k_2

        # The legacy implementation formed yy.T @ inv(xx). Solving the
        # transposed system yields the same vector without constructing a
        # complete inverse for every SNP.
        rhs = np.broadcast_to(yy, (M, P)).copy()[:, :, None]
        weighted = np.linalg.solve(np.swapaxes(xx, 1, 2), rhs)[:, :, 0]
        beta_denom = np.einsum('mp,p->m', weighted, yy)

        mtag_factor[:, p] = np.einsum(
            'mp,m->m', weighted, 1.0 / beta_denom
        )
        mtag_betas[:, p] = (
            np.einsum('mp,mp->m', weighted, W_inv_Z) / beta_denom
        )
        mtag_se[:, p] = np.sqrt(1.0 / beta_denom)

    return mtag_betas, mtag_se, mtag_factor


def mtag_analysis(Zs, Ns, omega_hat, sigma_LD, batch_size=100000):
    logging.info('Beginning MTAG calculations...')
    Zs = np.asarray(Zs, dtype=float)
    Ns = np.asarray(Ns, dtype=float)
    omega_hat = np.asarray(omega_hat, dtype=float)
    sigma_LD = np.asarray(sigma_LD, dtype=float)
    if batch_size <= 0:
        raise ValueError('MTAG batch size must be a positive integer')
    if Zs.ndim != 2 or Ns.shape != Zs.shape:
        raise ValueError('MTAG Z scores and sample sizes must be matching matrices')
    M, P = Zs.shape
    if omega_hat.shape != (P, P) or sigma_LD.shape != (P, P):
        raise ValueError(
            'MTAG covariance matrix dimensions must match the number of traits'
        )

    mtag_betas = np.empty((M, P), dtype=float)
    mtag_se = np.empty((M, P), dtype=float)
    mtag_factor = np.empty((M, P), dtype=float)

    for start in range(0, M, batch_size):
        stop = min(start + batch_size, M)
        batch_results = _mtag_analysis_batch(
            Zs[start:stop], Ns[start:stop], omega_hat, sigma_LD
        )
        mtag_betas[start:stop] = batch_results[0]
        mtag_se[start:stop] = batch_results[1]
        mtag_factor[start:stop] = batch_results[2]

    logging.info(' ... Completed MTAG calculations.')
    return mtag_betas, mtag_se, mtag_factor


####################
## SAVING RESULTS ##
####################

def save_mtag_results(args,results_template,Zs,Ns,Fs,mtag_betas,mtag_se,mtag_factor):
    '''
    Output will be of the form:

    snp_name z n maf mtag_beta mtag_se mtag_zscore mtag_pval

   '''
    p_values = lambda z: 2*(scipy.stats.norm.cdf(-1.*np.abs(z)))

    M,P  = mtag_betas.shape

    if args.std_betas:
        logging.info('Outputting standardized betas..')

    # meta-analysis mode
    if args.equal_h2 and args.perfect_gencov:
        logging.info('With meta-analysis mode, MTAG produces a single set of sumstats, where betas are unstandardized using 2p(1-p) where p is the average allele frequencies across traits.')
        Fs = np.mean(Fs, axis=1)

        if args.std_betas:
            weights = np.ones(M,dtype=float)
        else:
            weights = np.sqrt( 2*Fs*(1.-Fs))

        # check betas and se are identical in all columns
        for p in range(1,P):
            for col in ['mtag_betas','mtag_se']:
                if not np.all(mtag_betas[:,p] == mtag_betas[:,0]):
                    raise ValueError('Meta-analysis mode is not implemented correctly')

        # output meta-analysis results
        logging.info('Writing Meta-analysis results to file ...')
        out_df = results_template.copy()
        out_df['meta_freq'] = Fs
        out_df['mtag_beta'] = mtag_betas[:,0] / weights
        out_df['mtag_se'] = mtag_se[:,0] / weights
        out_df['mtag_z'] = mtag_betas[:,0]/mtag_se[:,0]
        out_df['mtag_pval'] = p_values(out_df['mtag_z'])
        out_path = args.out +'_mtag_meta.txt'
        _write_output_frame(args, out_df, out_path)

    else:
        for p in range(P):
            logging.info('Writing Phenotype {} to file ...'.format(p+1))
            out_df = results_template.copy()
            out_df['Z'] = Zs[:,p]
            out_df['N'] = Ns[:,p]
            out_df['FRQ'] = Fs[:,p]

            if args.std_betas:
                weights = np.ones(M,dtype=float)
            else:
                weights = np.sqrt( 2*Fs[:,p]*(1. - Fs[:,p]))
            out_df['mtag_beta'] = mtag_betas[:,p] / weights
            out_df['mtag_se'] = mtag_se[:,p] / weights

            out_df['mtag_z'] = mtag_betas[:,p]/mtag_se[:,p]
            out_df['mtag_pval'] = p_values(out_df['mtag_z'])

            if P == 1:
                out_path = args.out +'_trait.txt'
            else:
                out_path = args.out +'_trait_' + str(p+1) + '.txt'

            _write_output_frame(args, out_df, out_path)

def write_summary(args,Zs,Ns,Fs,mtag_betas,mtag_se,mtag_factor):
    '''
    Note that in the current version, Ns is the full dataframe under the meta_format mode.
    '''

    _,P = mtag_factor.shape

    if not args.equal_h2:
        omega_out = "\nEstimated Omega:\n"
        omega_out += str(args.omega_hat)
        omega_out += '\n'

        omega_out += "\n(Correlation):\n"
        omega_out += str(cov2corr(args.omega_hat))
        omega_out += '\n'

        np.savetxt(args.out +'_omega_hat.txt',args.omega_hat, delimiter ='\t')
    else:
        omega_out = "Omega hat not computed because --equal_h2 was used.\n"

    sigma_out = "\nEstimated Sigma:\n"
    sigma_out += str(args.sigma_hat)
    sigma_out += '\n'

    sigma_out += "\n(Correlation):\n"
    sigma_out += str(cov2corr(args.sigma_hat))
    sigma_out += '\n'

    np.savetxt(args.out +'_sigma_hat.txt',args.sigma_hat, delimiter ='\t')

    weight_out = "\nMTAG weight factors: (average across SNPs)\n"
    weight_out += str(np.mean(mtag_factor,axis=0))
    weight_out += '\n'

    summary_df = pd.DataFrame(index=np.arange(1,P+1))
    input_phenotypes = [ '...'+f[-16:] if len(f) > 20 else f for f in args.sumstats.split(',')]

    for p in range(P):

        summary_df.loc[p+1,'Trait'] = input_phenotypes[p]
        summary_df.loc[p+1, '# SNPs used'] = int(len(Zs[:,p]))

        if args.meta_format:
            comb_df_extract = [Ns[y][x] for y in Ns for x in Ns[y] if x==p]
            out_df = pd.concat(comb_df_extract, axis=0)
            summary_df.loc[p+1, 'N (max)'] = np.max(out_df[args.n_name])
            summary_df.loc[p+1, 'N (mean)'] = np.mean(out_df[args.n_name])
        else:
            summary_df.loc[p+1, 'N (max)'] = np.max(Ns[:,p])
            summary_df.loc[p+1, 'N (mean)'] = np.mean(Ns[:,p])

        summary_df.loc[p+1, 'GWAS mean chi^2'] = np.mean(np.square(Zs[:,p])) / args.sigma_hat[p,p]
        Z_mtag = mtag_betas[:,p]/mtag_se[:,p]
        summary_df.loc[p+1, 'MTAG mean chi^2'] = np.mean(np.square(Z_mtag))
        summary_df.loc[p+1, 'GWAS equiv. (max) N'] = int(summary_df.loc[p+1, 'N (max)']*(summary_df.loc[p+1, 'MTAG mean chi^2'] - 1) / (summary_df.loc[p+1, 'GWAS mean chi^2'] - 1))

    summary_df['N (max)'] = summary_df['N (max)'].astype(int)
    summary_df['N (mean)'] = summary_df['N (mean)'].astype(int)
    summary_df['# SNPs used'] = summary_df['# SNPs used'].astype(int)
    summary_df['GWAS equiv. (max) N'] = summary_df['GWAS equiv. (max) N'].astype(int)

    final_summary = "\nSummary of MTAG results:\n"
    final_summary +="------------------------\n"
    final_summary += summary_df.round(3).to_string()+'\n'
    final_summary += omega_out
    final_summary += sigma_out
    final_summary += weight_out

    logging.info(final_summary)
    logging.info(' ')
    logging.info('MTAG results saved to file.')

def save_mtag_results_U(args, comb_df):
    '''
    Concatenate mtag results by subtypes and write to files
    '''    
    for p in range(args.P):
        logging.info('Writing Phenotype {} to file...'.format(p+1))
        comb_df_extract = [comb_df[y][x] for y in comb_df for x in comb_df[y] if x==p]
        out_df = pd.concat(comb_df_extract, axis=0)
        M_0 = out_df.shape[0]

        if M_0 - out_df.shape[0] != 0:
            raise ValueError('--meta_format option was not implemented correctly.')

        out_path = args.out +'_trait_' + str(p+1) + '.txt'
        _write_output_frame(args, out_df, out_path, na_rep='NA')

## maxFDR Functions
MAXFDR_INPUTS_FORMAT_VERSION = 1
MAXFDR_INPUTS_SUFFIX = '_maxfdr_inputs.npz'


def _maxfdr_inputs_path(output_prefix):
    return output_prefix + MAXFDR_INPUTS_SUFFIX


def _write_maxfdr_inputs(args, Ns):
    """Save the sufficient sample-size statistics for default maxFDR."""
    sample_sizes = np.asarray(Ns, dtype=float)
    if (
        sample_sizes.ndim != 2
        or sample_sizes.shape[0] == 0
        or sample_sizes.shape[1] == 0
        or not np.all(np.isfinite(sample_sizes))
        or np.any(sample_sizes <= 0.0)
    ):
        raise ValueError(
            'Cannot save maxFDR inputs from an invalid sample-size matrix'
        )

    # fdr() historically rounds every SNP's N before taking trait means.
    # Saving that already-aggregated value avoids both a numerical change and
    # a later scan of every multi-gigabyte trait output.
    n_approx = np.mean(np.round(sample_sizes), axis=0)
    output_path = _maxfdr_inputs_path(args.out)
    np.savez(
        output_path,
        format_version=np.asarray(MAXFDR_INPUTS_FORMAT_VERSION, dtype=np.int64),
        n_approx=n_approx,
        n_snps=np.asarray(sample_sizes.shape[0], dtype=np.int64),
    )
    logging.info(
        'Saved compact inputs for a separate default maxFDR run to {}'.format(
            output_path
        )
    )


def _load_maxfdr_inputs(output_prefix):
    """Load and validate the default maxFDR sufficient-statistics sidecar."""
    input_path = _maxfdr_inputs_path(output_prefix)
    with np.load(input_path, allow_pickle=False) as saved:
        required = {'format_version', 'n_approx', 'n_snps'}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(
                '{} is missing required maxFDR fields: {}'.format(
                    input_path, ', '.join(sorted(missing))
                )
            )
        version = np.asarray(saved['format_version'])
        n_approx = np.asarray(saved['n_approx'], dtype=float)
        n_snps = np.asarray(saved['n_snps'])

    if (
        version.size != 1
        or int(version.reshape(-1)[0]) != MAXFDR_INPUTS_FORMAT_VERSION
    ):
        raise ValueError(
            '{} uses an unsupported maxFDR input format version'.format(
                input_path
            )
        )
    if n_snps.size != 1 or int(n_snps.reshape(-1)[0]) <= 0:
        raise ValueError('{} contains an invalid SNP count'.format(input_path))
    if (
        n_approx.ndim != 1
        or n_approx.size == 0
        or not np.all(np.isfinite(n_approx))
        or np.any(n_approx <= 0.0)
    ):
        raise ValueError(
            '{} contains invalid maxFDR sample-size means'.format(input_path)
        )
    return n_approx.reshape(1, -1)


def _mtag_trait_output_paths(output_prefix):
    out_dir, out_file = os.path.split(output_prefix)
    out_dir = out_dir or '.'
    trait_pattern = re.compile(
        r'^{}_trait_(\d+)\.txt$'.format(re.escape(out_file))
    )
    trait_numbers = sorted(
        int(match.group(1))
        for name in os.listdir(out_dir)
        for match in [trait_pattern.match(name)]
        if match is not None
    )
    if not trait_numbers:
        raise ValueError(
            'No MTAG trait output files found for --out {}'.format(
                output_prefix
            )
        )
    expected = list(range(1, trait_numbers[-1] + 1))
    if trait_numbers != expected:
        raise ValueError(
            'MTAG trait output files for --out {} are not consecutively '
            'numbered from 1'.format(output_prefix)
        )
    return [
        '{}_trait_{}.txt'.format(output_prefix, trait_number)
        for trait_number in trait_numbers
    ]


def _load_skip_mtag_sumstats(args):
    """Load only the result columns required by the requested maxFDR mode."""
    sidecar_path = _maxfdr_inputs_path(args.out)
    if args.n_approx and not args.fit_ss and os.path.isfile(sidecar_path):
        logging.info(
            'Loading compact default maxFDR inputs from {}'.format(
                sidecar_path
            )
        )
        return _load_maxfdr_inputs(args.out), None, True

    trait_paths = _mtag_trait_output_paths(args.out)
    need_z = args.fit_ss
    columns = ['N', 'Z'] if need_z else ['N']

    if args.n_approx and not need_z:
        if not os.path.isfile(sidecar_path):
            logging.info(
                'No compact maxFDR input file was found; scanning N from '
                'legacy MTAG trait outputs.'
            )
        n_approx = []
        for trait_path in trait_paths:
            trait_n = pd.read_csv(
                trait_path, index_col=None, sep=r'\s+', usecols=columns
            )['N'].to_numpy(dtype=float)
            if trait_n.size == 0:
                raise ValueError(
                    'MTAG trait output contains no SNPs: {}'.format(trait_path)
                )
            n_approx.append(np.mean(np.round(trait_n)))
        return np.asarray(n_approx, dtype=float).reshape(1, -1), None, True

    n_columns = []
    z_columns = []
    expected_rows = None
    for trait_path in trait_paths:
        trait_data = pd.read_csv(
            trait_path, index_col=None, sep=r'\s+', usecols=columns
        )
        if expected_rows is None:
            expected_rows = len(trait_data)
        elif len(trait_data) != expected_rows:
            raise ValueError('MTAG trait output files have unequal row counts')
        n_columns.append(trait_data['N'].to_numpy(dtype=float))
        if need_z:
            z_columns.append(trait_data['Z'].to_numpy(dtype=float))

    n_matrix = np.column_stack(n_columns)
    z_matrix = np.column_stack(z_columns) if need_z else None
    return n_matrix, z_matrix, False


create_S = lambda P: np.asarray(list(itertools.product([False,True], repeat=P)))

def MTAG_var_Z_jt_c(t, Omega, Omega_c, sigma_LD, Ns):

    '''
    Omega: full Omega matrix
    Omega_c: conditional Omega
    Sigma_LD
    N_mean: vector of length of "sample sizes" (1/c**2).

    This formula only works with constant N, etc.
    '''


    Sigma_j = _sample_size_adjusted_covariance(Ns, sigma_LD)

    gamma_k = Omega[:,t]
    tau_k_2 = Omega[t,t]

    om_min_gam = Omega - np.outer(gamma_k, gamma_k) / tau_k_2
    xx = om_min_gam + Sigma_j
    inv_xx = np.linalg.inv(xx)

    # num_L / R are the same due to symmetry
    num_L = np.einsum('p,mpq->mq', gamma_k / tau_k_2, inv_xx)
    num_R = np.einsum('mpq,q->mp', inv_xx, gamma_k / tau_k_2)


    numer = np.einsum('mp,mp->m', num_L, np.einsum('mpq,mq->mp', Omega_c + Sigma_j, num_R))

    denom = np.einsum('p,mp->m', gamma_k / tau_k_2, np.einsum('mpq,q->mp', inv_xx, gamma_k /tau_k_2))

    return numer / denom

def simplex_walk(num_dims, samples_per_dim):
    """
    A generator that returns lattice points on an n-simplex.
    """
    max_ = samples_per_dim + num_dims - 1
    for c in itertools.combinations(range(max_), num_dims):
        #logging.info(c)
        c = list(c)
        yield np.array([(y - x - 1.) / (samples_per_dim - 1.)
               for x, y in zip([-1] + c, c + [max_])])


def _causal_pair_probabilities(priors, S):
    """Return joint causal probabilities for every pair of traits."""
    states = np.asarray(S, dtype=float)
    priors = np.asarray(priors, dtype=float)
    return states.T @ (priors[:, None] * states)


def scale_omega(gen_corr_mat, priors, S=None):
    gen_corr_mat = np.asarray(gen_corr_mat, dtype=float)
    priors = np.asarray(priors, dtype=float)
    if gen_corr_mat.ndim != 2 or gen_corr_mat.shape[0] != gen_corr_mat.shape[1]:
        raise ValueError('genetic covariance matrix must be square')
    T = gen_corr_mat.shape[1]
    if S is None:
        S = create_S(T)
    return gen_corr_mat / _causal_pair_probabilities(priors, S)


def load_probability_grid(file_path, n_states, sum_tolerance=1.0e-8):
    """Load and validate a user-supplied maxFDR probability grid."""
    prob_grid = np.atleast_2d(np.asarray(np.loadtxt(file_path), dtype=float))
    if prob_grid.shape[1] != n_states:
        raise ValueError(
            'maxFDR grid file must contain {} columns (one for each causal '
            'state); found {}'.format(n_states, prob_grid.shape[1])
        )

    valid_rows = (
        np.all(np.isfinite(prob_grid), axis=1)
        & np.all(prob_grid >= 0.0, axis=1)
        & np.all(prob_grid <= 1.0, axis=1)
        & np.isclose(
            np.sum(prob_grid, axis=1),
            1.0,
            rtol=0.0,
            atol=sum_tolerance,
        )
    )
    excluded = int(np.sum(~valid_rows))
    if excluded:
        logging.info(
            'Excluded {} maxFDR grid rows with invalid probabilities.'.format(
                excluded
            )
        )
    prob_grid = prob_grid[valid_rows]
    if len(prob_grid) == 0:
        raise ValueError(
            'maxFDR grid file contains no valid probability rows; each row '
            'must contain finite values in [0, 1] that sum to 1'
        )
    return prob_grid

def _prepare_fdr_calculation(omega, sigma, Ns, N_counts, p_threshold):
    """Precompute maxFDR terms invariant across grid points and states."""
    sigma_by_n = _sample_size_adjusted_covariance(Ns, sigma)
    trait_terms = []
    for t in range(omega.shape[0]):
        gamma_k = omega[:, t]
        tau_k_2 = omega[t, t]
        yy = gamma_k / tau_k_2
        omega_conditional = omega - np.outer(gamma_k, gamma_k) / tau_k_2
        inverse = np.linalg.inv(omega_conditional[None, :, :] + sigma_by_n)
        num_left = np.einsum('p,mpq->mq', yy, inverse)
        num_right = np.einsum('mpq,q->mp', inverse, yy)
        denominator = np.einsum('p,mp->m', yy, num_right)
        sigma_numerator = np.einsum(
            'mp,mp->m',
            num_left,
            np.einsum('mpq,mq->mp', sigma_by_n, num_right),
        )
        trait_terms.append(
            (num_left, num_right, denominator, sigma_numerator)
        )

    return {
        'n_counts': np.asarray(N_counts, dtype=float),
        'n_total': float(np.sum(N_counts)),
        'trait_terms': trait_terms,
        'z_threshold': scipy.stats.norm.isf(p_threshold / 2.0),
    }


def _compute_fdr_values(prob, omega, S, prepared):
    """Compute maxFDR for every trait at one causal-state grid point."""
    omega_TT = scale_omega(omega, prob, S)
    T = S.shape[1]
    if not is_pos_semidef(omega_TT):
        return np.full(T, np.inf)

    omega_by_state = np.einsum('st,sr->str', S, S) * omega_TT
    fdr_values = np.empty(T, dtype=float)
    for t, trait_terms in enumerate(prepared['trait_terms']):
        num_left, num_right, denominator, sigma_numerator = trait_terms
        omega_numerator = np.einsum(
            'mp,spq,mq->sm',
            num_left,
            omega_by_state,
            num_right,
            optimize=True,
        )
        variances = (
            omega_numerator + sigma_numerator[None, :]
        ) / denominator[None, :]
        sd = np.sqrt(variances)
        probability_significant = np.sum(
            2.0
            * scipy.stats.norm.sf(
                prepared['z_threshold'], loc=0, scale=sd
            )
            * prepared['n_counts'][None, :],
            axis=1,
        ) / prepared['n_total']
        power_by_state = probability_significant * prob
        total_power = np.sum(power_by_state)
        if not np.isfinite(total_power) or total_power <= 0.0:
            raise ValueError(
                'maxFDR could not calculate positive finite discovery power '
                'for the supplied grid point'
            )
        fdr_values[t] = np.sum(power_by_state[~S[:, t]]) / total_power

    return fdr_values


def compute_fdr(prob, t, omega, sigma, S, Ns, N_counts, p_threshold):
    """Compute maxFDR for one trait while preserving the public helper API."""
    prepared = _prepare_fdr_calculation(
        omega, sigma, Ns, N_counts, p_threshold
    )
    return _compute_fdr_values(prob, omega, S, prepared)[t]

def is_pos_semidef(m):
    m = np.asarray(m, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    if not np.all(np.isfinite(m)) or not np.allclose(m, m.T):
        return False
    eigs = np.linalg.eigvalsh(m)
    tolerance = np.finfo(float).eps * max(1.0, np.max(np.abs(eigs))) * len(m)
    return np.all(eigs >= -tolerance)

def neglogL_single_SS(x, beta, se, transformed=True):
    '''
    Returns the negative loglikelihood of betas from a spike-slab
    distribution. Used in the numerical optimziation of the `ss_estimation`.

    Arguments:
    ----------
    x: 2-tuple (pi_null, tau). If transformed, `x` consists of bijective transformations of pi_null and tau so that the image of the mapping is all real numbers.
    betas: The Mx1 vector of betas
    se:    The Mx1 vector of standard errors. Must allign with the reported betas.
    transformed: boolean, default True,
                If True, will perform inverse transformation on pi_null, tau so that they return to their "correct" domain.

    '''
    if  transformed:
        prob_null = scipy.special.expit(x[0])
        tau = np.exp(-x[1])
    else:
        prob_null, tau = x

    causal_logpdf = scipy.stats.norm.logpdf(
        beta, loc=0, scale=np.sqrt(tau**2 + se**2)
    )
    noncausal_logpdf = scipy.stats.norm.logpdf(beta, loc=0, scale=se)
    component_logpdfs = np.vstack(
        (
            np.log1p(-prob_null) + causal_logpdf,
            np.log(prob_null) + noncausal_logpdf,
        )
    )
    return -1.0 * np.sum(scipy.special.logsumexp(component_logpdfs, axis=0))

def cback_print(x):
    logging.info(x)

def _optim_ss(f_args):
    beta_t, se_t, starting_params, solver_opts = f_args
    start_pi, start_tau = starting_params
    if not 0.0 < start_pi < 1.0:
        raise ValueError('spike-slab starting pi must be strictly between 0 and 1')
    if start_tau <= 0.0:
        raise ValueError('spike-slab starting tau must be positive')
    x_0 = (np.log(start_pi / (1.0 - start_pi)), -np.log(start_tau))
    # beta_t, se_t = f_args
    optim_results = scipy.optimize.minimize(neglogL_single_SS, x_0, args=(beta_t, se_t,True), method='Nelder-Mead', options=solver_opts, callback=None)

    t_pi, t_tau = optim_results.x
    pi_null = scipy.special.expit(t_pi)
    tau = np.exp(-t_tau)
    if not np.isfinite(pi_null) or not np.isfinite(tau):
        raise ValueError('spike-slab optimization returned non-finite parameters')
    return pi_null, tau

def ss_estimation(args, betas, se, max_iter=1000, tol=1.0e-10,
                  starting_params =(0.5, 1.0e-3),
                  callback=False):
    '''
    Numerically fit the distribution of betas and standard errors to a spike slab distribution.

    Arguments:
    ----------
    betas: The Mx1 vector of betas
    se:    The Mx1 vector of standard errors. Must allign with the reported betas.
    max_iter: int,
            Maximum number of iterations
    tol:    float,
            Tolerance used in numerical optimization (for both fatol, xatol)

    starting_params: 2-tuple: (pi_0, tau_0)
            Starting parameters for optimization. Default is 0.5, 1.0e-3
    callback:       boolean ,default False
            If True, the parameters values will be printed at each step of optimization.
    '''
    M,T = betas.shape


    solver_opts = dict()
    solver_opts['maxiter'] = max_iter
    solver_opts['fatol'] = tol
    solver_opts['xatol'] = tol
    solver_opts['disp'] = True
    callback = cback_print if callback else None
    arg_list_ss = [(betas[:,t], se[:,t], starting_params, solver_opts) for t in range(T)]
    if args.cores == 1:
        ss_results = [_optim_ss(f_args) for f_args in arg_list_ss]
    else:
        ss_results = joblib.Parallel(
            n_jobs=args.cores,
            backend='multiprocessing',
            verbose=0,
            batch_size=1,
        )(joblib.delayed(_optim_ss)(f_args) for f_args in arg_list_ss)
    return ss_results

def some_causal_for_allT(probs, S):
    # probability of being causal is nonzero for all traits
    return np.all(_causal_pair_probabilities(probs, S) > 0.0)

def _FDR_par(func_args):
    """Compute all-trait maxFDR values for one grid point in a worker."""
    probs, omega_hat, S, prepared, g = func_args
    return _compute_fdr_values(probs, omega_hat, S, prepared), g


def _run_numba_fdr_grid(args, num_traits, prepared, pi_causal_ss):
    """Run the optional fused Numba automatic-grid maxFDR engine."""
    try:
        import numba
        import mtag_numba
    except ImportError as error:
        raise RuntimeError(
            '--fdr_backend numba requires the optional Numba dependencies; '
            'install requirements-numba.txt'
        ) from error

    available_threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
    thread_count = min(args.cores, available_threads)
    if thread_count != args.cores:
        logging.warning(
            'Requested {} maxFDR cores, but Numba supports at most {} in '
            'this process; using {}.'.format(
                args.cores, available_threads, thread_count
            )
        )
    numba.set_num_threads(thread_count)
    num_states = 1 << num_traits
    total_points = mtag_numba.nominal_grid_size(
        num_states, args.intervals
    )
    write_full_grid = getattr(args, 'fdr_write_full_grid', False)
    requested_search = getattr(args, 'fdr_search', 'auto')
    if requested_search == 'branch' and write_full_grid:
        raise ValueError(
            '--fdr-search branch currently supports max-only output; omit '
            '--fdr-write-full-grid or use --fdr-search exhaustive'
        )
    search_method = requested_search
    if search_method == 'auto':
        search_method = (
            'branch'
            if not write_full_grid and num_traits >= 5
            else 'exhaustive'
        )

    if search_method == 'branch':
        logging.info(
            'Exact branch-and-prune maxFDR search will use principal '
            'covariance constraints instead of exhaustively visiting all '
            '{:,} candidate grid points.'.format(total_points)
        )
        try:
            branch_result = mtag_numba.evaluate_automatic_grid_max_branch(
                args.intervals,
                num_traits,
                args.omega_hat,
                prepared,
                pi_causal_ss=pi_causal_ss,
            )
        except mtag_numba.BranchSearchLimitExceeded as error:
            if requested_search == 'branch' or total_points > 100_000_000:
                raise ValueError(
                    '{}. Automatic exhaustive fallback is disabled for the '
                    '{:,}-point grid; use --fdr-search exhaustive explicitly '
                    'if that cost is intentional.'.format(error, total_points)
                ) from error
            logging.info(
                'Branch-and-prune guard reached ({}); falling back to the '
                'bounded exhaustive Numba search.'.format(error)
            )
            search_method = 'exhaustive'
        else:
            max_fdr, maximizing_probabilities, feasible_count, diagnostics = (
                branch_result
            )
            logging.info(
                'Branch search selected trait order {} (original 1-based '
                'indices).'.format(
                    [trait + 1 for trait in diagnostics['trait_order']]
                )
            )
            logging.info(
                'Branch seed: {:,} candidates for each subset; {:,} of '
                '{:,} bounded candidates completed ({:,} short-circuited '
                'after losing to the best seed, {:,} skipped by the memory '
                'guard, {:,} subsets possible); {:,} sparse partial tables '
                'retained.'.format(
                    diagnostics['seed_candidates_per_subset'],
                    diagnostics['seed_subsets_tested'],
                    diagnostics['seed_subsets_considered'],
                    diagnostics['seed_subsets_short_circuited'],
                    diagnostics['seed_subsets_skipped'],
                    diagnostics['seed_subsets_available'],
                    diagnostics['seed_retained'],
                )
            )
            for level in diagnostics['levels']:
                logging.info(
                    'Branch extension adding Trait {}: {:,} candidates per '
                    'ordering choice across {:,} choices; {:,} sparse '
                    'partial tables retained (bordered PSD: {:,} accepted, '
                    '{:,} rejected, {:,} eigen fallbacks).'.format(
                        level['trait'] + 1,
                        level['candidates_per_choice'],
                        level['choices_tested'],
                        level['retained'],
                        level['fast_psd_accepts'],
                        level['fast_psd_rejects'],
                        level['eigen_fallbacks'],
                    )
                )
            logging.info(
                'Branch search evaluated {:,} complete pruned leaves; {:,} '
                'grid points passed the exact historical constraints.'.format(
                    diagnostics['final_pruned_leaves'], feasible_count
                )
            )
            return max_fdr, maximizing_probabilities, feasible_count

    logging.info(
        'Fused Numba maxFDR engine will generate and evaluate {:,} candidate '
        'grid points using {} threads, {}.'.format(
            total_points,
            thread_count,
            (
                'retaining the full feasible grid'
                if write_full_grid
                else 'retaining only each trait\'s maximum'
            ),
        )
    )
    start_fdr = time.time()
    next_progress = [10]

    def report_progress(completed, total):
        percentage = int(100.0 * completed / total)
        while next_progress[0] <= 100 and percentage >= next_progress[0]:
            logging.info(
                'Grid search: {} percent finished. Time: \t{:.3f} min'.format(
                    next_progress[0], (time.time() - start_fdr) / 60.0
                )
            )
            next_progress[0] += 10

    evaluation_function = (
        mtag_numba.evaluate_automatic_grid
        if write_full_grid
        else mtag_numba.evaluate_automatic_grid_max
    )
    chunk_size = getattr(args, 'fdr_chunk_size', None)
    if chunk_size is None:
        chunk_size = (
            100000
            if write_full_grid
            else 1000000
        )
    causal_states = create_S(num_traits)
    return evaluation_function(
        args.intervals,
        causal_states,
        args.omega_hat,
        prepared,
        pi_causal_ss=pi_causal_ss,
        chunk_size=chunk_size,
        progress_callback=report_progress,
    )


def _validated_fdr_values(fdr_values):
    """Validate and clip maxFDR values within numerical tolerance."""
    fdr_values = np.asarray(fdr_values, dtype=float)
    if not np.all(np.isfinite(fdr_values)):
        raise ValueError('maxFDR grid search returned non-finite values')
    numerical_tolerance = 1.0e-12
    if (
        np.any(fdr_values < -numerical_tolerance)
        or np.any(fdr_values > 1.0 + numerical_tolerance)
    ):
        raise ValueError('maxFDR grid search returned a value outside [0, 1]')
    return np.clip(fdr_values, 0.0, 1.0)


def _save_and_log_max_fdr(args, max_fdr, maximizing_probabilities):
    """Save the compact result and log its maximizing point per trait."""
    max_fdr_path = args.out + '_max_fdr.txt'
    np.savetxt(max_fdr_path, max_fdr[None, :], delimiter='\t')
    logging.info('Saved maximum FDR estimates in {}'.format(max_fdr_path))
    logging.info(borderline)
    if args.fit_ss:
        logging.info('FDR with the Spike-Slab parameters')
        message = 'FDR of Trait {}: {} at probs = {}'
    else:
        logging.info('Maximum FDR')
        message = 'Max FDR of Trait {}: {} at probs = {}'
    for trait, value in enumerate(max_fdr):
        if hasattr(maximizing_probabilities, 'format_row'):
            maximizing_point = maximizing_probabilities.format_row(trait)
        else:
            maximizing_point = maximizing_probabilities[trait]
        logging.info(
            message.format(
                trait + 1, value, maximizing_point
            )
        )
    logging.info(borderline)
    logging.info('Completed FDR calculations.')


def fdr(args, Ns_f, Zs, n_approx_precomputed=False):
    '''
    Ns_f: MxT matrix of sample sizes. When n_approx_precomputed is true,
    Ns_f must instead be the single row of already-rounded trait means saved
    by _write_maxfdr_inputs.
    '''
    logging.info('Beginning maxFDR calculations. Depending on the number of grid points specified, this might take some time...')

    if args.intervals <= 0:
        raise ValueError('spacing of grid points for the max FDR calculation must be a positive integer')
    if args.cores <= 0:
        raise ValueError('number of cores for the max FDR calculation must be positive')
    if not 0.0 < args.p_sig < 1.0:
        raise ValueError('maxFDR significance threshold must be strictly between 0 and 1')

    if n_approx_precomputed and (not args.n_approx or args.fit_ss):
        raise ValueError(
            'Precomputed maxFDR sample-size means require --n_approx '
            'without --fit_ss'
        )
    if n_approx_precomputed:
        Ns = np.asarray(Ns_f, dtype=float)
    else:
        Ns = np.asarray(np.round(Ns_f), dtype=float) # round to avoid decimals
    if Ns.ndim != 2:
        raise ValueError('maxFDR sample sizes must be a two-dimensional matrix')
    M,T = Ns.shape
    if M == 0 or T == 0 or not np.all(np.isfinite(Ns)) or np.any(Ns <= 0.0):
        raise ValueError('maxFDR sample sizes must be a non-empty matrix of positive finite values')
    args.omega_hat = np.asarray(args.omega_hat, dtype=float)
    args.sigma_hat = np.asarray(args.sigma_hat, dtype=float)
    if args.omega_hat.shape != (T, T):
        raise ValueError('maxFDR genetic covariance matrix dimensions do not match the number of traits')
    if args.sigma_hat.shape != (T, T):
        raise ValueError('maxFDR residual covariance matrix dimensions do not match the number of traits')
    if not np.all(np.isfinite(args.omega_hat)) or not np.all(np.isfinite(args.sigma_hat)):
        raise ValueError('maxFDR covariance matrices must contain only finite values')
    if not np.allclose(args.omega_hat, args.omega_hat.T):
        raise ValueError('maxFDR genetic covariance matrix must be symmetric')
    if not np.allclose(args.sigma_hat, args.sigma_hat.T):
        raise ValueError('maxFDR residual covariance matrix must be symmetric')
    logging.info('T='+str(T))

    pi_causal_ss = None
    if args.fit_ss:
        Zs = np.asarray(Zs, dtype=float)
        if Zs.shape != Ns.shape or not np.all(np.isfinite(Zs)):
            raise ValueError(
                'maxFDR Z scores must be a finite matrix matching the sample '
                'size matrix when --fit_ss is used'
            )
        gwas_se = 1. / np.sqrt(Ns)
        gwas_betas = gwas_se * Zs

        ss_params_list = ss_estimation(args, gwas_betas, gwas_se)
        pi_causal_ss = np.array([1.- x[0] for x in ss_params_list])
        logging.info('Completed estimation of spike-slab parameters resulting in the following causal probabilities')
        for t in range(T):
            logging.info('Trait {}: \t {:.3f}'.format(t, pi_causal_ss[t]))

    if args.n_approx:
        if n_approx_precomputed:
            if M != 1:
                raise ValueError(
                    'Precomputed maxFDR sample-size means must contain one row'
                )
            N_vals = Ns
        else:
            N_vals = np.mean(Ns, axis=0, keepdims=True)
        N_weights = np.ones(1)
    else:
        Ns_unique, Ns_counts = np.unique(Ns, return_counts=True, axis=0)
        N_vals = Ns_unique
        N_weights = Ns_counts
        assert np.sum(N_weights) == len(Ns)

    prepared = _prepare_fdr_calculation(
        args.omega_hat, args.sigma_hat, N_vals, N_weights, args.p_sig
    )
    fdr_backend = getattr(args, 'fdr_backend', 'python')
    if fdr_backend == 'numba':
        if args.grid_file is not None:
            raise ValueError(
                '--fdr_backend numba currently supports the automatic grid '
                'only; omit --grid_file or use --fdr_backend python'
            )
        numba_result = _run_numba_fdr_grid(
            args, T, prepared, pi_causal_ss
        )
        if getattr(args, 'fdr_write_full_grid', False):
            prob_grid, FDR = numba_result
            feasible_count = len(prob_grid)
            if feasible_count == 0:
                raise ValueError(
                    'No feasible maxFDR grid points remain after applying '
                    'causal-state and covariance constraints'
                )
            logging.info(
                'Number of gridpoints searched: {}'.format(feasible_count)
            )
            np.savetxt(
                args.out + '_prob_grid.txt', prob_grid, delimiter='\t'
            )
        else:
            max_FDR, maximizing_probabilities, feasible_count = numba_result
            if feasible_count == 0:
                raise ValueError(
                    'No feasible maxFDR grid points remain after applying '
                    'causal-state and covariance constraints'
                )
            if pi_causal_ss is not None:
                logging.info('{} probabilities remain after restricting to the grid points with causal probabilities less than one unit (i.e. 1/intervals) from the Spike-Slab fitted causal probabilities.'.format(feasible_count))
            logging.info(
                'Number of gridpoints searched: {}'.format(feasible_count)
            )
            max_FDR = _validated_fdr_values(max_FDR)
            _save_and_log_max_fdr(
                args, max_FDR, maximizing_probabilities
            )
            return max_FDR, maximizing_probabilities
    else:
        if getattr(args, 'fdr_search', 'auto') != 'auto':
            raise ValueError(
                '--fdr-search applies only with --fdr-backend numba'
            )
        S = create_S(T)
        causal_prob = lambda x, SS: np.sum(
            np.einsum('s,st->st', x, SS), axis=0
        )
        if args.grid_file is not None:
            candidate_grid = load_probability_grid(args.grid_file, len(S))
        else:
            # automate the creation of the probability grid
            candidate_grid = simplex_walk(len(S)-1, args.intervals+1)
        # exclude probabilities that have at least one trait with zero pi_causal
        # exclude probabilities that don't yield a valid NPD matrix
        prob_grid = []
        for probabilities in candidate_grid:
            pair_probabilities = _causal_pair_probabilities(probabilities, S)
            if not np.all(pair_probabilities > 0.0):
                continue
            if pi_causal_ss is not None and not np.all(
                np.abs(causal_prob(probabilities, S) - pi_causal_ss)
                < (1.0 / args.intervals)
            ):
                continue
            if is_pos_semidef(args.omega_hat / pair_probabilities):
                prob_grid.append(probabilities)

        if pi_causal_ss is not None:
            logging.info('{} probabilities remain after restricting to the grid points with causal probabilities less than one unit (i.e. 1/intervals) from the Spike-Slab fitted causal probabilities.'.format(len(prob_grid)))

        if len(prob_grid) == 0:
            raise ValueError(
                'No feasible maxFDR grid points remain after applying causal-state '
                'and covariance constraints'
            )
        prob_grid = np.asarray(prob_grid, dtype=float)

        logging.info('Number of gridpoints to search: {}'.format(len(prob_grid)))
        FDR = np.full((len(prob_grid), T), np.nan)
        logging.info('Performing grid search using {} cores.'.format(args.cores))
        NN = len(prob_grid)
        K = min(10, NN)
        start_fdr =time.time()
        np.savetxt(args.out + '_prob_grid.txt', prob_grid, delimiter='\t')

        def run_grid_search(parallel=None):
            for k in range(K):
                k0 = int(k*NN / K)
                k1 = int((k+1) * NN / K)
                task_arguments = (
                    (prob_grid[g], args.omega_hat, S, prepared, g)
                    for g in range(k0, k1)
                )
                if parallel is None:
                    grid_results = [
                        _FDR_par(f_args) for f_args in task_arguments
                    ]
                else:
                    grid_results = parallel(
                        joblib.delayed(_FDR_par)(f_args)
                        for f_args in task_arguments
                    )
                logging.info('Grid search: {} percent finished. Time: \t{:.3f} min'.format((k+1)*100./K, (time.time()-start_fdr)/ 60.))
                for fdr_values, grid_index in grid_results:
                    FDR[grid_index, :] = fdr_values

                np.savetxt(args.out + '_fdr_mat.txt', FDR, delimiter='\t')

        if args.cores == 1:
            run_grid_search()
        else:
            with joblib.Parallel(
                n_jobs=args.cores,
                backend='multiprocessing',
                verbose=0,
                batch_size='auto',
            ) as parallel:
                run_grid_search(parallel)

    FDR = _validated_fdr_values(FDR)

    # save FDR file once more
    np.savetxt(args.out+'_fdr_mat.txt', FDR, delimiter='\t')
    logging.info('Saved calculations of fdr over grid points in {}'.format(args.out+'_fdr_mat.txt'))

    logging.info(borderline)
    ind_max = np.argmax(FDR, axis=0)
    logging.info('grid point indices for max FDR for each trait: {}'.format(ind_max))
    max_FDR = np.max(FDR, axis=0)
    maximizing_probabilities = prob_grid[ind_max]
    _save_and_log_max_fdr(args, max_FDR, maximizing_probabilities)
    return FDR, prob_grid

def mtag(args):

    #1. Administrative checks
    if args.equal_h2 and not args.perfect_gencov:
        raise ValueError("--equal_h2 option used without --perfect_gencov. To use --equal_h2, --perfect_gencov must be also be included.")

     ## Instantiate log file and masthead
    logging.basicConfig(format='%(asctime)s %(message)s', filename=args.out + '.log', filemode='w', level=logging.INFO,datefmt='%Y/%m/%d/%I:%M:%S %p')
    if args.stream_stdout:
        logging.getLogger().addHandler(logging.StreamHandler())
    header_sub = header
    header_sub += "Calling ./mtag.py \\\n"
    defaults = vars(parser.parse_args(''))
    opts = vars(args)
    non_defaults = [x for x in opts if opts[x] != defaults[x]]
    options = ['--'+x.replace('_','-')+' '+str(opts[x])+' \\' for x in non_defaults]
    header_sub += '\n'.join(options).replace('True','').replace('False','')
    header_sub = header_sub[0:-1] + '\n'

    if args.ld_ref_panel is None:
        # mtag_path =-os.path.dirname(os.path.abspath(__file__))

        mtag_path = os.path.dirname(os.path.abspath(__file__)) +"/"

        args.ld_ref_panel = mtag_path+'ld_ref_panel/eur_w_ld_chr/'

    start_time = time.time()  # starting time of analysis
    # take output directory from --out path
    try :
        args.outdir = re.search('.+/', args.out).group(0)
    except AttributeError:
        logging.info('Invalid path used for --out: must have at least one / (use ./[tag] for current directory) and must not end in /')
        raise
    ## TODO Check all input paths
    if not os.path.isdir(args.outdir):
        if args.make_full_path or args.outdir[0] != '/':
            logging.info("Output folder provided does not exist, creating the directory")
            safely_create_folder(args.outdir)
        else:
            raise ValueError('Could not find output directory:\n {} \n at the specified absolute path. To create this directory, use the --make_full_path option.'.format(args.outdir))

    logging.info(header_sub)
    logging.info("Beginning MTAG analysis...")

    if args.median_z_cutoff != DEFAULT_MEDIAN_Z_THRESHOLD:
        logging.info("WARNING: Using non-default median Z score cutoff for QC.")

    # check for beta/se vs z/n option
    if args.use_beta_se:
        logging.info('MTAG will use the provided BETA/SE columns for analyses.')
    else:
        logging.info('MTAG will use the Z column for analyses.') 

    # 2. Load Data and perform restrictions
    DATA_U, DATA, args = load_and_merge_data(args)

    # 3. Extract core information from combined GWAS data
    Zs , Ns ,Fs, res_temp, DATA, N_raw = extract_gwas_sumstats(DATA,args,list(np.arange(args.P)))

    logging.info('Using {} SNPs to estimate Omega ({} SNPs excluded due to strand ambiguity)'.format(len(Zs)- np.sum(DATA['strand_ambig']), np.sum(DATA['strand_ambig'])))
    not_SA = np.logical_not(np.array(DATA['strand_ambig']))

    # 4. Estimate Sigma
    if args.residcov_path is None:
        logging.info('Estimating sigma..')
        sigma_start_time = time.time()
        if args.verbose:
            args.sigma_hat = estimate_sigma(DATA[not_SA], args)
        else:
            with DisableLogger():
                args.sigma_hat = estimate_sigma(DATA[not_SA], args)
        logging.info(
            'Sigma estimation completed in {:.3f} seconds.'.format(
                time.time() - sigma_start_time
            )
        )

    else:
        args.sigma_hat = _read_matrix(args.residcov_path)
    args.sigma_hat = _posDef_adjustment(args.sigma_hat)
    logging.info('Sigma hat:\n{}'.format(args.sigma_hat))


    G_mean_c2_adj = np.mean(np.square(Zs),axis=0) / np.diag(args.sigma_hat)
    low_c2 = G_mean_c2_adj < 1.1
    if np.any(low_c2):
        low_c2_msg = 'Mean chi^2 of SNPs used to estimate Omega is low for some SNPs'
        #low_c2_msg += 'Traits {}'.format(' '.join(np.arange(1,args.P+1)[low_c2])) if np.sum(low_c2) > 1 else 'Trait {}'.format(' '.join(np.arange(1,args.P+1)[low_c2]))
        #low_c2_msg += '(= {})'.format(' '.join(G_mean_c2_adj[low_c2]))
        low_c2_msg += 'MTAG may not perform well in this situation.'
        logging.info(low_c2_msg)

    #5. Estimate Omega
    if args.gencov_path is None:
        not_SA = np.logical_not(np.array(DATA['strand_ambig']))
        args.omega_hat = estimate_omega(args, Zs[not_SA], Ns[not_SA], args.sigma_hat)
        logging.info('Completed estimation of Omega ...')
    else:
        args.omega_hat = _read_matrix(args.gencov_path)


    assert args.omega_hat.shape[0] == args.omega_hat.shape[1] == Zs.shape[1] == args.sigma_hat.shape[0] == args.sigma_hat.shape[1]

    #6. Meta format analysis
    if args.meta_format:
        create_ind = lambda P: np.asarray(list(itertools.product([0,1], repeat=P)))
        snp_type = create_ind(args.P)
        snp_dict = {tuple(snp_type[s]): snp_type[s] for s in range(len(snp_type))}

        # Generate subgroups of SNPs
        sub_list = [x for x in DATA_U.groupby([x for x in DATA_U.columns if "Trait" in x])]
        sub_dict = {sub_list[x][0]: sub_list[x][1] for x in range(len(sub_list))}
        combo_dict = dict()

        # Loop over subtypes of SNPs
        comb_df = dict.fromkeys(sub_dict)

        for s in range(len(sub_dict)):
            t = np.sum(list(sub_dict.keys())[s])
            tl = np.arange(args.P)[np.in1d(list(sub_dict.keys())[s],1)] #indexes of true trait

            # extract columns that are of p trait
            GWAS_sub = dict()
            for i in tl:
                GWAS_sub[i] = sub_dict[sub_list[s][0]].filter(regex='^SNP|.+{}$'.format(i))

            # inner merge sumstats for each trait
            snp_df = reduce(lambda L,R: pd.merge(L,R,how='inner',on='SNP'), list(GWAS_sub.values()))
            combo_dict[sub_list[s][0]] = snp_df
            logging.info('There are {M} SNPs present in Traits {tl}'.format(M=snp_df.shape[0], tl=np.array_str(tl)))

            # No need to flip SNPs if they are present in some but not all sumstats? 

            # extract gwas sumstats for each combo
            Zs, Ns, Fs, resid_cols, _, N_raw = extract_gwas_sumstats(snp_df, args, tl)

            # perform MTAG on each type of SNPs
            omega_sub, sigma_sub = args.omega_hat[tl[:,None],tl[None,:]], args.sigma_hat[tl[:,None],tl[None,:]]
            mtag_betas, mtag_se, mtag_factor = mtag_analysis(Zs, Ns, omega_sub, sigma_sub)

            # combine types of SNPs by traits <save_mtag_results>

            p_values = lambda z: 2*(scipy.stats.norm.cdf(-1.*np.abs(z)))

            M,_ = mtag_betas.shape

            comb_df[sub_list[s][0]] = dict() # combo's keys are the index of true traits

            for p,t in enumerate(tl):
                comb_df[sub_list[s][0]][t] = resid_cols.copy()
                comb_df[sub_list[s][0]][t][args.z_name] = Zs[:,p]
                comb_df[sub_list[s][0]][t][args.n_name] = N_raw[:,p]
                comb_df[sub_list[s][0]][t][args.eaf_name] = Fs[:,p]

                if args.std_betas:
                    weights = np.ones(M,dtype=float)
                else:
                    weights = np.sqrt( 2*Fs[:,p]*(1. - Fs[:,p]))
                comb_df[sub_list[s][0]][t]['mtag_beta'] = mtag_betas[:,p] / weights
                comb_df[sub_list[s][0]][t]['mtag_se'] = mtag_se[:,p] / weights
                comb_df[sub_list[s][0]][t]['mtag_z'] = mtag_betas[:,p]/mtag_se[:,p]
                comb_df[sub_list[s][0]][t]['mtag_pval'] = p_values(comb_df[sub_list[s][0]][t]['mtag_z'])

        # check all elements sum to union SNPs
        comb_flat = [list(comb_df[y].values())[0] for y in comb_df]
        assert DATA_U.shape[0] == np.sum(np.asarray([x.shape[0] for x in comb_flat]))

    mtag_betas, mtag_se, mtag_factor = mtag_analysis(Zs, Ns, args.omega_hat, args.sigma_hat)

    #7. Save sumstats to files
    if args.meta_format:
        save_mtag_results_U(args, comb_df)
        write_summary(args, Zs, comb_df, Fs, mtag_betas, mtag_se, mtag_factor)

    else:
        save_mtag_results(args, res_temp, Zs, N_raw, Fs, mtag_betas, mtag_se, mtag_factor)
        write_summary(args, Zs, N_raw, Fs, mtag_betas, mtag_se, mtag_factor)
        _write_maxfdr_inputs(args, Ns)

    if args.fdr:
        fdr(args, Ns, Zs)

    logging.info('MTAG complete. Time elapsed: {}'.format(sec_to_str(time.time()-start_time)))

parser = argparse.ArgumentParser(description="\n **mtag: Multitrait Analysis of GWAS**\n This program is the implementation of MTAG method described by Turley et. al. Requires the input of a comma-separated list of GWAS summary statistics with identical columns. It is recommended to pass the column names manually to the program using the options below. The implementation of MTAG makes use of the LD Score Regression (ldsc) for cleaning the data and estimating residual variance-covariance matrix, so the input must also be compatible ./munge_sumstats.py command in the ldsc distribution included with mtag. The default estimation method for the genetic covariance matrix Omega is GMM (as described in the paper). \n\n Note below: any list of passed to the options below must be comma-separated without whitespace.")

in_opts = parser.add_argument_group(title='Input Files', description="Input files to be used by MTAG. The --sumstats option is required, while using the other two options take priority of their corresponding estimation routines, if used.")
in_opts.add_argument("--sumstats", metavar="{File1},{File2}...", type=str, nargs='?',required=False, help='Specify the list of summary statistics files to perform multitrait analysis. Multiple files paths must be separated by \",\". Please read the documentation  to find the up-to-date set of acceptable file formats. A general guideline is that any files you pass into MTAG should also be parsable by ldsc and you should take the additional step of specifying the names of the main columns below to avoid reading errors.')
in_opts.add_argument("--gencov_path",metavar="FILE_PATH", default=None, action="store", help="If specified, will read in the genetic covariance matrix saved in the file path below and skip the estimation routine. The rows and columns of the matrix must correspond to the order of the GWAS input files specified. FIles can either be in whitespace-delimited .txt  or .npy format. Use with caution as the genetic covariance matrix specified will be weakly nonoptimal.")
in_opts.add_argument("--residcov_path",metavar="FILE_PATH", default=None, action="store", help="If specified, will read in the residual covariance matrix saved in the file path below and skip the estimation routine. The rows and columns of the matrix must correspond to the order of the GWAS input files specified. FIles can either be in .txt  or .npy format. Use with caution as the genetic covariance matrix specified will be weakly nonoptimal. File must either be in whitespace-delimited .txt  or .npy")

out_opts = parser.add_argument_group(title="Output formatting", description="Set the output directory and common name of prefix files.")
out_opts.add_argument("--out", metavar='DIR/PREFIX', default='./mtag_results', type=str, help='Specify the directory and name prefix to output MTAG results. All mtag results will be prefixed with the corresponding tag. Default is ./mtag_results')
out_opts.add_argument("--make_full_path", default=False, action="store_true", help="option to make output path specified in -out if it does not exist.")
out_opts.add_argument("--meta_format", default=False, action="store_true", help="In addition to the typical results file that are restricted to the intersection of SNPs across files, this creates a file of the union of SNPs, with applications of the MTAG estimator restricted to the set of traits for which that SNP is available.")
out_opts.add_argument('--output_backend', '--output-backend', choices=('pandas', 'polars'), default='polars', help='Results-table writer. The single-threaded Polars backend is the default. Use pandas to preserve historical text formatting, or --legacy-loader for the full historical I/O path.')

input_formatting = parser.add_argument_group(title="Column names of input files", description="These options manually pass the names of the relevant summary statistics columns used by MTAG. It is recommended to pass these names because only narrow searches for these columns are performed in the default cases. Moreover, it is necessary that these input files be readable by ldsc's munge_sumstats command.")
input_formatting.add_argument("--snp_name", default="snpid", action="store",type=str, help="Name of the single column that provides the unique identifier for SNPs in the GWAS summary statistics across all GWAS results. Default is \"snpid\". This the index that will be used to merge the GWAS summary statistics. Any SNP lists passed to ---include or --exclude should also contain the same name.")
input_formatting.add_argument("--z_name", default="z", help="The common name of the column of Z scores across all input files. Default is the lowercase letter z.")
input_formatting.add_argument("--beta_name", default="beta", help="The common name of the column of beta coefficients (effect sizes) across all input files. Must be specified with se.")
input_formatting.add_argument("--se_name", default="se", help="The common name of the column of standard errors of the betas across all input files. Default is lowercase se. Must be specified with --beta_name.")
input_formatting.add_argument("--n_name", default="n", help="the common name of the column of sample sizes in the GWAS summary statistics files. Default is the lowercase letter  n.")
input_formatting.add_argument("--n_value", default=None, metavar="N1, N2,...", type=str, help="Comma separated sample size values for each GWAS summary statistics files. This option is useful for GWAS input that does not include an N column, e.g. BOLT-LMM.")
input_formatting.add_argument('--eaf_name',default="freq", help="The common name of the column of effect allele frequencies in the GWAS input files. The default is \"freq\".")
input_formatting.add_argument('--no_chr_data',default=False,action='store_true', help="If used, will not use information related to the chromosome and base pair position columns. Use only it chromosome/base pair positional data is missing, but are certain that the snpids correctly identify the SNPs across traits.")
input_formatting.add_argument('--chr_name',default='chr', type=str, help="Name of the column containing the chromosome of each SNP in the GWAS input. Default is \"chr\".")
input_formatting.add_argument('--bpos_name',default='bpos', type=str, help="Name of the column containing the base pair of each SNP in the GWAS input. Default is \"bpos\".")
input_formatting.add_argument('--a1_name',default='a1', type=str, help="Name of the column containing the effect allele of each SNP in the GWAS input. Default is \"a1\".")
input_formatting.add_argument('--a2_name',default='a2', type=str, help="Name of the column containing the non-effect allele of each SNP in the GWAS input. Default is \"a2\".")
input_formatting.add_argument('--p_name',default='p', type=str, help="Name of the column containing the p-value of the effect size in the GWAS input. Default is \"p\".")

filter_opts = parser.add_argument_group(title="Filter Options", description="The input summary statistics files can be filtered using the options below. Note that there is some default filtering according to sample size and allele frequency, following the recommendations we make in the corresponding paper. All of these column-based options allow a list of values to be passed of the same length as the number of traits ")
filter_opts.add_argument("--include",default=None, metavar="SNPLIST1,SNPLIST2,..", type=str, help="Restricts MTAG analysis to the union of snps in the list of  snplists provided. The header line must match the SNP index that will be used to merge the GWAS input files.")
filter_opts.add_argument("--exclude", "--excludeSNPs",default=None, metavar="SNPLIST1,SNPLIST2,..", type=str, help="Similar to the --include option, except that the union of SNPs found in the specified files will be excluded from MTAG. Both -exclude and -include may be simultaneously specified, but -exclude will take precedent (i.e., SNPs found in both the -include and -exclude SNP lists will be excluded).")
filter_opts.add_argument('--only_chr', metavar="CHR_A,CHR_B,..", default=None, type=str, action="store", help="Restrict MTAG to SNPs on one of the listed, comma-separated chromosome. Can be specified simultaneously with --include and --exclude, but will take precedent over both. Not generally recommended. Multiple chromosome numbers should be separated by commas without whitespace. If this option is specified, the GWAS summary statistics must also list the chromosome of each SNPs in a column named `chr`.")
filter_opts.add_argument("--homogNs_frac", default=None, type=str, action="store", metavar="FRAC", help="Restricts to SNPs within FRAC of the mode of sample sizes for the SNPs as given by (N-Mode)/Mode < FRAC. This filter is not applied by default.")
filter_opts.add_argument("--homogNs_dist", default=None, type=str, action="store", metavar="D", help="Restricts to SNPs within DIST (in sample size) of the mode of sample sizes for the SNPs. This filter is not applied by default.")
filter_opts.add_argument('--maf_min', default='0.01', type=str, action='store', help="set the threshold below SNPs with low minor allele frequencies will be dropped. Default is 0.01. Set to 0 to skip MAF filtering.")
filter_opts.add_argument('--n_min', default=None, type=str, action='store', help="set the minimum threshold for SNP sample size in input data. Default is 2/3*(90th percentile). Any SNP that does not pass this threshold for all of the GWAS input statistics will not be included in MTAG.")
filter_opts.add_argument('--n_max', default=None, type=str, action='store', help="set the maximum threshold for SNP sample size in input data. Not used by default. Any SNP that does not pass this threshold for any of the GWAS input statistics will not be included in MTAG.")
filter_opts.add_argument("--info_min", default=None,type=str, help="Minimim info score for filtering SNPs for MTAG.")
filter_opts.add_argument("--incld_ambig_snps", default=False, action="store_true", help="Include strand ambiguous SNPs when performing MTAG. by default, they are not used to estimate Omega or Sigma.")
filter_opts.add_argument("--no_allele_flipping", default=False, action="store_true", help="Prevents flipping the effect sizes of summary statistics when the effect and non-effect alleles are reversed (reletive the first summary statistics file.")

special_cases = parser.add_argument_group(title="Special Cases",description="These options deal with notable special cases of MTAG that yield improvements in runtime. However, they should be used with caution as they will yield non-optimal results if the assumptions implicit in each option are violated.")
special_cases.add_argument('--use_beta_se', default=False, action='store_true', help='If turned on, MTAG will use the provided BETA and SE columns to perform estimation. By default, MTAG uses the Z-statistic.')
special_cases.add_argument('--no_overlap', default=False, action='store_true', help='Imposes the assumption that there is no sample overlap between the input GWAS summary statistics. MTAG is performed with the off-diagonal terms on the residual covariance matrix set to 0.')
special_cases.add_argument('--perfect_gencov', default=False, action='store_true', help='Imposes the assumption that all traits used are perfectly genetically correlated with each other. The off-diagonal terms of the genetic covariance matrix are set to the square root of the product of the heritabilities')
special_cases.add_argument('--equal_h2', default=False, action='store_true', help='Imposes the assumption that all traits passed to MTAG have equal heritability. The diagonal terms of the genetic covariance matrix are set equal to each other. Can only be used in conjunction with --perfect_gencov')
special_cases.add_argument('--force', default=False, action='store_true', help='Force MTAG estimation even though the mean chi2 is small.')

fdr_opts = parser.add_argument_group(title='Max FDR calculation', description="These options are used for the calculation of an upper bound on the false disovery under the model described in Supplementary Note 1.1.4 of Turley et al. (2017). Note that there is one of three ways to define the space of grid points over which the upper bound is searched. ")
fdr_opts.add_argument('--fdr', default=False, action='store_true', help='Perform max FDR calculations')
fdr_opts.add_argument('--skip_mtag', default=False, action='store_true', help='Skip MTAG and perform maxFDR from an existing output prefix. Standard MTAG runs write a compact input sidecar used automatically with the default --n_approx; older outputs and SNP-level maxFDR modes remain supported through the trait result files.')
fdr_opts.add_argument('--grid_file',default=None, action='store', help='Pre-set list of grid points. Users can define a list of grid points over which the search is conducted. The list of grid points should be passed in text file as a white-space delimited matrix of dimnesions, G x S, where G is the number of grid points and S = 2^T is the number of possible causal states for SNPs. States are ordered according to a tree-like recursive structure from right to left. For example, for 3 traits, with the triple TFT denoting the state for which SNPs are causal for State 1, not causal for state 2, and causal for state 3, then the column ordering of probabilities should be: \nFFF FFT FTF FTT TFF TFT TTF TTT\n There should be no headers, or row names in the file. Any rows for which (i) the probabilities do not sum to 1, the prior of a SNP being is causal is 0 for any of the traits, and (iii) the resulting genetic correlation matrix is non positive definite will excluded in the search.')
fdr_opts.add_argument('--fit_ss', default=False, action='store_true', help='This estimates the prior probability that a SNP is null for each trait and then proceeds to restrict the grid search to the set of probability vectors that sum to the prior null for each trait. This is useful for restrict the search space of larger-dimensional traits.')
fdr_opts.add_argument('--intervals', default=10, action='store',type=int, help='Number of intervals that you would like to partition the [0,1] interval. For example example, with two traits and --intervals set 10, then maxFDR will calculated over the set of feasible points in {0., 0.1, 0.2,..,0.9,1.0}^2.')
fdr_opts.add_argument('--cores', default=1, action='store', type=int, help='Number of threads/cores use to compute the FDR grid points for each trait.')
fdr_opts.add_argument('--p_sig', default=5.0e-8, type=float, action='store', help='P-value threshold used for statistical signifiance. Default is p=5.0e-8 (genome-wide significance).' )
fdr_opts.add_argument('--n_approx', default=True, dest='n_approx', action='store_true', help='Speed up FDR calculation by replacing the sample size of a SNP for each trait by the mean across SNPs (for each trait). Recommended and enabled by default.')
fdr_opts.add_argument('--no_n_approx', '--no-n-approx', dest='n_approx', action='store_false', help='Use each distinct row of SNP sample sizes in the maxFDR power calculation instead of trait means.')
fdr_opts.add_argument('--fdr_backend', '--fdr-backend', choices=('python', 'numba'), default='python', help='maxFDR execution engine. The optional numba backend fuses automatic-grid generation and evaluation; install requirements-numba.txt first. Default is python.')
fdr_opts.add_argument('--fdr_search', '--fdr-search', choices=('auto', 'branch', 'exhaustive'), default='auto', help='Automatic-grid search strategy for the Numba backend. Auto uses exact branch-and-prune for five or more traits with max-only output and bounded exhaustive streaming otherwise. Use exhaustive as a compatibility reference.')
fdr_opts.add_argument('--fdr_chunk_size', '--fdr-chunk-size', default=None, type=int, help='Number of automatic maxFDR candidates evaluated per native chunk with --fdr_backend numba. Defaults to 1000000 for max-only reduction and 100000 with --fdr-write-full-grid.')
fdr_opts.add_argument('--fdr_write_full_grid', '--fdr-write-full-grid', action='store_true', help='With --fdr_backend numba, retain and write the complete feasible probability grid and FDR matrix instead of only the per-trait maxima.')

# fdr_opts.add_argument('--binned_n', default=False, action='store_true', help='When --n_approx is off, this options allows for a sped-up version of the max_FDR calculation by weighting the power calculations of unique rows.')

# wc = parser.add_argument_group(title='Winner\'s curse adjustment', description='Options related to the winner\'s curse adjustment of estimates of effect sizes from MTAG that could be used when replicating analyses.')
# GWAS or MTAG results?
# maybe both?

misc = parser.add_argument_group(title="Miscellaneous")

misc.add_argument('--ld_ref_panel', default=None, action='store',metavar="FOLDER_PATH", type=str, help='Specify folder of the ld reference panel (split by chromosome) that will be used in the estimation of the error VCV (sigma). This option is passed to --ref-ld-chr and --w-ld-chr when running LD score regression. The default is to use the reference panel of LD scores computed from 1000 Genomes European subjects (eur_w_ld_chr) that is included with the distribution of MTAG')
misc.add_argument('--time_limit', default=100.,type=float, action="store", help="Set time limit (hours) on the numerical estimation of the variance covariance matrix for MTAG, after which the optimization routine will complete its current iteration and perform MTAG using the last iteration of the genetic VCV.")

misc.add_argument('--std_betas', default=False, action='store_true', help="Results files will have standardized effect sizes, i.e., the weights 1/sqrt(2*MAF*(1-MAF)) are not applied when outputting MTAG results, where MAF is the minor allele frequency.")
misc.add_argument("--tol", default=1e-6,type=float, help="Set the relative (x) tolerance when numerically estimating the genetic variance-covariance matrix. Not recommended to change unless you are facing strong runtime constraints for a large number of traits.")
misc.add_argument('--numerical_omega', default=False, action='store_true', help='Option to use the MLE estimator of the genetic VCV matrix, implemented through a numerical routine.')
misc.add_argument('--verbose', default=False, action='store_true', help='When used, will include output from running ldsc scripts as well additional information (such as optimization routine information.')
misc.add_argument('--chunksize', default=1e7, type=int, help='Chunksize for reading in data.')
misc.add_argument('--load_backend', '--load-backend', choices=('pandas', 'polars'), default='polars', help='Summary-statistics loading engine. The single-threaded Rust-backed Polars backend is the default for tab-delimited inputs; arbitrary whitespace inputs fall back to the fused pandas loader.')
misc.add_argument('--stream_stdout', default=False, action='store_true', help='Will streat mtag processing on console in addition to writing to log file.')
misc.add_argument('--median_z_cutoff', default=DEFAULT_MEDIAN_Z_THRESHOLD, type=float, help='Maximum allowed median Z-score for sumstats during input QC')
misc.add_argument('--legacy_loader', '--legacy-loader', default=False, action='store_true', help='Use the original pandas loading, merge, and output path. This is intended as a compatibility fallback and for validating the optimized default I/O path.')

if __name__ == '__main__':
    start_t = time.time()
    args = parser.parse_args()
    if args.use_beta_se:
        raise RuntimeError("Due to bugs in the beta-se code, this option has been temporarily removed from the MTAG software. Dec 2, 2021")

    if args.skip_mtag:
        # avoid overwriting the original mtag log file
        logging.basicConfig(format='%(asctime)s %(message)s', filename=args.out + '.FDR.log', filemode='w', level=logging.INFO,datefmt='%Y/%m/%d/%I:%M:%S %p')
        if args.stream_stdout:
            logging.getLogger().addHandler(logging.StreamHandler()) # logging.infos to console

        N_mat, Z_mat, n_approx_precomputed = _load_skip_mtag_sumstats(args)

        # Keep the covariance text files as the source of truth so existing
        # workflows that inspect or intentionally replace them behave as before.
        args.sigma_hat = np.atleast_2d(
            np.loadtxt(args.out + '_sigma_hat.txt')
        )
        args.omega_hat = np.atleast_2d(
            np.loadtxt(args.out + '_omega_hat.txt')
        )

        fdr(
            args,
            N_mat,
            Z_mat,
            n_approx_precomputed=n_approx_precomputed,
        )

    else:             
        try:
            mtag(args)
        except Exception as e:
            logging.error(e,exc_info=True)
            logging.info('Analysis terminated from error at {T}'.format(T=time.ctime()))
            time_elapsed = round(time.time() - start_t, 2)
            logging.info('Total time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))
