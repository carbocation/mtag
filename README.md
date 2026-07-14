# `mtag` (Multi-Trait Analysis of GWAS)

`mtag` is a Python-based command line tool for jointly analyzing multiple sets of GWAS summary statistics as described by [Turley et. al. (2018)](https://www.nature.com/articles/s41588-017-0009-4). It can also be used as a tool to meta-analyze GWAS results.

## Getting Started

We recommend installing the [Anaconda python distribution](https://www.anaconda.com/download/) as it includes all of the packages listed below. It also makes updating packages relatively painless with the `conda update` command.

To run `mtag`, you will need to have Python 2.7 installed with the following packages:

* `numpy (>=1.13.1)`  
* `scipy`
* `pandas (>=0.18.1)`
* `argparse`
* `bitarray` (for `ldsc`)
* `joblib`

### Optional CUDA acceleration

The main per-SNP MTAG calculation can run on an NVIDIA GPU through CuPy. Add
`--cuda` to the MTAG command after installing a CuPy build compatible with the
Python and CUDA versions in the MTAG environment:

    ./mtag.py [existing options] --cuda

CUDA execution is opt-in; without `--cuda`, MTAG uses the original NumPy path
and does not require CuPy. The GPU calculation uses float64, returns ordinary
NumPy arrays, and processes SNPs in chunks to bound GPU memory use. Use
`--cuda_device` to select a GPU and `--cuda_batch_size` to tune the chunk size.
The default batch size is 100,000 SNPs; larger-memory GPUs can generally use a
larger value, while an out-of-memory error should be handled by reducing it.
Omega/Sigma estimation and the maxFDR grid search remain CPU-based.

The environment shipped with this repository uses Python 2.7. CuPy 6.7 is the
last release line compatible with Python 2 and supports CUDA through 10.1; for
example, a CUDA 10.1 environment can use `cupy-cuda101==6.7.0`. Current CuPy
releases require Python 3, so using a current CUDA runtime for the complete MTAG
CLI also requires a Python 3-compatible MTAG environment. See the
[CuPy upgrade guide](https://docs.cupy.dev/en/stable/upgrade.html) and the
[CuPy 6.7 installation guide](https://docs.cupy.dev/en/v6.7.0/install.html).

To check numerical agreement and measure speed on a CUDA host:

    python benchmarks/benchmark_mtag_cuda.py --snps 1000000 --traits 3

(Note: if you already have the Python 3 version of the Anaconda distribution installed, then you will need to create and activate a Python 2.7 environment to run `mtag`. See [here](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) for details.)


`mtag` may be downloaded by cloning this github repository:

	git clone https://github.com/omeed-maghzian/mtag.git
	cd mtag

To test that the tool has been successfully installed, type:

	./mtag.py -h

You should see a list of command-line flags and a description of the program. If an error is thrown instead, then there was some problem with the installation process.

A tutorial that walks through an example use of `mtag` may be found in the wiki.

### Updating `mtag`

The easiest was to update `mtag` is through `git`. When you are in the `mtag/` directory, simply enter

	git pull

which will update the `mtag` files. If there have been no new updates since the last download of `mtag` then the terminal will print: `Already up-to-date.`

### Support

We will try our best to address any problems that one may encounter when using `mtag`. However, before [opening an issue](https://github.com/omeed-maghzian/mtag/issues) or emailing us, please first:

1. Read the wiki, especially the tutorial and FAQ pages
2. Read the desciption of the method in the paper listed below

If that doesn't prove to be enlightening, then feel free to post a question [as a Github issue](https://github.com/omeed-maghzian/mtag/issues) or [contact us](mailto:maghzian@nber.org). The more information you provide about your problem, the more helpful we can be!

### Citation

If you use the `mtag` software or methodology, please cite:

Turley, et. al. (2018) Multi-Trait analysis of genome-wide association summary statistics using MTAG. Nature Genetics  doi: <https://doi.org/10.1038/s41588-017-0009-4>.

### License

This project is licensed under GNU General Public License v3.

### Authors

Omeed Maghzian (Harvard University, Department of Economics)

Raymond Walters (Broad Institute of MIT and Harvard)

Patrick Turley (Broad Institute of MIT and Harvard)

### Acknowledgments

The development of this software was carried out under the auspices of the Social Science Genetic Association Consortium (SSGAC). This work was supported by the Ragnar Söderberg Foundation (E9/11 E42/15), the Swedish Research Council (421-2013-1061), The Jan Wallander and Tom Hedelius Foundation, an ERC Consolidator Grant (647648 EdGe), the Pershing Square Fund of the Foundations of Human Behavior, the National Science Foundation’s Graduate Research Fellowship Program (DGE 1144083), and the NIA/NIH through grants P01-AG005842, P01-AG005842-20S2, P30-AG012810, and T32-AG000186-23 to NBER, R01-AG042568-02 to the University of Southern California, and 1R01MH107649-01and 1R01MH101244-02 to the Broad Institute at Harvard and MIT. 
