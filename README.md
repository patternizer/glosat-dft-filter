![image](https://github.com/patternizer/glosat-dft-filter/blob/main/ts-dft-filter-pctl-90.png)
![image](https://github.com/patternizer/glosat-dft-filter/blob/main/ts-dft-filter-pctl-80.png)
![image](https://github.com/patternizer/glosat-dft-filter/blob/main/ts-dft-filter-optimisation.png)

# glosat-dft-filter

Python codebase to calculate DFT-filtered timeseries using a Hamming window filter and variance optimisation for long station timeseries. Part of ongoing work for the [GloSAT](https://www.glosat.org) project: www.glosat.org. 

## Contents

* `ts-dft-filter-lowpass-no-window.py` - python script DFT low pass filter using a cut-off frequency and no window
* `ts-dft-filter-lowpass-hamming-window.py` - python script DFT low pass filter using a hamming window filter and zero-padding
* `ts-dft-filter-lowpass-extrapolation.py` - python script DFT low pass filter extrapolation estimate for periodic timeseries
* `ts-dft-filter-lowpass.py` - python script DFT low pass filter for given cut-off frequency and low pass variance percentile
* `ts-dft-filter-lowpass-optimiser.py` - python script DFT low pass filter look-up table ML generator for low pass variance percentiles in the range [0,100,0.1]
* `filter_cru.py` - python translation of IDL Gaussian filter code
* `OUT/ml_optimisation.csv` - look-up table of variance percentile, cut-off frequency, low and high pass signal variance, high pass signal mean, and smoothing period (years)

## Instructions for use

The first step is to clone the latest glosat-dft-filter code and step into the installed Github directory: 

    $ git clone https://github.com/patternizer/glosat-dft-filter.git
    $ cd glosat-dft-filter

Then create a DATA/ directory and copy to it the required GloSAT.p03 pickled temperature archive file.

### Using Standard Python

The code is designed to run in an environment using Miniconda3-latest-Linux-x86_64.

    $ python ts-dft-filter-lowpass-no-window.py
    $ python ts-dft-filter-lowpass-hamming-window.py
    $ python ts-dft-filter-lowpass-extrapolation.py
    $ python ts-dft-filter-lowpass.py
    $ python ts-dft-filter-lowpass-optimiser.py
    $ python filter_cru.py

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)


