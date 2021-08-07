![image](https://github.com/patternizer/glosat-dft-filter/blob/main/ts-dft-filter-5.png)
![image](https://github.com/patternizer/glosat-dft-filter/blob/main/ts-dft-filter-10.png)
![image](https://github.com/patternizer/glosat-dft-filter/blob/main/ts-dft-filter-optimisation.png)

# glosat-dft-filter

Python codebase to calculate DFT-filtered timeseries using a Hamming window filter and variance optimisation for long station timeseries. Part of ongoing work for the [GloSAT](https://www.glosat.org) project: www.glosat.org. 

## Contents

* `ts-dft-filter-lowpass-optimiser.py` - python DFT low pass filter look-up table ML generator for low pass variance percentiles in the range [0,100,0.1]
* `filter_cru_dft.py` - python script DFT low pass filter using a hamming window with zero-padding
* `filter_cru_dft_wrapper.py` - python wrapper calling filter_cru_dft.py
* `filter_cru_gaussian.py` - python translation of CRU IDL Gaussian weighted filter code
* `filter_cru_gaussian_wrapper.py` - python wrapper calling filter_cru_gaussian.py
* `ml_optimisation.csv` - look-up table of variance percentile, cut-off frequency, low and high pass signal variance, high pass signal mean, and smoothing period (years)

## Instructions for use

The first step is to clone the latest glosat-dft-filter code and step into the installed Github directory: 

    $ git clone https://github.com/patternizer/glosat-dft-filter.git
    $ cd glosat-dft-filter

Then create a DATA/ directory and copy to it the required GloSAT.p03 pickled temperature archive file.

### Using Standard Python

The code is designed to run in an environment using Miniconda3-latest-Linux-x86_64. You can set the GloSAT station code in the wrapper functions which will perform the timeseries filtering.

    $ python filter_cru_dft_wrapper.py
    $ python filter_cru_gaussian_wrapper.py

The syntax for the filter functions are as follows (please see the doc strings for further details):

	tslow, tshigh, weight = cru_filter(thalf, tsin, nan, truncate)
	y_lo, y_hi, zvarlo, zvarhi, fc, pctl = dft_filter(y, w)
    
## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)


