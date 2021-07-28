#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: ts-dft-filter-lowpass.py
#------------------------------------------------------------------------------
# Version 0.2
# 28 July, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
#from scipy import fftpack
#from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

w = 10                                                          # smoothing windows >= 2 (Nyquist freq) 
stationcode = '037401'

fontsize = 16
make_plot = True 
use_filter = True                                               # True --> Hamming window, False --> cut-off (no window)
show_pandas = True                                              # Overlay equi-window Pandas smoother
lut = 'OUT/ml_optimisation.csv'
   
#-----------------------------------------------------------------------------
# METHODS
#-----------------------------------------------------------------------------

def nextpowerof2(x):
    
    if x > 1:
        for i in range(1, int(x)):
            if ( 2**i >= x ):
                return 2**i
    else:
        return 1

def dft(x):

    # Discrete Fourier Transform

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
        
    return np.dot(e, x)

def idft(x):

    # Inverse Discrete Fourier Transform

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)
        
    return 1/N * np.dot(e, x)
    
def dft_filter(y, w):
    
    N = len(y)                                                      # signal length
    n = np.arange(N)
    Fs = 1                                                          # sampling rate
    Ts = N/Fs                                                       # sampling interval 
    f = n/Ts                                                        # frequencies [0,1] 
    freqA = f[0:int(N/2)]                                           # frequencies: RHS
    freqB = f[int(N/2):]-1                                          # frequencies: LHS
    freq = np.array(list(freqA) + list(freqB))                      # construct SciPy's fftfreq(len(y), d=1) function
          
    y_mean = np.nanmean(y)
    y = y - y_mean                                                  # cneter timeseries (NB: add back in later)
            
    z = dft(y)                                                      # DFT
    zamp = np.sqrt(np.real(z)**2.0 + np.imag(z)**2.0) / N           # ampplitudes
    zphase = np.arctan2( np.imag(z), np.real(z) )                   # phases

    zvar = zamp**2                                                  # variance of each harmonic
    zvarsum = np.sum(zvar)                                          # total variance of spectrum
    yvarsum = np.var(y)                                             # total variance of timeseries
    zvarpc = zvar / zvarsum                                         # percentage variance per harmonic
        
    if w >= 2:
        
        df = pd.read_csv( lut )
        fc = 1.0/w
        pctl = df[df['fc'] > fc]['pctl'].iloc[-1]        

    else:
                       
        pctl = 90                                                   # (default) if no fc provided
        
    zpeaks = zvarpc[ zvarpc > ( np.percentile(zvarpc, pctl) ) ]     # high variance peaks > p(pctl)
    zpeaks_idx = np.argsort( zpeaks )                               # peak indices
    znopeaks_idx = np.setdiff1d( np.argsort( zvarpc ), zpeaks_idx)  # remaining indices
    npeaks = len(zpeaks_idx)                                        # number of peaks
    zvarlo = np.sum( [zvarpc[i] for i in zpeaks_idx] )              # total percentage variance of low pass 
    zvarhi = np.sum( [zvarpc[i] for i in znopeaks_idx] )            # total percentage variance of high pass  

    print('w',w)
    print('fc',fc)                                                  # user provided value
    print('pctl', pctl)
    print('zvarlo', zvarlo)
    print('zvarhi', zvarhi)

    if w < 2:
        
        fc = freq[ zpeaks_idx.max() ]                               # estimate low pass / high pass cut-off       
                 
    
    if use_filter == False:
        
        # DFT zero-order estimate: low pass filter (no window)
         
        y_dft = dft(y)
        y_dft_filtered = y_dft.copy()
#       frequencies = fftfreq(len(y), d=1)    
        y_dft_filtered[ np.abs(freq) > fc ] = 0                     # low-pass filter
        y_filtered_lo = np.real( idft(y_dft_filtered) )             # low pass signal
        y_filtered_hi = y - y_filtered_lo                           # high pass signal

    else:
        
        # FILTER DESIGN: low pass filter (Hamming window) with zero-padding
        
        L = N+1                                                     # filter length (M+1)
        h_support = np.arange( -int((L-1)/2), int((L-1)/2)+1 )      # filter support
        h_ideal = ( 2*fc/Fs) * np.sinc( 2*fc*h_support/Fs )         # filter (ideal)
        h = np.hamming(L).T*h_ideal                                 # filter
        
        # ZERO-PAD: (next power of 2 > L+M-1) signal and impulse-response
        
        #Ndft = int(2**(np.ceil(np.log2(L+N-1))))
        Ndft = nextpowerof2(L+N-1)
        yzp = list(y) + list(np.zeros(Ndft-N+1))
        hzp = list(h) + list(np.zeros(Ndft-L+1))    
        
        # COMPUTE: FFT of signal and filter in freq domain
        
#       Y = fftpack.fft(yzp)                                        # FFT signal 
#       H = fftpack.fft(hzp)                                        # FFT filter
        Y = dft(yzp)                                                # DFT signal 
        H = dft(hzp)                                                # DFT filter
        
        # COMPUTE: cyclic convolution (pairwise product) of signal and filter in freq domain
        
        Z = np.multiply(Y, H)
        y_filtered_lo = np.real( idft(Z)[int(N/2):N+int((N)/2)] )   # low pass signal    
        y_filtered_hi = y - y_filtered_lo                           # high pass signal
                
    print('mean(y_hi)',np.nanmean(y_filtered_hi))
    
    return y_filtered_lo + y_mean, y_filtered_hi, zvarlo, zvarhi, fc, pctl

if __name__ == "__main__":
    
    #------------------------------------------------------------------------------
    # LOAD: a station anomaly timeseries
    #------------------------------------------------------------------------------
            
    df_temp = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')    
    da = df_temp[df_temp['stationcode'] == stationcode]
    stationname = da['stationname'].iloc[0]
    da = da[da.year >= 1678].reset_index(drop=True)
    db = da.copy()
    for j in range(1,13):
        baseline = np.nanmean(np.array(da[(da['year'] >= 1981) & (da['year'] <= 2010)].iloc[:,j]).ravel())
        db.loc[da.index.tolist(), str(j)] = da[str(j)]-baseline                        
    ts = np.array(db.groupby('year').mean().iloc[:,0:12]).ravel()    
    t = pd.date_range(start=str(db.year.iloc[0]), periods=len(ts), freq='MS')
    df = pd.DataFrame({'Tg':ts}, index=t)        

    # RESAMPLE: to yearly
        
    df_xr = df.to_xarray()    
    df_xr_resampled = df_xr.Tg.resample(index='AS').mean().to_dataset()      
    y = df_xr_resampled.Tg.values
    t = df_xr_resampled.Tg.index
        
    #------------------------------------------------------------------------------
    # COMPUTE: DFT low and high pass --> y_lo, y_hi
    #------------------------------------------------------------------------------

    y_lo, y_hi, zvarlo, zvarhi, fc, pctl = dft_filter(y, w)

    if make_plot:

        figstr = 'ts-dft-filter' + '-' + str(w) + '.png'
        
        fig, ax = plt.subplots(figsize=(15,10))
        plt.subplot(211)
        plt.plot(t, y, ls='-', lw=1, color='blue', alpha=0.5, label='signal: yearly')
        if show_pandas == True:
            if w >= 2:                
                T = w                
            else:    
                T = int(1/fc)            
            plt.plot(t, pd.Series(y).rolling(T,center=True).mean(), ls='-', lw=3, color='red', alpha=1, label='Pandas rolling(' + str(T) + ')')
        plt.plot(t, y_lo, ls='-', lw=3, color='blue', label=r'DFT low pass ($\nu$=' + str(np.round(zvarlo*100.0,2)) + '%)' + ': w=' + str(w) + ', fc=' + str(np.round(fc,2)) + r', P$\geq$p(' + str(np.round(pctl,1)) + ')')        
        plt.ylim(-3,2)
        plt.tick_params(labelsize=fontsize)    
        plt.ylabel(r'T(2m) anomaly (from 1981-2010), $^{\circ}$C', fontsize=fontsize)
        plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
        plt.subplot(212)
        plt.plot(t, y_hi, ls='-', lw=1, color='teal', alpha=1, label=r'DFT high pass ($\nu$=' + str(np.round(zvarhi*100.0,2)) + '%)' + r': $\mu$=' + str(np.round(np.mean(y_hi),2)) + r'$^{\circ}$C')
        plt.axhline(y=np.nanmean(y_hi), ls='--', lw=1, color='black' )
        plt.ylim(-3,2)
        plt.tick_params(labelsize=fontsize)    
        plt.xlabel('Time', fontsize=fontsize)
        plt.ylabel(r'T(2m) anomaly (from 1981-2010), $^{\circ}$C', fontsize=fontsize)
        plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
#       plt.suptitle(stationname + ' (' + stationcode + ')')
        plt.suptitle('HadCET (' + stationcode + ')')
        fig.tight_layout()
        plt.savefig(figstr, dpi=300)
        plt.close('all')    

#------------------------------------------------------------------------------
print('** END')


















    
    
