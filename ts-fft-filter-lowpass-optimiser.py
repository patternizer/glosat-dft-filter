#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: ts-fft-filter-lowpass-optimiser.py
#------------------------------------------------------------------------------
# Version 0.1
# 26 July, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy import fftpack
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

fc = 0 # --> estimate from spectrum    
fontsize = 16
use_filter = True
make_plot = True
make_plot_loop = True

stationcode = '037401'
   
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
    
def fft_filter(y, fc, use_filter):

    N = len(y)                                                      # signal length
    n = np.arange(N)
    Fs = 1                                                          # sampling rate
    Ts = N/Fs                                                       # sampling interval 
    freq = n/Ts                                                     # frequencies [0,1] 
       
    y_mean = np.nanmean(y)
    y = y - y_mean                                                  # cneter timeseries (NB: add back in later)
            
    z = dft(y)                                                      # DFT
    zamp = np.sqrt(np.real(z)**2.0 + np.imag(z)**2.0) / N           # ampplitudes
    zphase = np.arctan2( np.imag(z), np.real(z) )                   # phases

    zvar = zamp**2                                                  # variance of each harmonic
    zvarsum = np.sum(zvar)                                          # total variance of spectrum
    yvarsum = np.var(y)                                             # total variance of timeseries
    zvarpc = zvar / zvarsum                                         # percentage variance per harmonic

    y_hi_mean_min = 1.0
    zvarhimax = 0.0
    
    vec_pctl = []
    vec_fc = []
    vec_zvarlo = []
    vec_zvarhi = []
    vec_y_hi_mean = []
        
    for pctl in np.arange(0.0,100.0,0.1):
                               
        zpeaks = zvarpc[ zvarpc > ( np.percentile(zvarpc, pctl) ) ]     # high variance peaks > q(pctl)
        zpeaks_idx = np.argsort( zpeaks )                               # peak indices
        znopeaks_idx = np.setdiff1d( np.argsort( zvarpc ), zpeaks_idx)  # remaining indices
        npeaks = len(zpeaks_idx)                                        # number of peaks
        zvarlo = np.sum( [zvarpc[i] for i in zpeaks_idx] )              # total percentage variance of low pass 
        zvarhi = np.sum( [zvarpc[i] for i in znopeaks_idx] )            # total percentage variance of high pass  

        fc = freq[ zpeaks_idx.max() ]                                   # low pass / high pass cut-off       
                
        if use_filter == False:
            
            # FFT zero-order estimate: low pass filter (no window)
             
            y_fft = fftpack.fft(y)
            y_fft_filtered = y_fft.copy()
            frequencies = fftfreq(len(y), d=1)    
            y_fft_filtered[ np.abs(frequencies) > fc ] = 0               # low-pass filter
            y_filtered_lo = np.real( fftpack.ifft(y_fft_filtered) )      # low pass signal
            y_filtered_hi = y - y_filtered_lo                            # high pass signal
    
        else:
            
            # FILTER DESIGN: low pass filter (Hamming window)
            
            L = N+1                                                     # filter length (M+1)
            h_support = np.arange( -int((L-1)/2), int((L-1)/2)+1 )      # filter support
            h_ideal = ( 2*fc/Fs) * np.sinc( 2*fc*h_support/Fs )         # filter (ideal)
            h = np.hamming(L).T*h_ideal                                 # filter
            
            # ZERO-PAD: (next power of 2 > L+M-1) signal and impulse-response
            
            #Nfft = int(2**(np.ceil(np.log2(L+N-1))))
            Nfft = nextpowerof2(L+N-1)
            yzp = list(y) + list(np.zeros(Nfft-N+1))
            hzp = list(h) + list(np.zeros(Nfft-L+1))    
            
            # COMPUTE: FFT of signal and filter in freq domain
            
            Y = fftpack.fft(yzp)                                        # FFT signal 
            H = fftpack.fft(hzp)                                        # FFT filter
            
            # COMPUTE: cyclic convolution (pairwise product) of signal and filter in freq domain
            
            Z = np.multiply(Y, H)
            y_filtered_lo = np.real( fftpack.ifft(Z)[int(N/2):N+int((N)/2)] ) # low pass signal    
            y_filtered_hi = y - y_filtered_lo                                 # high pass signal

        y_hi_mean = np.nanmean(y_filtered_hi)
                           
        print('pctl=', pctl)            
        print('fc', fc)
        print('zvarlo', zvarlo)
        print('zvarhi', zvarhi)
        print('y_hi_mean', y_hi_mean)

        vec_pctl.append(np.round(pctl,1))
        vec_fc.append(fc)
        vec_zvarlo.append(zvarlo)
        vec_zvarhi.append(zvarhi)
        vec_y_hi_mean.append(y_hi_mean)

        if make_plot_loop:
        
            figstr = 'ts-fft-filter' + '-' + 'pctl' + '-' + str(np.round(pctl,1)) + '.png'
                            
            fig, ax = plt.subplots(figsize=(15,10))
            plt.subplot(211)
            plt.plot(t, y, ls='-', lw=1, color='blue', alpha=0.5, label='signal: fc=' + str(np.round(fc,3)) + ' Hz')     
            plt.plot(t, y_filtered_lo, ls='-', lw=3, color='blue', label=r'FFT low pass ($\nu$=' + str(np.round(zvarlo*100.0,3)) + r'%): P$\geq$p(' + str(np.round(pctl,1)) + ')')    
            plt.ylim(-3,2)
            plt.tick_params(labelsize=fontsize)    
            plt.ylabel(r'T(2m) anomaly (from 1981-2010), $^{\circ}$C', fontsize=fontsize)
            plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
            plt.subplot(212)
            plt.plot(t, y_filtered_hi, ls='-', lw=1, color='teal', alpha=1, label=r'FFT high pass ($\nu$=' + str(np.round(zvarhi*100.0,3)) + '%)')    
            plt.ylim(-3,2)
            plt.axhline(y=y_hi_mean, ls='--', lw=1, color='black', label=r'$\mu$=' + str(np.round(y_hi_mean,3)) )
            plt.tick_params(labelsize=fontsize)    
            plt.xlabel('Time', fontsize=fontsize)
            plt.ylabel(r'T(2m) anomaly (from 1981-2010), $^{\circ}$C', fontsize=fontsize)
            plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
#           plt.suptitle(stationname + ' (' + stationcode + ')')
            plt.suptitle('HadCET (' + stationcode + ')')
            fig.tight_layout()
            plt.savefig(figstr, dpi=300)
            plt.close('all')    
                
        if (zvarhi > zvarhimax):
            
            zvarhimax = zvarhi
            
            y_filtered_lo_best = y_filtered_lo
            y_filtered_hi_best = y_filtered_hi
            zvarlo_best = zvarlo
            zvarhi_best = zvarhi
            pctl_best = pctl
            fc_best = fc
            
        else:
            
            continue
        
    y_filterred_lo = y_filtered_lo_best
    y_filterred_hi = y_filtered_hi_best
    zvarlo = zvarlo_best
    zvarhi = zvarhi_best
    pctl = pctl_best
    fc = fc_best

    df = pd.DataFrame({'pctl':vec_pctl, 'fc':vec_fc, 'zvarlo':vec_zvarlo, 'zvarhi':vec_zvarhi, 'y_hi_mean':vec_y_hi_mean})
    df['period'] = 1.0/df['fc']
    df.to_csv('ml_optimisation.csv')                       
            
    print('fc_best=', fc)
    print('pctl_best=', pctl)
            
    return y_filtered_lo + y_mean, y_filtered_hi, zvarlo, zvarhi, pctl, fc

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
    # COMPUTE: FFT low and high pass --> y_lo, y_hi
    #------------------------------------------------------------------------------

    y_lo, y_hi, zvarlo, zvarhi, pctl, fc = fft_filter(y,0,use_filter)

    if make_plot:

        y_hi_mean = np.nanmean(y_hi)
        
        fig, ax = plt.subplots(figsize=(15,10))
        plt.subplot(211)
        plt.plot(t, y, ls='-', lw=1, color='blue', alpha=0.5, label='signal: fc=' + str(np.round(fc,3)) + ' Hz')     
        plt.plot(t, y_lo, ls='-', lw=3, color='blue', label=r'FFT low pass ($\nu$=' + str(np.round(zvarlo*100.0,3)) + r'%): P$\geq$p(' + str(np.round(pctl,1)) + ')')    
        plt.ylim(-3,2)
        plt.tick_params(labelsize=fontsize)    
        plt.ylabel(r'T(2m) anomaly (from 1981-2010), $^{\circ}$C', fontsize=fontsize)
        plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
        plt.subplot(212)
        plt.plot(t, y_hi, ls='-', lw=1, color='teal', alpha=1, label=r'FFT high pass ($\nu$=' + str(np.round(zvarhi*100.0,3)) + '%)')    
        plt.ylim(-3,2)
        plt.axhline(y=y_hi_mean, ls='--', lw=1, color='black', label=r'$\mu$=' + str(np.round(y_hi_mean,3)) )
        plt.tick_params(labelsize=fontsize)    
        plt.xlabel('Time', fontsize=fontsize)
        plt.ylabel(r'T(2m) anomaly (from 1981-2010), $^{\circ}$C', fontsize=fontsize)
        plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
#       plt.suptitle(stationname + ' (' + stationcode + ')')
        plt.suptitle('HadCET (' + stationcode + ')')
        fig.tight_layout()
        plt.savefig('ts-fft-filter.png', dpi=300)
        plt.close('all')    

#------------------------------------------------------------------------------
print('** END')


















    
    
