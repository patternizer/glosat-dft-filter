#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: ts-fft-filter-lowpass-optimiser.py
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
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

stationcode = '037401'

fontsize = 16
make_plot_loop = False                                           # True --> loop over pctl [0,100,0.1] to generate LUT
use_filter = True                                               # True --> Hamming window, False --> cut-off (no window)
show_pandas = True                                              # Overlay equi-window Pandas smoother
   
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
    
def dft_filter(y):

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
    
    # GENERATE: look-up table pctl <--> fc <--> zvarlo & zvarhi

    vec_pctl = []
    vec_fc = []
    vec_zvarlo = []
    vec_zvarhi = []
    vec_y_hi_mean = []
        
    for pctl in np.arange(0,100,0.1):
                               
        zpeaks = zvarpc[ zvarpc > ( np.percentile(zvarpc, pctl) ) ]     # high variance peaks > p(pctl)
        zpeaks_idx = np.argsort( zpeaks )                               # peak indices
        znopeaks_idx = np.setdiff1d( np.argsort( zvarpc ), zpeaks_idx)  # remaining indices
        npeaks = len(zpeaks_idx)                                        # number of peaks
        zvarlo = np.sum( [zvarpc[i] for i in zpeaks_idx] )              # total percentage variance of low pass 
        zvarhi = np.sum( [zvarpc[i] for i in znopeaks_idx] )            # total percentage variance of high pass  

        fc = f[ zpeaks_idx.max() ]                                      # low pass / high pass cut-off       
                
        print('pctl',pctl)
        print('fc',fc)
        
        if use_filter == False:
            
            # DFT zero-order estimate: low pass filter (no window)
             
            y_dft = dft(y)
            y_dft_filtered = y_dft.copy()
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
            
            #Nfft = int(2**(np.ceil(np.log2(L+N-1))))
            Ndft = nextpowerof2(L+N-1)
            yzp = list(y) + list(np.zeros(Ndft-N+1))
            hzp = list(h) + list(np.zeros(Ndft-L+1))    
            
            # COMPUTE: FFT of signal and filter in freq domain
            
            Y = dft(yzp)                                                # DFT signal 
            H = dft(hzp)                                                # DFT filter
            
            # COMPUTE: cyclic convolution (pairwise product) of signal and filter in freq domain
            
            Z = np.multiply(Y, H)
            y_filtered_lo = np.real( idft(Z)[int(N/2):N+int((N)/2)] )   # low pass signal    
            y_filtered_hi = y - y_filtered_lo                           # high pass signal

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
        
            figstr = 'ts-dft-filter' + '-' + 'pctl' + '-' + str(np.round(pctl,1)) + '.png'
                            
            fig, ax = plt.subplots(figsize=(15,10))
            plt.subplot(211)
            plt.plot(t, y, ls='-', lw=1, color='blue', alpha=0.5, label='signal: yearly')                             
            if show_pandas == True:
                if fc < 1e-16:
                    T = N
                else:
                    T = int(1/fc)            
                plt.plot(t, pd.Series(y).rolling(T,center=True).mean(), ls='-', lw=3, color='red', alpha=1, label='Pandas rolling(' + str(T) + ')')            
            plt.plot(t, y_filtered_lo, ls='-', lw=3, color='blue', label=r'DFT low pass ($\nu$=' + str(np.round(zvarlo*100.0,2)) + '%)' + ': T=' + str(T) + ', fc=' + str(np.round(fc,2)) + r', P$\geq$p(' + str(np.round(pctl,1)) + ')')   
            plt.ylim(-3,2)
            plt.tick_params(labelsize=fontsize)    
            plt.ylabel(r'T(2m) anomaly (from 1981-2010), $^{\circ}$C', fontsize=fontsize)
            plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
            plt.subplot(212)
            plt.plot(t, y_filtered_hi, ls='-', lw=1, color='teal', alpha=1, label=r'DFT high pass ($\nu$=' + str(np.round(zvarhi*100.0,2)) + '%)' + r': $\mu$=' + str(np.round(np.mean(y_filtered_hi),2)) + r'$^{\circ}$C')
            plt.axhline(y=np.nanmean(y_filtered_hi), ls='--', lw=1, color='black' )
            plt.ylim(-3,2)
            plt.tick_params(labelsize=fontsize)    
            plt.xlabel('Time', fontsize=fontsize)
            plt.ylabel(r'T(2m) anomaly (from 1981-2010), $^{\circ}$C', fontsize=fontsize)
            plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
#           plt.suptitle(stationname + ' (' + stationcode + ')')
            plt.suptitle('HadCET (' + stationcode + ')')
            fig.tight_layout()
            plt.savefig(figstr, dpi=300)
            plt.close('all')    

    df = pd.DataFrame({'pctl':vec_pctl, 'fc':vec_fc, 'zvarlo':vec_zvarlo, 'zvarhi':vec_zvarhi, 'y_hi_mean':vec_y_hi_mean})
    df['period'] = 1.0/df['fc']
    df.to_csv('ml_optimisation.csv')                       
                        
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

    y_lo, y_hi, zvarlo, zvarhi, pctl, fc = dft_filter(y)
    
    #------------------------------------------------------------------------------
    # PLOT: optimisation parameters
    #------------------------------------------------------------------------------

    df = pd.read_csv('OUT/ml_optimisation.csv')
    df['y_hi_mean'][df['fc']==0] = np.nan
            
    figstr = 'ts-dft-filter' + '-' + 'optimisation.png'
                            
    fig, ax = plt.subplots(figsize=(15,10))
    plt.subplot(211)
    plt.plot(df.pctl, df.fc, ls='-', lw=3, color='purple', alpha=1, label='fc')                             
    plt.plot(df.pctl, df.zvarlo, ls='-', lw=3, color='red', label=r'DFT low pass ($\nu$)')   
    plt.plot(df.pctl, df.zvarhi, ls='-', lw=3, color='blue', label=r'DFT high pass ($\nu$)')   
    plt.tick_params(labelsize=fontsize)    
    plt.xlabel('Percentile', fontsize=fontsize)
    plt.ylabel('Magnitude', fontsize=fontsize)
    plt.legend(loc='upper center', ncol=3, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
    plt.subplot(212)
    plt.plot(df.pctl, df.y_hi_mean, ls='-', lw=3, color='teal', alpha=1, label=r'DFT high pass ($\nu$) mean')
    plt.tick_params(labelsize=fontsize)    
    plt.xlabel('Percentile', fontsize=fontsize)
    plt.ylabel('Magnitude', fontsize=fontsize)
    plt.legend(loc='upper center', ncol=3, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)   
#   plt.suptitle(stationname + ' (' + stationcode + ')')
    plt.suptitle('HadCET (' + stationcode + ')')
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')        

#------------------------------------------------------------------------------
print('** END')


















    
    
