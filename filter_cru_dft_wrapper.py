#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: filter_cru_dft_wrapper.py
#------------------------------------------------------------------------------
#
# wrapper to load station timeseries and apply CRU DFT filter
#
# Version 0.1
# 6 August, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#----------------------------------------------------------------------------------
import filter_cru_dft as cru # CRU DFT filter
#----------------------------------------------------------------------------------

# =======================================
# INCLUDE PLOT CODE:
# exec(open('plot_filter_cru.py').read())
# =======================================

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

w = 5                                                         # smoothing windows >= 2 (Nyquist freq) 
stationcode = '037401'
fontsize = 16
show_pandas = True                                            # Overlay equi-window Pandas smoother
   
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
# CALL: cru_filter_dft
#------------------------------------------------------------------------------

y_lo, y_hi, zvarlo, zvarhi, fc, pctl = cru.cru_filter_dft(y, w)

#------------------------------------------------------------------------------
# PLOT: ts and tslow and tshigh components
#------------------------------------------------------------------------------

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
# plt.suptitle(stationname + ' (' + stationcode + ')')
plt.suptitle('HadCET (' + stationcode + ')')
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')    

#------------------------------------------------------------------------------
print('** END')


