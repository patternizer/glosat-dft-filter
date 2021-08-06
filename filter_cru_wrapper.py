#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: filter_cru_wrapper.py
#------------------------------------------------------------------------------
#
# wrapper to load station timeseries and apply CRU Gaussian filter
#
# Version 0.1
# 29 July, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------
import filter_cru as cru # CRU Gaussian filter
#----------------------------------------------------------------------------------

# =======================================
# INCLUDE PLOT CODE:
# exec(open('plot_filter_cru.py').read())
# =======================================

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

fontsize = 16
baseline_start = 1981
baseline_end = 2010
#stationcode = '071560'
stationcode = '744920'
thalf = 300
nan = False
truncate = 0

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

tsin = df_xr_resampled.Tg.values
t = df_xr_resampled.index.values

#------------------------------------------------------------------------------
# CALL: cru_filter
#------------------------------------------------------------------------------

tslow, tshigh, weight = cru.cru_filter(thalf, tsin, nan, truncate)

#------------------------------------------------------------------------------
# PLOT: ts and tslow and tshigh components
#------------------------------------------------------------------------------

figstr = stationcode + '-' + 'gaussian-filter' + '.png'
titlestr = 'GloSAT.p03: ' + stationname + ' (' + stationcode + '): CRU Gaussian filter (thalf=' + str(thalf) + ', nw=' + str(len(weight)) + ')'
               
fig, ax = plt.subplots(figsize=(15,10))    
plt.fill_between(t, 0, tsin, ls='-', lw=3, color='black', alpha=0.1, zorder=1, label=r'tsin')
plt.plot(t, tslow.T, ls='-', lw=3, color='teal', alpha=1, zorder=1, label=r'tslow')
plt.plot(t, tshigh.T, ls='--', lw=1, color='teal', alpha=1, zorder=1, label=r'tshigh')
plt.axhline(y=np.nanmean(tshigh.T), ls='--', lw=1, color='black', alpha=1, zorder=1, label=r'$\mu(tshigh)$')
plt.xlabel('Year', fontsize=fontsize)
#ax.set_xlim(pd.to_datetime('1720-01-01'),pd.to_datetime('2021-01-01'))
plt.ylabel(r'2m Temperature anomaly (from ' + str(baseline_start) + '-' + str(baseline_end) + r'), $^{\circ}$C', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')

#------------------------------------------------------------------------------
print('** END')



