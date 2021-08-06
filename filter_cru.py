import numpy as np

def cru_filter(thalf, tsin, nan, truncate):
    
    ''' 
    -----------------------------------------------------------------------------
    translation of filter_cru.pro from IDL --> python 3.8+
    -----------------------------------------------------------------------------
    Version 0.1
    29 July, 2021	
    Michael Taylor
    https://patternizer.github.io
    patternizer AT gmail DOT com
    michael DOT a DOT taylor AT uea DOT ac DOT uk
    #------------------------------------------------------------------------------
    Uses the CRU Gaussian weighted filter to separate the low and high frequency
    components of a timeseries (or a set of time series of equal length).

    thalf          : period of oscillation that is reduced in amplitude by 50%
    tsin           : input timeseries (1D or 2D, if the latter then assume order is [SERIES,TIME]
    nan            : If this flag is set, missing values in tsin are replaced by the timeseries local mean before filtering, and then 
                     re-replaced by the missing code afterwards (and in the high and low frequency series too)
    truncate       : 0=pad with data and filter right to ends,
                   : 1=truncate filtered series to stop (thalf-1)/4 from end
                   : 2=truncate filtered series to stop (thalf-1)/2 from end

                     NB, truncate only affects the ends of the series, not missing data within the series, which can be "truncated"
                     by nan=0 (though with less control).  If a series has missing data past the start or end of the series already,
                     then TRUNCATE will not truncate the actual data values, but only the values from the ends of the complete series
                     including missing code.
                     
    tslow,tshigh   : low and high frequency components
    weight         : optionally return weights used
    ----------------------------------------------------------------------------- 
    CALL SYNTAX: tslow, tshigh, weight = cru_filter(thalf, tsin, nan, truncate)                                 
    ----------------------------------------------------------------------------- 
    '''

    # DEFINE: arrays

    tssize = np.shape(tsin)
    if len(tssize) == 1:
        nx = 1
        nt = tssize[0]
    elif len(tssize) == 2:
        nx = tssize[0]
        nt = tssize[1]
    tslow = np.zeros([nx,nt])
    tshigh = np.zeros([nx,nt])
    weight = []

    # COMPUTE: number of weights required

    nw = int(thalf/2.5 + 0.5)
    if (nw % 2) == 0: 
        nw = nw + 2 
    else: 
        nw = nw + 1
    if nw < 7:
        nw = 7
    weight = np.zeros(nw)

    # COMPUTE: weights

    wfactor = -18.0 / (thalf**2)
    wroot = 1.0 / np.sqrt(2.0*np.pi)
    weight[0] = wroot
    wsum = weight[0]
    for i in range(1,nw):        
        weight[i] = wroot * np.exp( wfactor*float(i)*float(i) )
        wsum = wsum + 2.0*weight[i]
    weight = weight / wsum

    # PAD: timeseries with local mean for missing values if nan = True

    tspad = tsin
    if nan == True:

        for ix in range(nx):            
            if nx == 1:
                misslist = np.argwhere(np.isnan( tsin ))
            elif nx > 1:
                misslist = np.argwhere(np.isnan( tsin[ix,:] ))
            nmiss = len(misslist)
            for i in range(nmiss): 
                ele1 = (misslist[i]-nw+1)[0]
                ele2 = (misslist[i]+nw-1)[0]
                if (ele1 > 0) & (ele2 < (nt-1)):
                    if nx == 1:
                        locvals = tsin[ele1:ele2]
                        locmean = np.nanmean( locvals )
                        tspad[misslist[i]] = locmean
                    elif nx > 1:                  
                        locvals = tsin[ix,ele1:ele2]
                        locmean = np.nanmean( locvals )
                        tspad[ix,misslist[i]] = locmean

    # EXTEND: ends of timeseries by mean from each end

    nend = nw-1                       
    tspad2 = np.zeros([nx,nt+2*nend])

    meanst = np.sum(tspad[0:nend-1])/float(nend)    
    meanen = np.sum(tspad[0:nt-nend:nt-1])/float(nend)
        
    for ix in range(nx):
        if nx == 1:
            tspad2 = np.array(list(np.tile(meanst,nend)) + list(tspad) + list(np.tile(meanen,nend)))
        elif nx > 1:
            tspad2[ix,:] = np.array(list(np.tile(meanst,nend)) + list(tspad) + list(np.tile(meanen,nend)))
    tspad=tspad2

    # APPLY: the filter

    for i in range(nt):

        if nx == 1:            
            wsum = weight[0]*tspad[i+nend]
            for j in range(nw):
                wsum = wsum + weight[j]*(tspad[i+nend-j]+tspad[i+nend+j])            
                tslow[0,i] = wsum
                        
        elif nx > 1:
            wsum = weight[0]*tspad[:,i+nend]            
            for j in range(nw):
                wsum = wsum + weight[j]*(tspad[:,i+nend-j]+tspad[:,i+nend+j])            
                tslow[:,i] = wsum

    # Compute the residual (high-frequency) component
    
    tshigh = tsin - tslow

    # INSERT: missing values if required

    if nan == True:
        
        for ix in range(nx):            
            if nx == 1:
                misslist = np.argwhere(np.isnan( tsin ))
            elif nx > 1:
                misslist = np.argwhere(np.isnan( tsin[ix,:] ))
            nmiss = len(misslist)
            if nmiss > 0:
                for i in range(nmiss):                 
                    if nx == 1:
                        tslow[misslist[i]] = np.nan
                        tshigh[misslist[i]] = np.nan
                    elif nx > 1:                                  
                        tslow[ix,misslist[i]] = np.nan
                        tshigh[ix,misslist[i]] = np.nan

    # Truncate ends of the filtered series if required

    if truncate:
              
        nend = (thalf-1.0)/2.0
        if truncate == 1:
            nend = nend/2.0
        nend = int(nend)
        if nend > 0:
          
            if nx == 1:
                tshigh[0:nend-1] = np.nan
                tshigh[nt-nend:] = np.nan
                tslow[0:nend-1] = np.nan
                tslow[nt-nend:] = np.nan                
            if nx > 1:
                tshigh[:,0:nend-1] = np.nan
                tshigh[:,nt-nend:] = np.nan
                tslow[:,0:nend-1] = np.nan
                tslow[:,nt-nend:] = np.nan

    return tslow, tshigh, weight
#------------------------------------------------------------------------------
