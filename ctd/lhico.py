
"""
Collection of scripts used by the LHiCo on pre treatment CTD casts.
"""

import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy import signal as sign
from scipy.stats import linregress

from ctd.processing import _rolling_window

from pandas_flavor import register_dataframe_method, register_series_method

@register_series_method
@register_dataframe_method
def loopedit2(df):
    """
    Remove values with negative pressure gradient.

    Credits
    -------
    Function extracted from an old OceanLab version:
                  https://github.com/iuryt/OceanLab
    """
    df_new = df.copy()

    try:
        flag = df_new['dz/dtM'].values>0
        df_new = df_new.iloc[flag,:]
    except:
        flag = np.hstack([1,np.diff(df_new.index.values)])>0
        df_new = df_new.iloc[flag,:]

    return df_new

@register_series_method
@register_dataframe_method
def longest_pressure(df, thresh=2, indexname='Pressure [dbar]'):
    """Separates the dataframe based into pieces based on a pressure gradient
    threshold and select the longest one.

    Parameters
    ----------
    data : pandas DataFrame
        Pandas dataframe with the ctd data.
        Notice the index must be the pressure values.
    thresh : integer or float
        gradient threshold used to separate the dataframe
    indexname  : str
        The cast must be indexed by pressure, so this flag can be used to make sure that is the case.
    Returns
    -------
    pandas DataFrame
        DataFrame with the data selected with the longest pressure vector

    """
    # TODO: adicionar flag para acionar o index, caso este seja pressure, ou uma coluna de pressure
    df_new = df.copy()

    # -- find cut positions where the pressure surpasses a given threshold -- #
    i = np.where(abs(np.gradient(df_new.index))>thresh)[0]
    df_new.iloc[i] = np.nan  # substitute values in cut positions with nan

    # -- identify positions with nan and give a integer id for each section -- #
    df_new['group'] = df_new.isnull().all(axis=1).cumsum()
    groups = df_new.groupby('group')
    gcount = groups.count()  # counting the number of elements in each group
    gcount1 = gcount[gcount==gcount.max()].dropna()  # select the largest one
    # -- select the largest group based on the integer id -- #
    df_new = groups.get_group(gcount1.index.values[0]).dropna()

    return df_new

@register_series_method
@register_dataframe_method
def downup_cast(df, winsize=500, thresh=0.02):
    
    df_new = df.copy()
    
    if df_new.shape[0] > winsize:
        df_new = _downcast_upcast(df_new, winsize=winsize, thresh=thresh)
    else:
        df_new = _downcast_upcast(df_new, winsize=50, thresh=0.012)
        
    return df_new

@register_series_method
@register_dataframe_method
def bindata2(df, reference='depSM', delta=1.):
    df_new = df.copy()
    
    df_new = _binmov(df, reference=reference, delta=delta)
    
    return df_new

@register_dataframe_method
def check_pump(df, pumpname='pumps'):
    df_new = df.copy()
    return df_new[df_new[pumpname] != 0]

@register_dataframe_method
def despike_(df, weight=2):
    
    df_new = df.copy()

    # get block size based on the cast shape
    block = _block_size(df_new.shape[0])
    
    # if series, use function directly. Otherwise, apply using .apply()
    if isinstance(df_new, pd.Series):
        df_new = _despike(df_new, block=block, weight=weight)
    else:
        indexname = df_new.index.name
        
        if indexname in df_new.columns:
            df_new = df_new.reset_index(drop=True)
        else:
            df_new = df_new.reset_index()
            
        df_new = df_new.apply(_despike, block=block, weight=weight)
        
        df_new.set_index(indexname)
    
    return df_new

#-- auxiliar functions --#

def _block_size(shp):
    # check the half size
    if (shp < 105) & (shp > 10):  # perfil for com menos de 105 medidas
        # if cast has less than 105 and even number observations 
        if (shp//2)%2 == 0:   
            # block is defined as the half + 1
            block = (shp//2)+1
        else:
            # if block has les than 105 and odd number of observations, then the block is the half
            block = shp//2
    # if cast has more than 105 observations, then the block is precisely 105
    elif shp >= 105:
        block = 105
    else:
        block = None
        
    return block

def _despike(series, block=100, weight=2):
    #  weight is a factor that controls the cutoff threshold
    prop = np.array(series)

    roll = _rolling_window(prop, block)
    std = weight * roll.std(axis=1)
    mean = roll.mean(axis=1)

    # gambiarra p séries com número par de elementos
    if block % 2 == 1:
        iblock0 = int((block - 1)/2)
        iblock1 = int((block - 1)/2)
    else:
        iblock0 = int((block - 1)/2)
        iblock1 = int((block - 1)/2)+1

    std = np.hstack([np.tile(std[0], iblock0),
                      std,
                      np.tile(std[-1], iblock1)])

    mean = np.hstack([np.tile(mean[0], iblock0),
                       mean,
                       np.tile(mean[-1], iblock1)])

    series = series[np.abs(series-mean) < std]
    
    clean = series.astype(float).copy()

    return clean

def _binmov(df, reference='depSM', delta=1.):
    
    indd = np.round(df[reference].values)
    binned = df.groupby(indd).mean()
    binned[reference] = binned.index.values

    return binned

def _local_slope(value):
    d = value - sign.detrend(value.values)
    slope = linregress(np.arange(d.size), d)[0]
    return slope

def _downcast_upcast(data, winsize=500, direction='down', thresh=0.02):
    """
        TODO - ADD DOCSTRING
    """
    df = pd.DataFrame(data.index)
    df = pd.DataFrame(data.index)
    # -- extend dataframe to account for blackman window size -- #
    index = df.index
    bsize = np.floor(winsize/2)
    if winsize % 2 == 0:
        reindex = np.arange(index[0]-bsize,index[-1]+bsize)
    else:
        reindex = np.arange(index[0]-bsize,index[-1]+1+bsize)

    # 'Extrapol.'
    filt_na =  df.reindex(index=reindex)
    filt_na =  filt_na.interpolate(limit_direction='both')

    trend = filt_na.rolling(winsize, center=True).apply(_local_slope)
    trend = trend.dropna()
    # i = np.where((trend>0) & (np.gradient(trend)>0))[0]
    if direction=='down':
        i = np.where((trend>thresh))[0]
        dataaux = data.iloc[i]
    elif direction=='up':
        i = np.where((trend<-thresh))[0]
        dataaux = data.iloc[i]
    else:
        raise IOError('wrong  direction input')
    return dataaux
