# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:25:07 2020

@author: kharl
"""

import xarray as xr
import pandas as pd
import numpy as np

import geopandas as gpd

import os
from tqdm import tqdm

from joblib import Parallel, delayed

import matplotlib.pyplot as plt

def sel_mon(month):
    return (month >= 3) & (month <= 5)
    #return (month >= 9) & (month <= 11)

def clip_by_shape(xr_file, shape):           #return mask
    ddd = xr_file[list(xr_file.variables)[3]].copy()        
    ddd = ddd.mean(dim = 'time')
    # перевод netcdf в geodataframe
    df = ddd.to_dataframe()
    df = df.reset_index()
    geom=gpd.points_from_xy(df['longitude'], df['latitude'])
    gdf = gpd.GeoDataFrame(df, geometry=geom)
    
    within = []
    #print('make mask')
    for i in range(len(gdf)):
        aaa = gdf['geometry'].loc[i].within(shape['geometry'].loc[0])
        if aaa == False:
            aaa = np.nan
        else:
            aaa = 1
        within.append(aaa)
        
    gdf['within'] = within
    
    #print('make xarray')
    gdf = gdf.set_index(['latitude', 'longitude'])
    nc_mask = gdf.to_xarray()
    xr_masked = xr_file * nc_mask['within']
    
    return nc_mask['within']

def zonalmean_xr(xr_dataset_mask, shape):
    #mask = clip_by_shape_da(xr_dataset, shape)
    #xr_dataset_mask = xr_dataset * mask
    vars_ = list(xr_dataset_mask.variables)
    vars_ = vars_[3:]
    df = pd.DataFrame()
    
    for v in vars_:
        df[v] = xr_dataset_mask[v].mean(dim = ['latitude', 'longitude']).to_pandas()
    return df

def shapestat(SD, station, shapes_path, directory_to_save=''):
    errors = []
    try:
        shape = gpd.read_file(shapes_path + os.sep + station)
        shape = shape.to_crs(epsg=4326)
        
        mask = clip_by_shape(SD, shape)
        
        SD_df = zonalmean_xr(SD*mask, shape = shape)
        
        SD_df.to_csv(directory_to_save +  station[:-4] + '.csv')
    
    except:
        errors.append(station)
        print(station, 'error')
 

def nc_per_shapes(nc, shapes_path, directory_to_save, n_jobs=1):
    '''

    Parameters
    ----------
    nc : netcdf file
        netcdf file with features
    shapes_path : str
        path to folder with shapes
    directory_to_save : str
        path were zonal statistic data will be save
    n_jobs : int
        number of cpu cores. The default is 1.

    Returns
    -------
    None.

    '''

    if not os.path.exists(directory_to_save): os.makedirs(directory_to_save)

    shapes = os.listdir(shapes_path)
    shapes = [x for x in shapes if x.endswith('.shp')]

    Parallel(n_jobs=n_jobs)(delayed(shapestat)(nc, station, shapes_path, directory_to_save) 
                            for station in tqdm(shapes, desc = 'zonal mean...'))
    


