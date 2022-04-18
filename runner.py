# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:36:53 2022

@author: Kharl
"""
import xarray as xr
import pandas as pd

import datetime as dt

from src.features.ERA5_TEMP_PREC_SD import era5_temp_prec_sd_culc
from src.features.ERA5_TEMP_PREC import era5_temp_prec_culc
from src.features.ERA5_SD import era5_sd_culc
from src.features.ERA5_soil_temp import era5_soil_temp_culc

start = dt.datetime.now()
sd_path='data/meteorology/processed/snow_depth_full.nc'
prec_path='data/meteorology/processed/total_precipitation_full.nc'
temp_path='data/meteorology/processed/2m_temperature_full.nc'
soil_temp_path='data/meteorology/processed/fr_depth_full.nc'


#save_path='sd_prec_temp.nc'
print('reading t2m...')
nc_temp = xr.open_dataset(temp_path)        
nc_temp = nc_temp['t2m'] - 273.15

print('reading tp...')
nc_pre = xr.open_dataset(prec_path)        
nc_pre = nc_pre['tp']*1000

print('reading sd...')
nc_sd = xr.open_dataset(sd_path)        
nc_sd = nc_sd['sd'] * 1000

print('reading soil_temp...')
nc_st = xr.open_dataset(soil_temp_path)        
nc_st = nc_st['fr_depth']

print()
print('        temp_prec_sd')
era5_temp_prec_sd_culc(nc_temp, nc_pre, nc_sd.copy(), 
                       'data/meteorology/features_nc/sd_prec_temp.nc', 
                       sample=True)
print()
print('        temp_prec')
era5_temp_prec_culc(nc_temp, nc_pre,  
                             'data/meteorology/features_nc/prec_temp.nc', 
                       sample=True)
print()
print('        sd')
era5_sd_culc(nc_sd.copy(), 'data/meteorology/features_nc/sd.nc', 
                       sample=True)
print()
print('        soil_temp')
era5_soil_temp_culc(nc_st, 'data/meteorology/features_nc/st.nc', 
                       sample=True)

print('total_time: ', dt.datetime.now() - start)