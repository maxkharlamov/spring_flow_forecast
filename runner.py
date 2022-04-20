# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:36:53 2022

@author: Kharl
"""
import xarray as xr
import pandas as pd

import datetime as dt

import os

from src.features.ERA5_TEMP_PREC_SD import era5_temp_prec_sd_culc
from src.features.ERA5_TEMP_PREC import era5_temp_prec_culc
from src.features.ERA5_SD import era5_sd_culc
from src.features.ERA5_soil_temp import era5_soil_temp_culc

from src.tools.nc_per_shapes import nc_per_shapes


sd_path='data/meteorology/processed/snow_depth_full.nc'
prec_path='data/meteorology/processed/total_precipitation_full.nc'
temp_path='data/meteorology/processed/2m_temperature_full.nc'
soil_temp_path='data/meteorology/processed/fr_depth_full.nc'

features_nc_path = 'data/meteorology/features_nc/'
features_to_save = 'features/'
shapes = 'data/shapes/wpol/'

hydrology_path = 'data/hydrology/wpol_all.xlsx'
#   готово

#   target hydrology
#   nc_per_shapes       -> оптимизация

#   dataset
#   all
#   train, test ???

#TODO

#   features
# 
#   1. swvl 
#   2. soil types
#   3. palmer?
#   4. regional features from camels and from land_use etc.
#   5. shape parametres

#   model
#
#   feature importance
#   CV
#   map of accuracy
#    

#   catboost
#   xgboost
#   lightgbm
#   stack


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

# features
start = dt.datetime.now()
print('temp_prec_sd')
era5_temp_prec_sd_culc(nc_temp, nc_pre, nc_sd.copy(), 
                       'data/meteorology/features_nc/sd_prec_temp.nc', 
                       sample=True)

print('temp_prec')
era5_temp_prec_culc(nc_temp, nc_pre,  
                             'data/meteorology/features_nc/prec_temp.nc', 
                       sample=True)

print('sd')
era5_sd_culc(nc_sd.copy(), 'data/meteorology/features_nc/sd.nc', 
                       sample=True)

print('soil_temp')
era5_soil_temp_culc(nc_st, 'data/meteorology/features_nc/st.nc', 
                       sample=True)

print('features_total_time: ', dt.datetime.now() - start)
''''''

''''''
# features_to_csv
features = os.listdir(features_nc_path)
for feature in features:
    nc_feature = xr.open_dataset(features_nc_path + feature)
    nc_per_shapes(nc=nc_feature, shapes_path=shapes, 
                  directory_to_save=features_to_save + feature[:-3] + '/')


# hydro preparing

df_hydro = pd.read_excel(hydrology_path, index_col='year')
df_hydro = df_hydro.melt(ignore_index=False)
df_hydro.columns = ['station', 'wpol']
df_hydro['wpol'] = pd.to_numeric(df_hydro['wpol'], errors='coerce')

df_hydro = df_hydro.dropna()
df_hydro = df_hydro.reset_index()
df_hydro['station'] = pd.to_numeric(df_hydro['station'])

#meteo to one file

features = os.listdir('features/')
meteo_features = []
for feature in features:
    path = 'features/' + feature + '/'
    union = []
    for file in os.listdir(path):
        df_meteo = pd.read_csv(path + file)
        df_meteo['time'] = df_meteo['time'].apply(lambda a: int(a[:4]))
        df_meteo['station'] = float(file[:-4])
        union.append(df_meteo)
    feature_df = pd.concat(union, axis=0)
    meteo_features.append(feature_df)
meteo_features = pd.concat(meteo_features, axis = 1)    
df = meteo_features.copy()
df = df.loc[:,~df.columns.duplicated()]

# make dataset
# hydro vs meteo
dataset = pd.merge(left=df_hydro, right=df, how='left',       
                   left_on=['year', 'station'], right_on=['time', 'station'])







