# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 04:02:32 2020

@author: kharl
"""
import numpy as np
import pandas as pd
import xarray as xr

from os import listdir
from itertools import product
from tqdm import tqdm

from multiprocessing import Pool
import multiprocessing as mp

from joblib import Parallel, delayed


def fr_depth_stat(df_cut):
# =============================================================================
#     Функция для расчета характеристик промерзшей почвы
#     
# =============================================================================
    #df_cut = df_new[df_new['year'] == 2000]
    
    df_cut = df_cut.drop(['year'], axis = 1)
    df_stat = pd.DataFrame()
    
    df_frdepth = df_cut[df_cut['fr_depth'] > 0]
    df_stat['fr_depth_days'] = [len(df_frdepth)]
    df_stat['fr_depth_max'] = [df_frdepth['fr_depth'].max()]
    df_stat['fr_depth_mean'] = [df_frdepth['fr_depth'].mean()]
    
    year_cut = df_cut.index.year[0]
    df_stat['fr_depth_21-31.01'] = df_cut[str(year_cut+1) + '-01-21' : str(year_cut+1) + '-01-31']['fr_depth'].mean()
    df_stat['fr_depth_21-28.02'] = df_cut[str(year_cut+1) + '-02-21' : str(year_cut+1) + '-02-28']['fr_depth'].mean()
    df_stat['fr_depth_21.03-31.03'] = df_cut[str(year_cut+1) + '-03-21' : str(year_cut+1) + '-03-31']['fr_depth'].mean()
    df_stat['fr_depth_21.04-31.04'] = df_cut[str(year_cut+1) + '-04-21' : str(year_cut+1) + '-04-30']['fr_depth'].mean()
    
    return df_stat
    
def sd_stat_groupby(df_pre):
# =============================================================================
#     В данной функции размечаем наши данные по полю year (задаем кастомный год)
#     Запускаем groupby
#     Меняем индексы в итоговой таблице
# =============================================================================
    df_new = df_pre.copy()
    df_new['year'] = df_new.index.year
    df_new.loc[df_new.index.month >= 8, 'year'] += 1    
     
    df_gr = df_new.groupby(df_new['year']).apply(fr_depth_stat)
               
    df_gr['time'] = pd.date_range(start = str(df_gr.index[0][0]) + '-01-01', freq = 'AS', periods = len(df_gr))

    df_gr['longitude'] = df_pre['longitude'].unique()[0]
    df_gr['latitude'] = df_pre['latitude'].unique()[0]
    
    df_gr = df_gr.set_index(['time', 'latitude', 'longitude'])
    
    return df_gr

def make_list(nc_st, list_ij):
    nc_sd = nc_st[:, list_ij[0], list_ij[1]].to_dataframe() 
    return nc_sd


def era5_soil_temp_culc(nc_st, save_path, sample=False):
    
    xarray_list_mp = []
    
    if sample==True:
        for i in tqdm(range(nc_st.shape[1]), desc = 'make list'):         
            for j in range(5):
                xarray_list_mp.append([i, j])
    else:
        for i in tqdm(range(nc_st.shape[1]), desc = 'make list'):         
            for j in range(nc_st.shape[2]):
                xarray_list_mp.append([i, j])
                
            
    xarray_list = Parallel(n_jobs=mp.cpu_count(), batch_size=1)(delayed(make_list)(nc_st, point) 
                                 for point in tqdm(xarray_list_mp, 
                                                   desc="Making list..."))
    

    result = Parallel(n_jobs=mp.cpu_count(), batch_size=1)(delayed(sd_stat_groupby)(point) 
                                 for point in tqdm(xarray_list, 
                                                   desc="Calculating..."))

    df_full = pd.concat(result)       
    
    xxx = df_full.to_xarray()
    xxx.to_netcdf(save_path)       
    