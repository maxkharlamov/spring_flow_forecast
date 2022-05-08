#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import geopandas as gpd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

import tqdm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import seaborn as sns
import mapclassify as mc

from mpl_toolkits.axes_grid1 import make_axes_locatable

def nse(predictions, targets):
    return 1-(np.sum((predictions-targets)**2)/np.sum((targets-np.mean(targets))**2))

def sample_stat(sample):
    
    sample = sample.sort_values(['year'])
    cor = sample.corr()**2
    
    stat = pd.DataFrame()
    stat['R2'] = [cor.loc['wpol', 'wpol_pred']]
    stat['RMSE'] = np.sqrt(mean_squared_error(sample['wpol'], sample['wpol_pred']))
    stat['mae'] = mean_absolute_error(sample['wpol'], sample['wpol_pred'])
    stat['NS'] = nse(sample['wpol_pred'].values, sample['wpol'].values)
    
    aaa = abs(sample['wpol_pred'] - sample['wpol'])/sample['wpol'].mean() * 100
    stat['delta, %'] = aaa.mean()
    
    return stat

def prepare(df, shape, rename = True):
    dff = df.copy()
    if rename:
        dff.columns = [int(x[:-4]) for x in list(dff)]
    dff = dff.T
    
    return shape.join(dff, how = 'right')

def importance_plot(shapes_new, feature, metric, folder, 
                    bins = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]):
    
    cmap = 'rainbow'
    
    fig = plt.figure(figsize = (20, 15))
    
    ax = plt.axes(projection = ccrs.AlbersEqualArea(30,35))
    
    shapes_new.plot(ax = ax, column = feature, cmap = cmap, scheme='equal_interval',
                    k=10,
                    markersize=100, edgecolor='black', zorder=2,
                    legend=True, transform=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    
    plt.title('prediction accuracy ' + metric, fontdict = {'fontsize': 20})
    
    directory = folder + '/'
    if not os.path.exists(directory): os.makedirs(directory)
    
    fig.savefig(directory + 'prediction accuracy ' + metric + '.png')
    #plt.close('all')

def graphs_per_watershed(data, stat, cols, path_to_save):
    stations = data['station'].unique()
    for st in tqdm.tqdm(stations):
        try:
            plt.close('all')
            sample = data[data['station'] == st]
            statka = stat[stat['station'] == st]
            
            sample = sample.sort_values(['year'])
            sample.index = sample['year']
            if len(cols) == 1:
                sample[cols[0]].plot()
            else:
                sample[cols].plot()
            plt.title('station ' + str(int(st)) + ' (R2='+str(round(statka['R2'].values[0],2))+' RMSE='+str(round(statka['RMSE'].values[0],2))+' delta%='+str(round(statka['delta, %'].values[0],2))+')')
            plt.grid()
            plt.legend()
            
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            plt.savefig(path_to_save + str(int(st)) + '.png')
        except:
            print('error in ', str(int(st)))

loo_file = 'loo_year1.csv'
loo_file = 'loo_station1.csv'

data = pd.read_csv(loo_file)

data = data.drop(['Unnamed: 0', 'index'], axis=1)
data['RMSE'] = abs(data['wpol_pred'] - data['wpol'])
data['abs_delta_proc'] = data['RMSE']/data['wpol'] * 100

shapes = gpd.read_file('data/shapes/hpost_rf/hp_2623.shp')
shapes.index = shapes['STATION']

stat = data.groupby('station').apply(sample_stat).reset_index().drop(['level_1'], axis=1)

shape1 = stat.merge(shapes[['STATION', 'X', 'Y']].reset_index(drop=True), 
                    left_on= 'station', right_on= 'STATION', how='left')

shape_stat = gpd.GeoDataFrame(shape1, 
                              geometry=gpd.points_from_xy(shape1['X'], shape1['Y']))

shape_stat = shape_stat[shape_stat['R2']<0.99]

if loo_file == 'loo_station1.csv':
    
    importance_plot(shape_stat, 'R2', 'R2', 'loo_graphs/station_loo/graphs_importance_loo_station', 
                        bins = [0, 0.2, 0.4, 0.6, 0.8, 1])
    
    importance_plot(shape_stat, 'NS', 'NS', 'loo_graphs/station_loo/graphs_importance_loo_station', 
                        bins = [0, 0.2, 0.4, 0.6, 0.8, 1])
    
    importance_plot(shape_stat, 'delta, %', 'delta_proc', 'loo_graphs/station_loo/graphs_importance_loo_station', 
                        bins = [-0.2, 0, 0.2, 0.4, 0.6])
    
    importance_plot(shape_stat, 'RMSE', 'RMSE', 'loo_graphs/station_loo/graphs_importance_loo_station', 
                        bins = [-0.2, 0, 0.2, 0.4, 0.6])
    
    importance_plot(shape_stat, 'mae', 'mae', 'loo_graphs/station_loo/graphs_importance_loo_station', 
                        bins = [-0.2, 0, 0.2, 0.4, 0.6])
    
    graphs_per_watershed(data, stat, ['wpol', 'wpol_pred'], 'loo_graphs/station_loo/graphs_predict_station_loo/')
    graphs_per_watershed(data, stat, ['RMSE'], 'loo_graphs/station_loo/graphs_predict_station_loo_rmse/')
    graphs_per_watershed(data, stat, ['abs_delta_proc'], 'loo_graphs/station_loo/graphs_predict_station_loo_abs_delta/')
    
    stat.to_excel('stat_loo_watershed.xlsx')

if loo_file == 'loo_year1.csv':
    
    importance_plot(shape_stat, 'R2', 'R2', 'loo_graphs/year_loo/graphs_importance_loo_year', 
                        bins = [0, 0.2, 0.4, 0.6, 0.8, 1])
    
    importance_plot(shape_stat, 'NS', 'NS', 'loo_graphs/year_loo/graphs_importance_loo_year', 
                        bins = [0, 0.2, 0.4, 0.6, 0.8, 1])
    
    importance_plot(shape_stat, 'delta, %', 'delta_proc', 'loo_graphs/year_loo/graphs_importance_loo_year', 
                        bins = [-0.2, 0, 0.2, 0.4, 0.6])
    
    importance_plot(shape_stat, 'RMSE', 'RMSE', 'loo_graphs/year_loo/graphs_importance_loo_year', 
                        bins = [-0.2, 0, 0.2, 0.4, 0.6])
    
    importance_plot(shape_stat, 'mae', 'mae', 'loo_graphs/year_loo/graphs_importance_loo_year', 
                        bins = [-0.2, 0, 0.2, 0.4, 0.6])
    
    graphs_per_watershed(data, stat, ['wpol', 'wpol_pred'], 'loo_graphs/year_loo/graphs_predict_year_loo/')
    graphs_per_watershed(data, stat, ['RMSE'], 'loo_graphs/year_loo/graphs_predict_year_loo_rmse/')
    graphs_per_watershed(data, stat, ['abs_delta_proc'], 'loo_graphs/year_loo/graphs_predict_year_loo_abs_delta/')
    
    stat.to_excel('stat_loo_year.xlsx')  


