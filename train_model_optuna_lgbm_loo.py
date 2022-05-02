#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import LeaveOneOut

from tqdm import tqdm

import optuna
import shap

import warnings
warnings.filterwarnings('ignore')

def objective_ltgbm(trial):

    param = {#"device":'gpu',
             "metric" : 'RMSE',
             "n_estimators": trial.suggest_int("n_estimators", 100, 4000, 100),
             "num_leaves": trial.suggest_int("num_leaves", 1, 300),
             'random_state' : 0,
             }

    nf=5
    cv=KFold(n_splits=nf, shuffle=True, random_state=18)
    cv_score = [] 
    for train_index, test_index in cv.split(X, y):
        print(test_index)
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        
        model = LGBMRegressor(**param)
        model.fit(X_train, y_train)
        pred = pd.DataFrame(model.predict(X_test), columns=['111'])
        score = r2_score(y_test, pred)
        score = np.sqrt(mean_squared_error(y_test, pred))
        
        cv_score.append(score)
        
    return np.mean(cv_score)

def loo_stat(df_all, model, label):
    years = df_all[label].unique()
    cv = LeaveOneOut()
    
    loo_data = []
    for train_ix, test_ix in tqdm(cv.split(years)):
        #print(train_ix, test_ix)    
        train_df = df_all[df_all[label].isin(years[train_ix])]
        test_df = df_all[df_all[label].isin(years[test_ix])]
        
        train_df.drop(['time'],axis=1, inplace=True)
        test_df.drop(['time'],axis=1, inplace=True)
        
         
        X=train_df.drop(['wpol', 'station', 'year'],axis=1)
        y=train_df['wpol']
        Z=test_df.drop(['wpol', 'station', 'year'],axis=1)
        
        model.fit(X,y)
        
        res=pd.DataFrame(model.predict(Z))
        res.columns=['wpol_pred']
    
        res[res['wpol_pred']<0]=0
    
        res_pivot=pd.concat([test_df[['station', 'year', 'wpol']].reset_index(), res],axis=1)
        
        loo_data.append(res_pivot)
    
    loo_data = pd.concat(loo_data)
    
    return loo_data

data_path='data/input/'

df_all = pd.read_csv('dataset_w_shape_features_geol_clim.csv', index_col='Unnamed: 0')
df_all = df_all.reset_index().drop(['index'], axis=1)

df_all = df_all.fillna(0)
df_all.replace([np.inf, -np.inf], 0, inplace=True)

test_df = df_all[df_all['year'].isin([2014, 2015, 2016, 2017])]
train_df = df_all[~df_all['year'].isin([2014, 2015, 2016, 2017])]

train_df.drop(['time'],axis=1, inplace=True)
test_df.drop(['time'],axis=1, inplace=True)

# =============================================================================
# # ### Final data
# =============================================================================
X=train_df.drop(['wpol', 'station', 'year'],axis=1)
y=train_df['wpol']
Z=test_df.drop(['wpol', 'station', 'year'],axis=1)

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# =============================================================================
# # optuna lgbm
# =============================================================================

study = optuna.create_study(direction="minimize")
study.optimize(objective_ltgbm, n_trials=100, timeout=600)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
params = {"metric" : 'RMSE', 'random_state' : 0,}
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    params[key] = value

# =============================================================================
# LOO test
# =============================================================================

#lgb_1=LGBMRegressor(**params)          # optuna params
lgb_1=LGBMRegressor(n_estimators=2600, num_leaves=62, metric='rmse', random_state=0)

# тест на устойчивость по годам
loo_year = loo_stat(df_all, lgb_1, 'year')
loo_year.to_csv('loo_year1.csv')

# тест на выбрасываемые водосборы
loo_year = loo_stat(df_all, lgb_1, 'station')
loo_year.to_csv('loo_station1.csv')
