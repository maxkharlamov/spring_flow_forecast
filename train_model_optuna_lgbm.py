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

def objective_cat(trial):

    param = {
             "task_type":'GPU',
             "loss_function" : 'RMSE',
             "n_estimators": trial.suggest_int("n_estimators", 100, 2000, 100),
             "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.5),
             "depth": trial.suggest_int("depth", 1, 12),
             'random_seed' : 155,
             }

    nf=5
    cv=KFold(n_splits=nf, shuffle=True, random_state=18)
    cv_score = [] 
    for train_index, test_index in cv.split(X, y):
        print(test_index)
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        
        model = CatBoostRegressor(**param)
        model.fit(X_train, y_train, verbose=0, early_stopping_rounds=100)
        pred = pd.DataFrame(model.predict(X_test), columns=['111'])
        score = r2_score(y_test, pred)
        score = np.sqrt(mean_squared_error(y_test, pred))
        
        cv_score.append(score)
        
    return np.mean(cv_score)

def objective_xgb(trial):

                  #reg_alpha=0.1, reg_lambda=1.5, subsample=0.8, random_state=0
    param = {
             "task_type":'GPU',
             "loss_function" : 'RMSE',      #???
             'reg_alpha' : 0.1, 
             'reg_lambda' : 1.5, 
             'subsample' : 0.8,
             "n_estimators": trial.suggest_int("n_estimators", 100, 3000, 100),
             "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.5),
             "max_depth": trial.suggest_int("depth", 1, 12),
             'random_state' : 0,
             }

    nf=5
    cv=KFold(n_splits=nf, shuffle=True, random_state=18)
    cv_score = [] 
    for train_index, test_index in cv.split(X, y):
        print(test_index)
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        
        model = XGBRegressor(**param)
        model.fit(X_train, y_train, verbose=0, early_stopping_rounds=100)
        pred = pd.DataFrame(model.predict(X_test), columns=['111'])
        score = r2_score(y_test, pred)
        score = np.sqrt(mean_squared_error(y_test, pred))
        
        cv_score.append(score)
        
    return np.mean(cv_score)

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
# # fit and predict
# =============================================================================
lgb_1=LGBMRegressor(**params)
lgb_1.fit(X,y)

res=pd.DataFrame(lgb_1.predict(Z))
res.columns=['wpol_pred']

res[res['wpol_pred']<0]=0


res_pivot=pd.concat([test_df[['station', 'year', 'wpol']].reset_index(), res],axis=1)
print(res_pivot.corr()**2)

# =============================================================================
# # stat
# =============================================================================
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(res_pivot['wpol_pred'], res_pivot['wpol'])
plt.xlim(0,500)
plt.ylim(0, 500)
plt.xlabel('wpol_pred')
plt.ylabel('wpol')

stat = res_pivot[['wpol_pred', 'wpol']]
from sklearn.metrics import mean_squared_error, mean_absolute_error

stat = pd.DataFrame()
stat['mse'] = [mean_squared_error(res_pivot['wpol_pred'], res_pivot['wpol'])]
stat['rmse'] = [np.sqrt(mean_squared_error(res_pivot['wpol_pred'], res_pivot['wpol']))]
stat['mae'] = [np.sqrt(mean_absolute_error(res_pivot['wpol_pred'], res_pivot['wpol']))]
print (stat)

# feature importance

def shap_graphs(model, X, output_prefix = ''):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # visualize the first prediction's explanation
    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=30)
    plt.tight_layout()
    plt.savefig('waterfall_' + output_prefix + '.png')
    
    fig = plt.figure()
    shap.plots.beeswarm(shap_values, max_display=30)
    plt.tight_layout()
    plt.savefig('beeswarm_' + output_prefix + '.png')
    
    fig = plt.figure()
    shap.plots.bar(shap_values, max_display=30)
    plt.tight_layout()
    plt.savefig('bar_' + output_prefix + '.png')
    
shap_graphs(model = lgb_1, X=X, output_prefix = 'lgbm_meteo_shapes_geol_clim')

#???
'''
import seaborn as sns
feature_imp = pd.DataFrame(sorted(zip(lgb_1.feature_importances_,X.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:40])
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
'''

# meteo, test (2014, 2015, 2016, 2017)
# xgb 26.7
# lgb 27.47
# cat 26.9
# composition  26.36
#R2=0.87

# meteo, test (2014, 2015, 2016, 2017), without station and year
# xgb 31.928
# lgb 32.22
# cat 31.79
# composition  31.61
#R2=0.84

# meteo, test (2014, 2015, 2016, 2017), without station and year
# lgb 33.13
#R2=0.831
#mse       rmse      mae
#1098.169685  33.138643  4.87322

# meteo+shape, test (2014, 2015, 2016, 2017), without station and year
# lgb 25.626
#R2=0.8968
#    mse       rmse       mae
#717.972122  26.795002  4.402685

# meteo+shape+geol, test (2014, 2015, 2016, 2017), without station and year
# lgb  25.46
#R2=0.8906
#    mse       rmse       mae
#738.892194  27.182572  4.460894

# meteo+shape+geol+clim, test (2014, 2015, 2016, 2017), without station and year
# lgb  25.465
#R2=0.9
#    mse       rmse       mae
#694.272717  26.349055  4.425478













