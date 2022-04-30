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

import warnings
warnings.filterwarnings('ignore')

def objective_ltgbm(trial):

    param = {#"device":'gpu',
             "metric" : 'RMSE',
             "n_estimators": trial.suggest_int("n_estimators", 100, 3000, 100),
             "num_leaves": trial.suggest_int("num_leaves", 1, 100),
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

#train_df=pd.read_csv('data/raw/dataset_train.csv', sep=',')
#test_df=pd.read_csv('data/raw/dataset_test.csv', sep=',')

df_all = pd.read_csv('dataset1.csv', index_col='Unnamed: 0')
df_all = df_all.reset_index().drop(['index'], axis=1)

df_all = df_all.fillna(0)
df_all.replace([np.inf, -np.inf], 0, inplace=True)

test_df = df_all[df_all['year'].isin([2014, 2015, 2016, 2017])]
train_df = df_all[~df_all['year'].isin([2014, 2015, 2016, 2017])]

train_df.drop(['time'],axis=1, inplace=True)
test_df.drop(['time'],axis=1, inplace=True)

# ### Final data

X=train_df.drop(['wpol', 'station', 'year'],axis=1)
y=train_df['wpol']
Z=test_df.drop(['wpol', 'station', 'year'],axis=1)#.drop(['cell_id','valid_time'],axis=1)  #???

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# optuna lgbm
study = optuna.create_study(direction="minimize")
study.optimize(objective_ltgbm, n_trials=100, timeout=600)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# ### Zoo models


'''
#cv=4.39
xgb_1=XGBRegressor(n_estimators=2500, learning_rate=0.05, max_depth=7, 
                  reg_alpha=0.1, reg_lambda=1.5, subsample=0.8, random_state=0, n_jobs=-1)
#cv=4.33
ctb_1=CatBoostRegressor(depth=8, iterations=2000, learning_rate=0.1, logging_level='Silent', random_seed=155)
#cv=4.22
lgb_1=LGBMRegressor(n_estimators=2500, num_leaves=64, metric='rmse', random_state=0)


# ### Stack

nf=5
cv=KFold(n_splits=nf, shuffle=True, random_state=18)


zoo_names=['xgb_1', 'lgb_1', 'ctb_1']
zoo=[xgb_1, lgb_1, ctb_1]

pd.options.mode.chained_assignment = None  # default='warn'

fold=0
i=0
name=0
res=np.zeros(nf)
meta_f=pd.DataFrame(columns=zoo_names, index=X.index).fillna(value=0)
print('start CV')
for model in zoo:
    i=0  
    for train, test in tqdm(cv.split(X,y)):
        model.fit(X.iloc[train],y.iloc[train])
        meta_f[zoo_names[name]].iloc[test]=model.predict(X.iloc[test])
        res[i]=np.sqrt(mean_squared_error(y.iloc[test], model.predict(X.iloc[test])))
        i+=1
    
    print (zoo_names[name],'||',np.mean(res))
    name+=1

for i in meta_f.columns:
    meta_f[meta_f[i]<0]=0

clf_meta=XGBRegressor(learning_rate=0.01, max_depth=4, n_estimators=500)

cv_result=cross_val_score(clf_meta, meta_f, y, n_jobs=-1, cv=cv, scoring='neg_mean_squared_error')

print(cv_result, '||',np.mean(np.sqrt(-cv_result)))


clf_meta.fit(meta_f, y)


name=0
Z_meta_f=pd.DataFrame(columns=zoo_names, index=Z.index).fillna(value=0)

for model in zoo: 
    model.fit(X,y)
    Z_meta_f[zoo_names[name]]=model.predict(Z)
    name+=1

for i in Z_meta_f.columns:
    Z_meta_f[Z_meta_f[i]<0]=0


res=pd.DataFrame(clf_meta.predict(Z_meta_f))
res.columns=['wpol_pred']

res[res['wpol_pred']<0]=0


res_pivot=pd.concat([test_df[['station', 'year', 'wpol']].reset_index(), res],axis=1)
print(res_pivot.corr()**2)

res_pivot.to_csv('sub.csv', index=True)

dmp_models=[xgb_1, lgb_1, ctb_1, clf_meta]


# stat
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(res_pivot['wpol_pred'], res_pivot['wpol'])
plt.xlim(0,500)
plt.ylim(0, 500)
plt.xlabel('wpol_pred')
plt.ylabel('wpol')

stat = res_pivot[['wpol_pred', 'wpol']]
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

stat = pd.DataFrame()
stat['mse'] = [mean_squared_error(res_pivot['wpol_pred'], res_pivot['wpol'])]
stat['rmse'] = [np.sqrt(mean_squared_error(res_pivot['wpol_pred'], res_pivot['wpol']))]
stat['mae'] = [np.sqrt(mean_absolute_error(res_pivot['wpol_pred'], res_pivot['wpol']))]
print (stat)


import shap
# feature importance
def shap_graphs(model, X, output = ''):
    
    #model = xgb_1.fit(X,y)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # visualize the first prediction's explanation
    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=20)
    plt.tight_layout()
    
    fig = plt.figure()
    shap.plots.beeswarm(shap_values, max_display=20)
    plt.tight_layout()
    
    fig = plt.figure()
    shap.plots.bar(shap_values, max_display=20)
    plt.tight_layout()
    
model_lgb = lgb_1.fit(X,y)
#shap_graphs(model = xgb_1.fit(X,y), X=X)
shap_graphs(model = model_lgb, X=X)
#shap_graphs(model = ctb_1.fit(X,y), X=X)

import seaborn as sns
feature_imp = pd.DataFrame(sorted(zip(model_lgb.feature_importances_,X.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:40])
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
'''

'''
with open('models/'+'models_final.pkl', 'wb') as f:
    for mdl in dmp_models:
         pickle.dump(mdl, f)
'''


# models_readed = []
# with open('models/'+'models_final.pkl', 'rb') as f:
#     while True:
#         try:
#             models_readed.append(pickle.load(f))
#         except EOFError:
#             break

# meteo, Vanya, test (2016, 2017)
# xgb 26.62
# lgb 27.33
# cat 26.74
# composition 26.325 
#R2=0.8678

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

# meteo, default, test (2016, 2017)
# xgb ??
# lgb ??
# cat 26.74
# 27.8 R2=0.8654