#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


data_path='data/input/'


#train_df=pd.read_csv('data/raw/dataset_train.csv', sep=',')
#test_df=pd.read_csv('data/raw/dataset_test.csv', sep=',')

df_all = pd.read_csv('dataset.csv', index_col='Unnamed: 0')
df_all = df_all.fillna(0)
train_df = df_all.copy()
test_df = df_all.tail(5000)
'''
org_train=pd.read_csv(data_path+'ground_measures_train_features.csv', index_col='Unnamed: 0')
org_test=pd.read_csv(data_path+'ground_measures_test_features.csv', index_col='Unnamed: 0')
org_new=pd.read_csv(data_path+'ground_measures_features.csv', index_col='Unnamed: 0')
gm_metadata=pd.read_csv(data_path+'ground_measures_metadata.csv', index_col='station_id')

org_train_df=org_train.unstack().reset_index()
org_test_df=org_test.unstack().reset_index()
org_new_df=org_new.unstack().reset_index()
org_train_df.columns=['date', 'id', 'org_value']
org_test_df.columns=['date', 'id', 'org_value']
org_new_df.columns=['date', 'id', 'org_value']


org=pd.concat([org_train_df, org_test_df], axis=0).merge(gm_metadata, how='left', left_on='id', right_on='station_id').reset_index(drop=True)


org['dt_date'] = pd.to_datetime(org['date'], format='%Y-%m-%d')


org['dayofyear'] = org['dt_date'].dt.dayofyear

org['year'] = org['dt_date'].dt.year

org_new_df2=org_new_df.merge(gm_metadata, how='left', left_on='id', right_on='station_id')


org_new_df2['dt_date'] = pd.to_datetime(org_new_df2['date'], format='%Y-%m-%d')
org_new_df2['dayofyear'] = org_new_df2['dt_date'].dt.dayofyear
org_new_df2['year'] = org_new_df2['dt_date'].dt.year


org_new_df2=org_new_df2[['latitude','longitude', 'elevation_m', 'year', 'dayofyear']]


org_new_df3=train_df[['lat','lon','alt','year','dayofyear']]
org_new_df3.columns=['latitude','longitude', 'elevation_m', 'year', 'dayofyear']
org_new_df5=test_df[['lat','lon','alt','year','dayofyear']]
org_new_df5.columns=['latitude','longitude', 'elevation_m', 'year', 'dayofyear']


org_new_df4=pd.concat([org_new_df3,org_new_df2,org_new_df5]).reset_index(drop=True)


X_org=org[['latitude','longitude', 'elevation_m', 'year', 'dayofyear']]
y_org=org['org_value']
y_org.fillna(0, inplace=True)
Z_org=org_new_df4


rf=RandomForestRegressor(n_estimators=250, random_state=0, n_jobs=-1)


get_ipython().run_cell_magic('time', '', 'rf.fit(X_org,y_org)')


rf_int=pd.DataFrame(rf.predict(Z_org))
rf_int.columns=['rf_org_value_v2']


rf_int_res=pd.concat([Z_org, rf_int],axis=1)
rf_int_res.columns=['lat','lon','alt','year','dayofyear','rf_org_value_v2']


rf_int2=rf_int_res.drop(['alt', 'year'],axis=1).groupby(by=['lat', 'lon', 'dayofyear']).mean('rf_org_value_v2').reset_index()
'''

#train_df=train_df.merge(rf_int2, how='left', left_on=['lat', 'lon', 'dayofyear'], right_on=['lat','lon', 'dayofyear'])
#test_df=test_df.merge(rf_int2, how='left', left_on=['lat', 'lon', 'dayofyear'], right_on=['lat','lon', 'dayofyear'])


train_df.drop(['time'],axis=1, inplace=True)
test_df.drop(['time'],axis=1, inplace=True)


# ### Final data

X=train_df.drop(['wpol'],axis=1)
y=train_df['wpol']
Z=test_df.drop(['wpol'],axis=1)#.drop(['cell_id','valid_time'],axis=1)  #???


# ### Zoo models

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

for model in zoo:
    i=0  
    for train, test in cv.split(X,y):
        model.fit(X.loc[train],y.loc[train])
        meta_f[zoo_names[name]].loc[test]=model.predict(X.loc[test])
        res[i]=np.sqrt(mean_squared_error(y.loc[test], model.predict(X.loc[test])))
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

#???
res_pivot=pd.concat([test_df[['station', 'year', 'wpol']], res],axis=1)

res_pivot.to_csv('sub.csv', index=True)

dmp_models=[xgb_1, lgb_1, ctb_1, clf_meta]


with open('models/'+'models_final.pkl', 'wb') as f:
    for mdl in dmp_models:
         pickle.dump(mdl, f)


# In[44]:


# models_readed = []
# with open('models/'+'models_final.pkl', 'rb') as f:
#     while True:
#         try:
#             models_readed.append(pickle.load(f))
#         except EOFError:
#             break

