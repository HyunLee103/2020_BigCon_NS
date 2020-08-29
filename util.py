#-*-coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='NanumBarunGothic')
from matplotlib import font_manager
import datetime
from sklearn.model_selection import cross_val_score
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.ensemble import IsolationForest, ensemble
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import collections
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, plot_importance
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from make_var_func import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier


## detect outlier
"""
x : dataset
p : ratio of deleted outlier(0 ~ 1)
return : dataset that dropped outlier
"""
def del_outlier(x,p):
    clf=IsolationForest(contamination=float(p),
                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=2020, verbose=0)
    tem = x['취급액'].reset_index()
    clf.fit(tem)
    tem['anomaly'] = clf.predict(tem)
    return x[tem['anomaly']==1]

## reset index
def reset_index(x):
    x = x.reset_index()
    del x['index']
    return x

## clustering
"""
x : dataset
k : # of cluster
return : 기존 dataset에 kmeans라는 컬럼 추가한 df
"""
def km_clust(x, k):
    Z = x['취급액'].reset_index()
    km = KMeans(n_clusters=k)
    km.fit(Z)
    Z['kmeans'] = km.labels_
    return pd.concat([x,Z['kmeans']],axis=1)


def preprocess(raw_data,drop_rate,k):
    perform = del_outlier(raw_data,drop_rate)
    perform = reset_index(perform)
    perform = km_clust(perform,k)

    X_km = perform[['방송일시','노출(분)','마더코드','상품코드','상품명','상품군','판매단가']]
    y_km = perform[['kmeans']]
    y = perform[['취급액']]
    test_km = test[['방송일시','노출(분)','마더코드','상품코드','상품명','상품군','판매단가']]
    data = pd.concat([X_km,test_km]) # 합쳐서 전처리
    data = reset_index(data)

    var = mk_var(data)
    data = var()
    return data, y, y_km


def modeling(data,y_km):
    ## mk train set
    data = data.fillna(0)
    X_train = data.iloc[:36250,:]
    X_test = data.iloc[36250:,:]

    ## modeling
    # gbm
    train_features, test_features, train_labels, test_labels = train_test_split(X_train, y_km,random_state=2020)
    gb_model = GradientBoostingClassifier(n_estimators=400,subsample=0.9,learning_rate=0.05,min_samples_split=0.9,criterion='mae',random_state=2020)
    gb_model.fit(train_features,train_labels)
    gb_score = accuracy_score(gb_model.predict(test_features),test_labels)

    # lgbm
    nama_ch = {v:k for k,v in enumerate(X_train.columns)}
    X_train.columns = [nama_ch[x] for x in X_train.columns]
    train_features, test_features, train_labels, test_labels = train_test_split(X_train, y_km,random_state=2020)
    lgb = LGBMClassifier(n_estimators=2000,learning_rate=0.04,subsample=0.7,colsample_bytree=0.8,random_state=2020,objective='multiclass')
    lgb.fit(train_features,train_labels,early_stopping_rounds = 100,eval_set = [(test_features,test_labels)],verbose=True)
    lgbm_score = accuracy_score(lgb.predict(test_features),test_labels)

    # parameter = {'learning_rate':[0.03,0.04,0.05],
    #         'colsample_bytree':[0.7,0.8,0.9,1.0],
    #         'subsample':[0.7,0.8,0.9]}
    # model = RandomizedSearchCV(LGBMClassifier(random_state=2020,n_estimators=2000),parameter,n_iter=50 ,cv=2, n_jobs=3,random_state=2020)
    # model.fit(train_features,train_labels,early_stopping_rounds = 100,eval_set = [(test_features,test_labels)],verbose=True)
    # print(model.best_estimator_)

    # xgbm
    xgb = XGBClassifier(n_estimators=400, random_state=2020,learning_rate=0.04,objective='multi:softmax',subsample=0.9,colsample_bytree=0.9)
    xgb.fit(train_features, train_labels)
    xgb_score = accuracy_score(xgb.predict(test_features),test_labels)

    # Ensemble
    models = list()
    models.append(('gbm',GradientBoostingClassifier(n_estimators=400,subsample=0.9,learning_rate=0.05,min_samples_split=0.9,criterion='mae',random_state=2020)))
    models.append(('lgbm',LGBMClassifier(n_estimators=2000,learning_rate=0.04,subsample=0.7,colsample_bytree=0.8,random_state=2020,objective='multiclass')))
    models.append(('XGB',XGBClassifier(n_estimators=400, random_state=2020,learning_rate=0.04,objective='multi:softmax',subsample=0.9,colsample_bytree=0.9)))
    ensemble = VotingClassifier(estimators = models, voting = 'hard')

    ensemble.fit(train_features,train_labels)
    ensemble_score = accuracy_score(ensemble.predict(test_features),test_labels)

    print('gbm : {}, lgbm : {}, XGB : {}, Voting : {}'.format(gb_score,lgbm_score,xgb_score,ensemble_score))
    return lgb, ensemble


def mk_trainset(data):
    data['sales_per'] = np.log1p(data['판매단가'])
    data.rename(columns={'마더코드':'mcode','상품군':'cate','노출(분)':'length_raw','상품코드':'item_code'},inplace=True)
    data = pd.get_dummies(data,columns=(['gender','pay','cate','day','hour','hour_gr','min','min_gr','len_gr','mcode_freq_gr']))
    data['mcode'] = data['mcode'].astype('str').apply(lambda x: x[3:])
    data['mcode'] = data['mcode'].astype(int)
    data['item_code'] = data['item_code'].astype('str').apply(lambda x: x[2:])
    data['item_code'] = data['item_code'].astype(int)
    data = data.drop(['방송일시','상품명','판매단가'],axis=1)
    return data

def clustering(data,y_km):
    X_train = data.iloc[:36250,:]
    X_test = data.iloc[36250:,:]
    lgb = LGBMClassifier(n_estimators=1272,learning_rate=0.04,subsample=0.7,colsample_bytree=0.8,random_state=2020,objective='multiclass')
    nama_ch = {v:k for k,v in enumerate(X_train.columns)}
    X_train.columns = [nama_ch[x] for x in X_train.columns]
    lgb.fit(X_train,y_km)

    X_train = data.iloc[:36250,:]
    X_test = data.iloc[36250:,:]
    X_test['kmeans'] = lgb.predict(X_test)
    X_train['kmeans'] = y_km

    X_train_c0 = X_train[X_train['kmeans']==0]
    X_train_c1 = X_train[X_train['kmeans']==1]
    X_train_c2 = X_train[X_train['kmeans']==2]

    X_test_c0 = X_test[X_test['kmeans']==0]
    X_test_c1 = X_test[X_test['kmeans']==1]
    X_test_c2 = X_test[X_test['kmeans']==2]

    return X_train_c0, X_train_c1, X_train_c2, X_test_c0, X_test_c1, X_test_c2


# final predict
def metric(real, pred):
    tem = np.abs(pred - real)/pred
    return tem.mean() * 100

train_features, test_features, train_labels, test_labels = train_test_split(X_train, 
                                                    y, 

                                                    test_size=0.2, 

                                                    shuffle=True, 

                                                    random_state=2020)

model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=700,random_state=2020,subsample=0.7,learning_rate=0.05)
model_lgb = lgb.LGBMRegressor(subsample= 0.7, colsample_bytree= 0.7, learning_rate=0.05)

model_xgb.fit(train_features,train_labels)
model_lgb.fit(train_features,train_labels)

pred_xgb = model_xgb.predict(test_features)
pred_lgb = model_lgb.predict(test_features)

print(metric(test_labels,pred_xgb), metric(test_labels,pred_lgb))
perform['취급액']

