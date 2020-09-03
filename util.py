#-*-coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='NanumBarunGothic')
from sklearn.ensemble import IsolationForest
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
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder

## load data
perform_raw = pd.read_csv('data/2019_performance.csv')
rating = pd.read_csv('data/2019_rating.csv',encoding='utf-8')
test = pd.read_csv('data/question.csv')

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
    y.rename(columns={'취급액':'sales'},inplace=True)
    test_km = test[['방송일시','노출(분)','마더코드','상품코드','상품명','상품군','판매단가']]
    data = pd.concat([X_km,test_km]) # 합쳐서 전처리
    data = reset_index(data)

    var = mk_var(data)
    data = var()
    return data, y, y_km


def mk_trainset(data,dummy = ['gender','pay','hour_gr','min_gr','len_gr','show_norm_order']):
    """
    select feature to make train set 
    arg : data, dummy(list that make it one-hot-encoding)
    return : data
    """
    data['sales_per'] = np.log1p(data['판매단가'])
    data.rename(columns={'마더코드':'mcode','상품군':'cate','노출(분)':'length_raw','상품코드':'item_code'},inplace=True)

    encoder = LabelEncoder()
    encoder.fit(data['cate'])
    data['cate'] = encoder.transform(data['cate'])

    all_cate = ['day','hour','min','mcode_freq_gr','show_order','gender','pay','hour_gr','min_gr','len_gr','show_norm_order','cate']
    left_cate = [x for x in all_cate if x not in dummy]

    if dummy:
        data = pd.get_dummies(data,columns=(dummy))

    if left_cate != []:
        for var in left_cate:
            data[var] = data[var].astype('category')

    data['mcode'] = data['mcode'].astype('str').apply(lambda x: x[3:])
    data['mcode'] = data['mcode'].astype(int)
    data['item_code'] = data['item_code'].astype('str').apply(lambda x: x[2:])
    data['item_code'] = data['item_code'].astype(int)
    data = data.drop(['방송일시','상품명','판매단가','length_raw'],axis=1)

    return data


def metric(real, pred):
    tem = np.abs(pred - real)/pred
    return tem.mean() * 100

def feature_impo(model,data):
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,data.columns)), columns=['Value','Feature'])
    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
    plt.savefig('lgbm_importances-01.png')