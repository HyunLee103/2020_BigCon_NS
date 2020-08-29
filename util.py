-*-coding: utf-8 -*-

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

## data load 
perform_raw = pd.read_csv('data/2019_performance.csv')
rating = pd.read_csv('data/2019_rating.csv',encoding='utf-8')
test = pd.read_csv('data/question.csv')

test = test[test['상품군']!='무형']

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


def main_preprocess(raw_data,drop_rate,k,):
    perform = del_outlier(raw_data,drop_rate)
    perform = reset_index(perform)
    perform = km_clust(perform,k)

    X_km = perform[['방송일시','노출(분)','마더코드','상품코드','상품명','상품군','판매단가']]
    y_km = perform[['kmeans']]
    y = perform[['취급액']]
    test_km = test[['방송일시','노출(분)','마더코드','상품코드','상품명','상품군','판매단가']]
    data = pd.concat([X_km,test_km]) # 합쳐서 전처리
    return data
    # var = mk_var(data)
    # data = var()
    # return data, y, y_km

def get_voting():
	# define the base models
    models = list()
    models.append(('logit',LogisticRegression(random_state=2020,multi_class='multinomial',max_iter=200)))
    models.append(('lgbm',lgb))
    models.append(('RF',rf))
    ensemble = VotingClassifier(estimators = models, voting = 'hard')

	return ensemble


def main_modeling(data,y,y_km):
    ## mk train set
    X_train = data.iloc[:36250,:]
    X_test = data.iloc[36250:,:]

    ## modeling
    # logistic regression
    train_features, test_features, train_labels, test_labels = train_test_split(X_train, y_km,random_state=2020)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # poly = PolynomialFeatures(degree = 2, interaction_only=False, include_bias=False)
    # X_poly = poly.fit_transform(train_features)
    # X_poly.shape
    logit = LogisticRegression(random_state=2020,multi_class='multinomial',max_iter=200)
    logit.fit(train_features,train_labels)
    logit_score = accuracy_score(logit.predict(test_features),test_labels)

    # lgbm
    train_features, test_features, train_labels, test_labels = train_test_split(X_train, y_km,random_state=2020)
    lgb = LGBMClassifier(n_estimators=2000,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=2020,objective='multiclass')
    lgb.fit(train_features,train_labels,early_stopping_rounds = 100,eval_set = [(test_features,test_labels)],verbose=True)
    lgbm_score = accuracy_score(lgb.predict(test_features),test_labels)

    # rf
    train_features, test_features, train_labels, test_labels = train_test_split(X_train, y_km,random_state=2020)
    rf = RandomForestClassifier(n_estimators=500, random_state=2020)
    rf.fit(train_features, train_labels)
    rf_score = accuracy_score(rf.predict(test_features),test_labels)

    print('logit : {}, lgbm : {}, RF : {}'.format(logit_score,lgbm_score,rf_score))

    # Ensemble
    models = list()
    models.append(('logit',logit))
    models.append(('lgbm',lgb))
    models.append(('RF',rf))

    ensemble = VotingClassifier(estimators = models, voting = 'hard')










## Feature Engineering
def g_filter(x):
    if '여성' in x:
        return 1
    elif '남성' in x:
        return 2
    else:
        return 0

def pay_filter(x):
    if '일시' in x:
        return 1
    elif '무이자' in x:
        return 2
    else:
        return 0

train_over0 = data[data['노출(분)']>0]
name_dic = train_over0['마더코드'].value_counts()

def s(x):
    try:
        return name_dic[x]
    except:
        return 0

data['gender'] = data['상품명'].apply(g_filter)
data['pay_cat'] = data['상품명'].apply(pay_filter)
data['item_cnt'] = data['마더코드'].apply(s)


data['판매단가_log'] = np.log1p(data['판매단가'])

data = pd.get_dummies(data,columns=(['방송요일','방송시각','방송시각_그룹','노출(분)_그룹','gender','pay_cat','상품군']))

del data['방송일시']
del data['상품명']
del data['판매단가']
# del data['마더코드']
# del data['상품코드']
# del data['노출(분)']
del data['방송ID']
# del data['상품코드']

data['마더코드'] = data['마더코드'].astype('str').apply(lambda x: x[3:])
data['마더코드'] = data['마더코드'].astype(int)
data['상품코드'] = data['상품코드'].astype('str').apply(lambda x: x[2:])
data['상품코드'] = data['상품코드'].astype(int)
# data = data.rename(columns={'판매단가_log':'sale_per_log','상품군_가구':'cate_0','상품군_가전':'cate_1','상품군_건강기능':'cate_2','상품군_농수축':'cate_3','상품군_무형':'cate_4','상품군_생활용품':'cate_5','상품군_속옷':'cate_6','상품군_의류':'cate_7','상품군_이미용':'cate_8','상품군_잡화':'cate_9','상품군_주방':'cate_10','상품군_침구':'cate_11'})
X_train = data.iloc[:36250,:]
X_test = data.iloc[36250:,:]



## modeling
train_features, test_features, train_labels, test_labels = train_test_split(X_train, y_km,random_state=2020)
# logistic regression(poly)
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# poly = PolynomialFeatures(degree = 2, interaction_only=False, include_bias=False)
# X_poly = poly.fit_transform(train_features)
# X_poly.shape

model = LogisticRegression(random_state=2020,multi_class='multinomial',max_iter=200)
model.fit(train_features,train_labels)
accuracy_score(model.predict(test_features),test_labels)


# lgbm
# feature 이름 바꾸기
nama_ch = {v:k for k,v in enumerate(X_train.columns)}
X_train.columns = [nama_ch[x] for x in X_train.columns]

train_features, test_features, train_labels, test_labels = train_test_split(X_train, y_km,random_state=2020)
lgb = LGBMClassifier(n_estimators=2000,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=2020,objective='multiclass')
lgb.fit(train_features,train_labels,early_stopping_rounds = 100,eval_set = [(test_features,test_labels)],verbose=True)
accuracy_score(lgb.predict(test_features),test_labels)

feature_imp = pd.DataFrame(sorted(zip(lgb.feature_importances_,data.columns)), columns=['Value','Feature'])
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()

# rf
train_features, test_features, train_labels, test_labels = train_test_split(X_train, y_km,random_state=2020)
rf = RandomForestClassifier(n_estimators=500, random_state=2020)
rf.fit(train_features, train_labels)
accuracy_score(rf.predict(test_features),test_labels)

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
