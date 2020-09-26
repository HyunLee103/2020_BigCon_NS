#-*-coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.reshape.reshape import get_dummies
import seaborn as sns
plt.rc('font', family='NanumBarunGothic')
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.decomposition import PCA
import pandas as pd



## load data
def load_data(data_path,weather=False,trend=False,query=False):
    perform_raw = pd.read_csv(os.path.join(f'{data_path}','2019_performance.csv'))
    test = pd.read_csv(os.path.join(f'{data_path}','question.csv'))
    rating = pd.read_csv(os.path.join(f'{data_path}','2019_rating.csv'),encoding='utf-8')
    
    if weather:
        weather_train = pd.read_csv(os.path.join(f'{data_path}','weather_data.csv'))
        weather_test = pd.read_csv(os.path.join(f'{data_path}','weather_data_test.csv'))
        perform_raw = pd.concat([perform_raw,weather_train],axis=1)
        test = pd.concat([test,weather_test],axis=1)
    if trend:
        trend_train = pd.read_csv(os.path.join(f'{data_path}','shopping_trend.csv'))
        trend_test = pd.read_csv(os.path.join(f'{data_path}','shopping_trend_test.csv'))
        perform_raw = pd.concat([perform_raw,trend_train],axis=1)
        test = pd.concat([test,trend_test],axis=1)
    

    perform_raw.reset_index(inplace=True)
    perform_raw.rename(columns={'index':'id'},inplace=True)
    perform_raw['is_mcode'] = 1

    test.reset_index(inplace=True)
    test.rename(columns={'index':'id'},inplace=True)
    test['id'] = test['id'] + 37372

    train_mcode = set(perform_raw['마더코드'])
    test_mcode = set(test['마더코드'])
    df = pd.DataFrame(columns=['is_mcode'])
    test = pd.concat([test,df])
    test['is_mcode'][test['마더코드'].isin(list(train_mcode & test_mcode))] = 1
    test.fillna(0,inplace=True)

    return perform_raw, rating, test

## detect outlier
"""
x : dataset
p : ratio of deleted outlier(0 ~ 1)
return : dataset that dropped outlier
"""
def del_outlier(x,p):
    clf=IsolationForest(contamination=float(p),
                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=2020, verbose=0)
    tem = x['sales'].reset_index()
    clf.fit(tem)
    tem['anomaly'] = clf.predict(tem)
    return x[tem['anomaly']==1]


## clustering
"""
x : dataset
k : # of cluster
return : 기존 dataset에 kmeans라는 컬럼 추가한 df
"""
def km_clust(x, k, var=['sales'],inner=3):
    # mean = x[var].mean()
    # std = x[var].std()
    # x[var] = (x[var] - x[var].mean())/x[var].std()
    Z = x[var].reset_index()
    km = KMeans(n_clusters=k,random_state=2020)
    km.fit(Z)
    Z['kmeans'] = km.labels_

    # inner cluster
    if inner!=0:
        object = Z[Z['kmeans']==1]
        K = Z[Z['kmeans']==1]['sales'].reset_index()
        left = Z[Z['kmeans']!=1]
        km = KMeans(n_clusters=inner,random_state=2020)
        km.fit(K)
        object['labels'] = km.labels_
        object['kmeans'][object['labels']==0] = 1
        object['kmeans'][object['labels']==1] = 3
        object['kmeans'][object['labels']==2] = 4    
        object.drop(['labels'],axis=1,inplace=True) 
        object.set_index('index')
        left.set_index('index')
        res = pd.concat([object.set_index('index'),left.set_index('index')]).sort_index()
        
        return pd.concat([x,res['kmeans']],axis=1)
    else:
        return pd.concat([x,Z['kmeans']],axis=1)


def preprocess(perform,question,drop_rate,k,inner=False):

    perform.reset_index(inplace=True,drop=True)
    question.reset_index(inplace=True,drop=True)

    train = perform[perform['sales']!=0]
    train.reset_index(inplace=True,drop=True)

    train = del_outlier(train,drop_rate)
    train.reset_index(inplace=True,drop=True)

    train = km_clust(train,k,inner=inner)
    print(train['kmeans'].value_counts())

    y_km = train[['kmeans']]
    train = train.drop(['kmeans'],axis=1)

    raw_data = pd.concat([train,question]).reset_index(drop=True)

    return raw_data, y_km, len(train)



def mk_trainset(data,dummy = ['gender','pay'],categorical=True):
    """
    select feature to make train set 
    arg : data, dummy(list that make it one-hot-encoding),catetgorical(True for lgbm, False for other models)
    return : data
    """

    encoder = LabelEncoder()
    encoder.fit(data['cate'])
    data['cate'] = encoder.transform(data['cate'])
    all_cate = ['day','hour','min','mcode_freq_gr','s_order','gender','pay','cate']
    left_cate = [x for x in all_cate if x not in dummy]

    if categorical:
        pass
    else:
        dummy = all_cate
        left_cate = []

    if dummy:
        data = pd.get_dummies(data,columns=(dummy))

    if left_cate != []:
        for var in left_cate:
            data[var] = data[var].astype('category')

    data = data.drop(['방송일시','상품명','판매단가'],axis=1)

    return data


def metric(real, pred):
    tem = np.abs((real - pred)/real)
    return tem.mean() * 100


def feature_impo(model,data):
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,data.columns)), columns=['Value','Feature'])
    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
    plt.savefig('lgbm_importances-01.png')






