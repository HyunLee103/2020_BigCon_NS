#-*-coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='NanumBarunGothic')
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from make_var_func import mk_var
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import os

## load data
def load_data(data_path):
    perform_raw = pd.read_csv(os.path.join(f'{data_path}','2019_performance.csv'))
    perform_raw = perform_raw[perform_raw['취급액'] != 0]
    perform_raw.reset_index(inplace=True)
    perform_raw.rename(columns={'index':'id'},inplace=True)
    
    test = pd.read_csv(os.path.join(f'{data_path}','question.csv'))
    test.reset_index(inplace=True)
    test.rename(columns={'index':'id'},inplace=True)
    test['id'] = test['id'].map(lambda x: x + perform_raw.loc[perform_raw.index[-1],'id'])

    rating = pd.read_csv(os.path.join(f'{data_path}','2019_rating.csv'),encoding='utf-8')

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
    tem = x['취급액'].reset_index()
    clf.fit(tem)
    tem['anomaly'] = clf.predict(tem)
    return x[tem['anomaly']==1]


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


def preprocess(train,test,drop_rate,k):
    perform = del_outlier(train,drop_rate)
    perform.reset_index(inplace=True, drop=True)
    perform = km_clust(perform,k) # kmeans column 생김.

    X_km = perform[['id','방송일시','노출(분)','마더코드','상품코드','상품명','상품군','판매단가']]
    y_km = perform[['kmeans']]
    y = perform[['취급액']]
    y.rename(columns={'취급액':'sales'},inplace=True)
    test_km = test[['id','방송일시','노출(분)','마더코드','상품코드','상품명','상품군','판매단가']]
    data = pd.concat([X_km,test_km]) # 합쳐서 전처리
    data.reset_index(inplace=True, drop=True)

    var = mk_var(data)
    data = var()
    return data, y, y_km


def mk_trainset(data,dummy = ['gender','pay','hour_gr','min_gr','len_gr','show_norm_order_gr','day','hour','min','mcode_freq_gr','show_order']):
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

    joblib.dump(encoder, 'cate_encoder.pkl')

    all_cate = ['day','hour','min','mcode_freq_gr','show_order','gender','pay','hour_gr','min_gr','len_gr','show_order','show_norm_order_gr','cate']
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