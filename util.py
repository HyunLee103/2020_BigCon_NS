#-*-coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.reshape.reshape import get_dummies
import seaborn as sns
plt.rc('font', family='NanumBarunGothic')
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from make_var_func import mk_var, mk_stat_var
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd



## load data
def load_data(data_path,weather=False,trend=False):
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

    test.reset_index(inplace=True)
    test.rename(columns={'index':'id'},inplace=True)
    test['id'] = test['id'] + 37372

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
def km_clust(x, k, var=['취급액'],inner=3):

    Z = x[var].reset_index()
    km = KMeans(n_clusters=k,random_state=2020)
    km.fit(Z)
    Z['kmeans'] = km.labels_

    # inner cluster
    if inner!=0:
        object = Z[Z['kmeans']==1]
        K = Z[Z['kmeans']==1]['취급액'].reset_index()
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
    data = pd.concat([perform,question])
    data.reset_index(inplace=True,drop=True)
    var = mk_var(data)
    data = var()
    train = data.iloc[:37372,:]
    test = data.iloc[37372:,:]
    test.reset_index(inplace=True,drop=True)
    train.reset_index(inplace=True,drop=True)

    train = train[train['취급액']!=0]
    train.reset_index(inplace=True,drop=True)

    train = del_outlier(train,drop_rate)
    train.reset_index(inplace=True,drop=True)

    train = km_clust(train,k,inner=inner)
    train.rename(columns={'취급액':'sales'},inplace=True)
    test.rename(columns={'취급액':'sales'},inplace=True)

    y_km = train[['kmeans']]
    train = train.drop(['kmeans'],axis=1)

    return train, test, y_km, len(train)

def mk_statistics_var(train,test):
    stat_var = mk_stat_var(train,test)
    data = stat_var()

    #data.fillna(0, inplace=True) # test set 무형 data 변수
    return data

def mk_trainset(data,dummy = ['gender','pay','min_gr','len_gr','show_norm_order_gr']):
    """
    select feature to make train set 
    arg : data, dummy(list that make it one-hot-encoding)
    return : data
    """
    data['sales_per'] = np.log1p(data['판매단가'])
    data.rename(columns={'마더코드':'mcode','상품군':'cate','노출(분)':'length_raw','상품코드':'item_code'},inplace=True)
    # data['sales'] = np.log1p(data['sales'])
    encoder = LabelEncoder()
    encoder.fit(data['cate'])
    data['cate'] = encoder.transform(data['cate'])

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
    tem = np.abs((real - pred)/real)
    return tem.mean() * 100

def s_metric(real, pred):
    tem = np.abs(pred - real)/((np.abs(pred) + np.abs(real))/2)
    return tem.mean() * 100


def feature_impo(model,data):
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,data.columns)), columns=['Value','Feature'])
    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
    plt.savefig('lgbm_importances-01.png')


def scoring(sales,pred):
    if sales==0:
        score = s_metric(sales,pred)
        return score
    else:
        score = metric(sales,pred)
        return score


"""
def kmeans(data,k,var=['sales'],visual=False):

    train_km = pd.concat([data.iloc[:36250,:],y],axis=1)
    Z = train_km[var].reset_index()

    km = KMeans(n_clusters=k,random_state=2020)
    km.fit(Z)
    df = pd.DataFrame(Z)
    df['kmeans'] = km.labels_
    df = inner_cluster(df,3,1)
    y_km = df['kmeans']

    if visual:
        pca3 = PCA(n_components=3)
        data_pca3 = pca3.fit_transform(scaled_Z)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_pca3[:,0], data_pca3[:,1], data_pca3[:,2], c=df['kmeans'], s=60, edgecolors='white')
        ax.set_title('3D of Target distribution by diagnosis')
        ax.set_xlabel('pc1')
        ax.set_ylabel('pc2')
        ax.set_zlabel('pc3')
        plt.show()

    return y_km
"""