from lightgbm.callback import early_stopping
from pandas.io.pytables import Term
from sklearn import dummy
from util import load_data,make_sid,preprocess,preprocess_sales_to_mean,mk_statistics_var,mk_trainset, metric, preprocess_sales_to_mean, scoring
from clustering import clustering
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from dim_reduction import train_AE,by_AE,by_PCA
import numpy as np
import pandas as pd


def boosting(X,y,X_val,y_val,col_sample=0.6,lr=0.04,iter=1500,six=True):

    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    model_lgb.fit(X,y,early_stopping_rounds = 500,eval_set = [(X_val,y_val)],verbose=False)
    pred_lgb = model_lgb.predict(X_val)
    
    res = pd.concat([y_val.reset_index(drop=True),pd.DataFrame(pred_lgb,columns=['pred'])],axis=1)
    res['score'] = res.apply(lambda x : scoring(x['sales'],x['pred']),axis=1)
    return res['score'].mean(), len(res)

def predict(X_train,val,k,col_sample=0.6,lr=0.04,iter=1500,six=True):
    """
    predict '취급액' score only using train set(perform)
    return : RMAE score for each cluster
    """

    origin, originlen = boosting(X_train.drop(['id','sales','mean_sales','kmeans'],axis=1),X_train['sales'],val.drop(['sales','mean_sales','kmeans','id'],axis=1),val['sales'],col_sample,lr,iter,six)
    print(f'origin error : {round(origin,2)}%\n')

    sum = 0
    total_len = 0
    for i in range(k):
        train_tem = X_train[X_train['kmeans']==i]
        val_tem = val[val['kmeans']==i]
        score,len = boosting(train_tem.drop(['mean_sales','sales','kmeans','id'],axis=1),train_tem['sales'],val_tem.drop(['mean_sales','sales','kmeans','id'],axis=1),val_tem['sales'],col_sample,lr,iter,six)
        sum += (score * len)
        total_len += len
        print(f'Cluster_{i} : {round(score,2)}%\n')
    print(f'Total error : {round(sum/total_len,2)}%')

# excution
if __name__=='__main__': 
    data_path = 'data/'
    perform_raw, rating, test_raw = load_data(data_path)
    perform_raw, test_raw = make_sid(perform_raw, test_raw)
    #train, test, y_km, train_len = preprocess(perform_raw,test_raw,0.03,3,inner=False) # train, test 받아서 쓰면 돼
    train, test, y_km, train_len = preprocess_sales_to_mean(perform_raw,test_raw,0.03,3,inner=False)
    raw_data = mk_statistics_var(train,test)
    data = mk_trainset(raw_data)
    # data - sales, mean_sales 모두 갖고 있는 상태
    train, val = clustering(data,y_km,train_len)
    predict(train,val,3)












"""

train['kmeans'].value_counts()
sns.boxplot(val[val['kmeans']==0]['sales'])

sns.boxplot(train[train['kmeans']==0]['sales'])
sns.boxplot(train[train['kmeans']==1]['sales'])
sns.boxplot(train[train['kmeans']==2]['sales'])
sns.boxplot(val[val['kmeans']==0]['sales'])
sns.boxplot(val[val['kmeans']==1]['sales'])
sns.boxplot(val[val['kmeans']==2]['sales'])

out = IsolationForest(contamination = 0.1,max_features=1.0, bootstrap=False, n_jobs=-1, random_state=2020, verbose=0)
val_0 = val[val['kmeans']==0]
val_0.reset_index(drop=True,inplace=True)
tem = val_0['sales'].reset_index()
out.fit(tem)
tem['anomaly'] = out.predict(tem)
sns.boxplot(val_0[tem['anomaly']==1]['sales'])
val_0 = val_0[tem['anomaly']==1]

val = pd.concat([val[val['kmeans']==1],val[val['kmeans']==2],val_0])




catfeature = ['cate','day','hour','hour_gr','min','min_gr','len_gr','mcode_freq_gr','gender','pay','show_order','show_norm_order_gr']
cat = CatBoostClassifier(loss_function='MultiClass')
train_dataset = Pool(data=train_features,label=train_labels,cat_features=catfeature)
val_dataset = Pool(data=val_features,label=val_labels,cat_features=catfeature)
cat.fit(train_features,train_labels,eval_set=val_dataset,early_stopping_rounds=200,cat_features=catfeature)


accuracy_score(lgb.predict(val_features), val_labels)
accuracy_score(cat.predict(val_features), val_labels)



<cluster 정확도 cv>

kfold = KFold(n_splits=5,shuffle=True,random_state=2020)
res = cross_val_score(lgb,X_train,y_km,cv=kfold)
print(f'Accuracy : {res}\nAverage acc : {np.array(res).mean()}')


<시각화>

plt.rcParams["figure.figsize"] = (10,6)

sns.boxplot(perform_raw['취급액'])

sns.boxplot(y)

Z = y.rename(columns={'sales':'취급액'}).reset_index()
km = KMeans(n_clusters=3)
km.fit(Z)
df = pd.DataFrame(Z)
df['kmeans'] = km.labels_
colormap = { 0: 'red', 1: 'green', 2: 'blue'}
colors = df.apply(lambda row: colormap[row.kmeans], axis=1)
ax = df.plot(kind='scatter', x=0, y=1, alpha=0.1, s=300, c=colors)

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(Z)
    kmeanModel.fit(Z)
    distortions.append(sum(np.min(cdist(Z, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / Z.shape[0])
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

sns.boxplot(X_train_c0['sales'])
plt.title('Cluster 0')
plt.xlabel('취급액')
sns.boxplot(X_train_c1['sales'])
plt.title('Cluster 1')
plt.xlabel('취급액')
sns.boxplot(X_train_c2['sales'])
plt.title('Cluster 2')
plt.xlabel('취급액')
"""
