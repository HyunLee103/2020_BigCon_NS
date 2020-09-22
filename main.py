from lightgbm.callback import early_stopping
from pandas.io.pytables import Term
from sklearn import dummy
from util import load_data,mk_sid,preprocess,mk_statistics_var,mk_sid_df,mk_trainset, metric, scoring
from clustering import clustering
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from dim_reduction import train_AE,by_AE,by_PCA
import pandas as pd


def boosting(X,y,X_val,y_val,col_sample=0.6,lr=0.04,iter=1500,six=True):

    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    model_lgb.fit(X,y,early_stopping_rounds = 500,eval_set = [(X_val,y_val)],verbose=False)
    pred_lgb = model_lgb.predict(X_val)

    res = pd.DataFrame(pred_lgb,columns=['pred'])
    
    #res = pd.concat([y_val.reset_index(drop=True),pd.DataFrame(pred_lgb,columns=['pred'])],axis=1)
    #res['score'] = res.apply(lambda x : scoring(x['sales'],x['pred']),axis=1)
    
    return res

def predict(X_train,val,k,col_sample=0.6,lr=0.04,iter=1500,six=True):
    """
    predict '취급액' score only using train set(perform)
    return : RMAE score for each cluster
    """
    orgin = boosting(X_train.drop(['show_id','item_code','sales_by_sid','kmeans'],axis=1),X_train['sales_by_sid'],val.drop(['show_id','item_code','sales_by_sid','kmeans'],axis=1),val['sales_by_sid'],col_sample,lr,iter,six)
    orgin = pd.concat([val[['show_id','item_code','kmeans','sales_by_sid']].reset_index(drop=True),orgin],axis=1)

    train_tem = X_train[X_train['kmeans']==0]
    val_tem = val[val['kmeans']==0]
    pred_0 = boosting(train_tem.drop(['show_id','item_code','sales_by_sid','kmeans'],axis=1),train_tem['sales_by_sid'],val_tem.drop(['show_id','item_code','sales_by_sid','kmeans'],axis=1),val_tem['sales_by_sid'],col_sample,lr,iter,six)
    pred_0 = pd.concat([val_tem[['show_id','item_code','kmeans','sales_by_sid']].reset_index(drop=True),pred_0],axis=1)

    train_tem = X_train[X_train['kmeans']==1]
    val_tem = val[val['kmeans']==1]
    pred_1 = boosting(train_tem.drop(['show_id','item_code','sales_by_sid','kmeans'],axis=1),train_tem['sales_by_sid'],val_tem.drop(['show_id','item_code','sales_by_sid','kmeans'],axis=1),val_tem['sales_by_sid'],col_sample,lr,iter,six)
    pred_1 = pd.concat([val_tem[['show_id','item_code','kmeans','sales_by_sid']].reset_index(drop=True),pred_1],axis=1)

    train_tem = X_train[X_train['kmeans']==2]
    val_tem = val[val['kmeans']==2]
    pred_2 = boosting(train_tem.drop(['show_id','item_code','sales_by_sid','kmeans'],axis=1),train_tem['sales_by_sid'],val_tem.drop(['show_id','item_code','sales_by_sid','kmeans'],axis=1),val_tem['sales_by_sid'],col_sample,lr,iter,six)
    pred_2 = pd.concat([val_tem[['show_id','item_code','kmeans','sales_by_sid']].reset_index(drop=True),pred_2],axis=1)

    return orgin,pd.concat([pred_0,pred_1,pred_2]).reset_index(drop=True)



# excution
if __name__=='__main__': 
    data_path = 'data/'
    perform_raw, rating, test_raw = load_data(data_path)
    perform_raw, test_raw = mk_sid(perform_raw,test_raw)
    train, test, km_by_sid, train_len = preprocess(perform_raw,test_raw,0.03,3,inner=False) # train, test 받아서 쓰면 돼
    # train - 35379 (= train_len) / km_by_sid - 12702 / test - 2716
    raw_data = mk_statistics_var(train,test) # raw_data - 38095
    sales, data = mk_sid_df(raw_data.copy(),train_len) # data - 13618
    # sales - 방송ID,상품코드,정답값(전체 row 대상), 학습때 필요한 방송별 sales 값은 df에 붙어있음.
    data = mk_trainset(data)

    train, val = clustering(data,km_by_sid)

    orgin_pred,k_pred = predict(train,val,3)
    
    metric(k_pred['sales_by_sid'],k_pred['pred'])

    # 얍

    sales
    k = sales.groupby(['show_id','상품코드'])['sales'].mean().reset_index()
    sales = sales.dropna()
    k = pd.merge(sales,k,on=['show_id','상품코드'],how='left')
    metric(k['sales_x'],k['sales_y'])

    
    # 함수화 해야
    orgin_ans = pd.merge(sales,orgin_pred,left_on=['show_id','상품코드'],right_on=['show_id','item_code'],how='left')
    orgin_ans = orgin_ans.dropna()

    k_ans = pd.merge(sales,k_pred,left_on=['show_id','상품코드'],right_on=['show_id','item_code'],how='left')
    k_ans = k_ans.dropna()

    # 방법1. 평균
    count = orgin_ans.groupby(['show_id','상품코드'])['sales'].count().reset_index().rename({'sales':'count'},axis=1)
    orgin_ans = pd.merge(orgin_ans,count,on=['show_id','상품코드'],how='left')
    k_ans = pd.merge(k_ans,count,on=['show_id','상품코드'],how='left')

    orgin_fin = orgin_ans[['show_id','상품코드','sales','pred','count']].copy()
    k_fin = k_ans[['show_id','상품코드','sales','pred','count']].copy()

    orgin_fin['ans'] = orgin_ans.apply(lambda x: x['pred']/x['count'],axis=1)
    k_fin['ans'] = k_ans.apply(lambda x: x['pred']/x['count'],axis=1)

    orgin_fin['score'] = orgin_fin.apply(lambda x: scoring(x['sales'],x['ans']), axis=1)
    orgin_fin['score'].mean() # 80.90

    k_fin['score'] = k_fin.apply(lambda x: scoring(x['sales'],x['ans']), axis=1)
    k_fin['score'].mean() # 74.52

    # 방법2. 방송순서 별 편차
    orgin_fin = orgin_ans[['show_id','상품코드','sales','pred','show_norm_order']].copy()
    k_fin = k_ans[['show_id','상품코드','sales','pred','show_norm_order']].copy()

    order_sum = orgin_ans.groupby(['show_id','상품코드'])['show_norm_order'].sum().reset_index()
    order_sum.rename({'show_norm_order':'order_sum'},axis=1,inplace=True)

    orgin_fin = pd.merge(orgin_fin,order_sum,on=['show_id','상품코드'],how='left')
    k_fin = pd.merge(k_fin,order_sum,on=['show_id','상품코드'],how='left')

    k_fin.head()

    orgin_fin['ans'] = orgin_fin.apply(lambda x: x['pred'] * x['show_norm_order']/x['order_sum'], axis=1)
    k_fin['ans'] = k_fin.apply(lambda x: x['pred'] * x['show_norm_order']/x['order_sum'],axis=1)

    orgin_fin['score'] = orgin_fin.apply(lambda x: scoring(x['sales'],x['ans']), axis=1)
    orgin_fin['score'].mean()

    k_fin['score'] = k_fin.apply(lambda x: scoring(x['sales'],x['ans']), axis=1)
    k_fin['score'].mean()

    orgin_ans.groupby(['show_id','상품코드'])


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