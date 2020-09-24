from lightgbm.callback import early_stopping, reset_parameter
from pandas.io.pytables import Term
from sklearn import dummy
from util import load_data,mk_sid,preprocess,mk_statistics_var,mk_trainset, metric
from clustering import clustering
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from dim_reduction import train_AE,by_AE,by_PCA
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier



def boosting(X,y,X_val,y_val,robustScaler,col_sample=0.6,lr=0.04,iter=50000,inference=True):

    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    model_lgb.fit(X,y,early_stopping_rounds = 500,eval_set = [(X_val,y_val)],verbose=False)
    if inference:
        pred_lgb = model_lgb.predict(X_val)
        res = pd.concat([y_val.reset_index(drop=True),pd.DataFrame(pred_lgb,columns=['pred'])],axis=1)
    else:
        pred_lgb = model_lgb.predict(X)
        res = pd.concat([y.reset_index(drop=True),pd.DataFrame(pred_lgb,columns=['pred'])],axis=1)

    # print(res)
    real = robustScaler.inverse_transform(np.array(res['sales']).reshape(-1,1))
    pred = robustScaler.inverse_transform(np.array(res['pred']).reshape(-1,1))
    print(real.shape,pred.shape)

    return metric(real,pred), len(res), pd.DataFrame({'real':real.flatten(), 'pred':pred.flatten()},columns=['real','pred'])

def boosting_2(pop,inference,robustScaler,col_sample=0.6,lr=0.04,iter=50000,test=True):
    y = pop['sales']
    X = pop.drop(['id','sales','kmeans'],axis=1)
    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    
    if test==True:
        model_lgb.fit(X,y)
        pred_lgb = model_lgb.predict(inference.drop(['id','sales','kmeans'],axis=1))
        res = pd.concat([inference.reset_index(drop=True),pd.DataFrame(pred_lgb,columns=['pred'])],axis=1)
        real = robustScaler.inverse_transform(np.array(res['sales']).reshape(-1,1))
        pred = robustScaler.inverse_transform(np.array(res['pred']).reshape(-1,1))

        return metric(real,pred), pd.DataFrame({'real':real.flatten(), 'pred':pred.flatten()},columns=['real','pred']) 
     
    else:
        X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=2020)
        model_lgb.fit(X_train,y_train,early_stopping_rounds = 1000,eval_set = [(X_val,y_val)],verbose=False)
        pred_lgb = model_lgb.predict(X_val)
        res = pd.concat([y_val.reset_index(drop=True),pd.DataFrame(pred_lgb,columns=['pred'])],axis=1)
        real = robustScaler.inverse_transform(np.array(res['sales']).reshape(-1,1))
        pred = robustScaler.inverse_transform(np.array(res['pred']).reshape(-1,1))

        return metric(real,pred), pd.DataFrame({'real':real.flatten(), 'pred':pred.flatten()},columns=['real','pred'])



def predict(X_train,val,k,robustScaler,col_sample=0.6,lr=0.04,iter=50000,inference=True):
    """
    predict '취급액' score only using train set(perform)
    return : RMAE score for each cluster
    """
    origin, originlen, tmp = boosting(X_train.drop(['id','sales','kmeans'],axis=1),X_train['sales'],val.drop(['sales','kmeans','id'],axis=1),val['sales'],robustScaler,col_sample,lr,iter,inference)
    print(f'origin error : {round(origin,2)}%\n')

    
    sum = 0
    total_len = 0
    for i in range(k):
        train_tem = X_train[X_train['kmeans']==i]
        val_tem = val[val['kmeans']==i]

        score,len,pred = boosting(train_tem.drop(['sales','kmeans','id'],axis=1),train_tem['sales'],val_tem.drop(['sales','kmeans','id'],axis=1),val_tem['sales'],robustScaler,col_sample,lr,iter,inference)
        if inference==True:
            results = pd.concat([val_tem.reset_index(drop=True),pred],axis=1)
        else:
            results = pd.concat([train_tem.reset_index(drop=True),pred],axis=1)

        results['MAPE'] = pred.apply(lambda x: metric(x['real'],x['pred']),axis=1)

        if i == 0:
            fin_results = results.copy()

        else:
            fin_results = pd.concat([fin_results,results])
            
        sum += (score * len)
        total_len += len
        print(f'Cluster_{i} : {round(score,2)}%\n')
    print(f'Total error : {round(sum/total_len,2)}%')

    return fin_results


def second_fit(input,robustScaler,train,val):
    neg = input.iloc[:5000,:]
    pos = input.iloc[5000:,:].sample(n=5000,random_state=2020)
    population = pd.concat([neg,pos]).reset_index(drop=True)

    result = predict(train,val,3,robustScaler,inference=True,iter=10000)
    result_0 = result[result['kmeans']==0]

    score,res = boosting_2(population.iloc[:,:-3],result_0.iloc[:,:-3],robustScaler,0.55,0.04,1500,True)
    result_0['pred_eva'] = (res['pred']*0.4 + result_0['pred']*0.6)
    # print(f'second_fit cluster0  : {score}')
    fin_score = (metric(result_0['pred_eva'], result_0['real'])*1412 + result[result['kmeans']!=0]['MAPE'].sum())/2746
    print(f'final error  : {round(fin_score,2)}%')


# excution
if __name__=='__main__': 
    data_path = 'data/'
    perform_raw, rating, test_raw = load_data(data_path,trend=False,weather=False)
    # perform_raw, test_raw = mk_sid(perform_raw,test_raw)
    train, test, y_km, train_len= preprocess(perform_raw,test_raw,0.03,3,inner=False) # train, test 받아서 쓰면 돼
    raw_data = mk_statistics_var(train,test)
    data = mk_trainset(raw_data)
    train, val, robustScaler = clustering(data,y_km,train_len)
    
    tem_result = predict(train,val,3,robustScaler,inference=False,iter=1500)

    second_fit(tem_result,robustScaler,train,val)





"""
sns.boxplot(train[train['kmeans']==0]['sales'])
sns.boxplot(train[train['kmeans']==1]['sales'])
sns.boxplot(train[train['kmeans']==2]['sales'])
sns.boxplot(val[val['kmeans']==0]['sales'])
sns.boxplot(val[val['kmeans']==1]['sales'])
sns.boxplot(val[val['kmeans']==2]['sales'])

"""







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