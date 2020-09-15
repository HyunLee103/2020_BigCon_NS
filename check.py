import numpy as np
import pandas as pd
from util import  load_data, preprocess, mk_trainset, metric
from clustering import clustering
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from dim_reduction import train_AE,by_AE,by_PCA
from sklearn.externals import joblib
import seaborn as sns

def boosting(X,y,col_sample=0.6,lr=0.04,iter=1500):
    train_features, test_features, train_labels, test_labels = train_test_split(X, y,random_state=2020)
    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    model_lgb.fit(train_features,train_labels,early_stopping_rounds = 200,eval_set = [(test_features,test_labels)],verbose=False)
    pred_lgb = model_lgb.predict(test_features)
    return metric(test_labels,pred_lgb), len(test_labels)


def boosting_return_value(X,y,col_sample=0.6,lr=0.04,iter=1500):
    train_features, test_features, train_labels, test_labels = train_test_split(X, y,random_state=2020) # id 살아 있음
    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    model_lgb.fit(train_features.drop(['id','kmeans'],axis=1),train_labels,early_stopping_rounds = 200, eval_set = [(test_features.drop(['id','kmeans'],axis=1),test_labels)],verbose=False)
    pred_lgb = model_lgb.predict(test_features.drop(['id','kmeans'],axis=1))
    tr_pred_lgb = model_lgb.predict(train_features.drop(['id','kmeans'],axis=1))


    features= pd.DataFrame(test_features[['id','kmeans']]).reset_index(drop=True)
    sales = pd.DataFrame(test_labels).reset_index(drop=True)
    pred = pd.DataFrame(pred_lgb).reset_index(drop=True)

    return pd.concat([features,sales,pred],axis=1),metric(test_labels,pred_lgb), len(test_labels), metric(train_labels,tr_pred_lgb), len(train_labels)

def predict_return_value(data,y): # 그냥 우선 4개로 고정,,
    for i in range(4):
        globals()[f'X_train_c{i}'] = X_train[X_train['kmeans']==i]
        globals()[f'X_test_c{i}'] = X_test[X_test['kmeans']==i]

    origin, originlen = boosting(data.drop(['id'],axis=1).iloc[:34317,:],y['sales'])
    r0, c0, len0, tr0, len_tr0 = boosting_return_value(X_train_c0.drop(['sales'],axis=1),X_train_c0['sales'])
    r1, c1, len1, tr1, len_tr1  = boosting_return_value(X_train_c1.drop(['sales'],axis=1),X_train_c1['sales'])
    r2, c2, len2, tr2, len_tr2  =  boosting_return_value(X_train_c2.drop(['sales'],axis=1),X_train_c2['sales'])
    r3, c3, len3, tr3, len_tr3  =  boosting_return_value(X_train_c3.drop(['sales'],axis=1),X_train_c3['sales'])

    total_error = (c0 * len0 + c1 * len1 + c2 * len2 + c3 *len3)/(len0+len1+len2+len3)
    tr_total_error = (tr0 * len_tr0 + tr1 * len_tr1 + tr2 * len_tr2 + tr3 * len_tr3) / (len0+len1+len2+len3)

    print(f'origin error : {round(origin,2)}%\n\nCluster_0 : {round(c0,2)}%\nCluster_1 : {round(c1,2)}%\nCluster_2 : {round(c2,2)}%\nCluster_3 : {round(c3,2)}%\n\nTotal error : {round(total_error,2)}%')
    print(f'\ntrain\n\n Cluster_0: {round(tr0,2)}% \n Cluster_1: {round(tr1,2)}% \n Cluster_2: {round(tr2,2)}% \n Cluster_3: {round(tr3,2)}% \n\nTotal error : {round(total_error,2)}%')

    return pd.concat([r0,r1,r2,r3]).reset_index(drop=True)

data_path = 'data/'
perform_raw, rating, test = load_data(data_path) 
# perform_raw, test에 연속된 id 부여. 취급액 0인 row는 삭제 됨.
raw_data, y, y_km = preprocess(perform_raw,test,0.03,4) # 이상치 삭제, raw_data에 train,test concat 되어있음.

data = mk_trainset(raw_data)
X_train, X_test = clustering(data,y_km,y) # data, X_train > ohe 되어 있음

results = predict_return_value(data,y)

# 이상한 거. valid test셋 > cluster 

''''
results = predict_return_value(data,y)
results = raw_data.merge(results,left_on='id',right_on='id',how='right') # 8581개

results.rename(columns={0:'predict'},inplace=True)
results['MAPE'] = results.apply(lambda x: abs(x['sales']-x['predict']) * 100 /x['sales'] ,axis=1)
results.to_csv('predict_results.csv',encoding='euc-kr',index=False)
'''

## 코랩 코딩 : / ##
results = pd.read_csv('predict_results.csv',encoding='euc-kr')

results.groupby('kmeans')['MAPE'].describe()
# cluster) 0 - 10.94, 1 - 48.48, 2 - 15.75, 3 - 9.93 
# ㄴ 취급액) 3 > 0 > 2 > 1

wrong = results.nlargest(round(len(results)*0.1),'MAPE',keep='all')
wrong.groupby('kmeans')['MAPE'].describe()

c1 = results[results['kmeans']==1]

encoder = joblib.load('cate_encoder.pkl')

cate = c1.groupby('cate')['MAPE'].describe()[['count','mean','std']]
cate['cate_ko'] = encoder.inverse_transform(cate.index)
cate.sort_values(by='mean',ascending=False)

# cluster 별로 lgbm input 변수를 다르게 주면? > 변수는 어떻게 select하지?
# 지금 중복변수 많음 (그룹핑 전 후) > 이런건 영향 안주나?

 


'''
wrong = results.nlargest(round(len(results)*0.1),'MAPE',keep='all') # 858개
good = results.nsmallest(round(len(results)*0.1),'MAPE',keep='all')

wrong.sort_values(by='MAPE', ascending=False).head(30)
good.sort_values(by='MAPE', ascending=True).head(30)

# cate
encoder = joblib.load('cate_encoder.pkl')

cate_m = wrong.groupby('cate')['MAPE'].describe()[['count','mean','std']]
cate_m['cate_ko'] = encoder.inverse_transform(cate_m.index)
cate_m.sort_values(by='mean',ascending=False)

tmp = good.groupby('cate')['MAPE'].describe()[['count','mean','std']]
tmp['cate_ko'] = encoder.inverse_transform(tmp.index)
tmp.sort_values(by='mean',ascending=True)

tmp2 = results.groupby('cate')['MAPE'].describe()[['count','mean','std']].sort_values(by='mean',ascending=False)
tmp2['cate_ko'] = encoder.inverse_transform(tmp2.index)
tmp2

# 못 맞히는 순서) 생활용품 건강기능 이미용 농수축 속옷 잡화 의류 주방 가구 가전 침구
# 잘 맞히는 순서) 침구 생활용품 가전 가구 의류 주방 농수축 속옷 잡화 건강기능 이미용

# 못해! ) 건강기능, 이미용, 

# 판매단가
sales_m = wrong.groupby('sales_per')['MAPE'].describe()['mean'].reset_index()
sns.jointplot(x='sales_per', y='mean',data=sales_m,kind='scatter')
# 비교적 저렴한 것들 多. 많이 팔려야했던 것들이 많이 안팔려서

# month
month_m = wrong.groupby('month')['MAPE'].describe()[['count','mean','std']]
month_m.sort_values(by='mean',ascending=False)

# holiday > 차이 없음
holiday_m = wrong.groupby('holiday')['MAPE'].describe()[['count','mean','std']]
holiday_m.sort_values(by='mean',ascending=False)

# hour
hour_m = wrong.groupby('hour')['MAPE'].describe()[['count','mean','std']]
hour_m.sort_values(by='mean',ascending=False)

hour_gr_m = wrong.groupby('hour_gr')['MAPE'].describe()[['count','mean','std']]
hour_gr_m.sort_values(by='mean',ascending=False)
'''