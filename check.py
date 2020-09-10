import numpy as np
import pandas as pd
from util import  load_data, preprocess, mk_trainset, metric
from clustering import clustering
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from dim_reduction import train_AE,by_AE,by_PCA

def boosting(X,y,col_sample=0.6,lr=0.04,iter=1500):
    train_features, test_features, train_labels, test_labels = train_test_split(X, y,random_state=2020)
    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    model_lgb.fit(train_features,train_labels,early_stopping_rounds = 200,eval_set = [(test_features,test_labels)],verbose=False)
    pred_lgb = model_lgb.predict(test_features)
    return metric(test_labels,pred_lgb), len(test_labels)

def boosting_return_value(X,y,col_sample=0.6,lr=0.04,iter=1500):
    train_features, test_features, train_labels, test_labels = train_test_split(X, y,random_state=2020) # id 살아 있음
    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    model_lgb.fit(train_features.drop(['id'],axis=1),train_labels,early_stopping_rounds = 200, eval_set = [(test_features.drop(['id'],axis=1),test_labels)],verbose=False)
    pred_lgb = model_lgb.predict(test_features.drop(['id'],axis=1))
    
    return pd.concat([pd.DataFrame(test_labels).reset_index(), pd.DataFrame(pred_lgb)], axis=1), metric(test_labels,pred_lgb), len(test_labels)

def predict_return_value(data,y): # 그냥 우선 4개로 고정,,
    for i in range(4):
        globals()[f'X_train_c{i}'] = X_train[X_train['kmeans']==i]
        globals()[f'X_test_c{i}'] = X_test[X_test['kmeans']==i]

    origin, originlen = boosting(data.drop(['id'],axis=1).iloc[:34317,:],y['sales'])
    r0, c0, len0 = boosting_return_value(X_train_c0.drop(['sales','kmeans'],axis=1),X_train_c0['sales'])
    r1, c1, len1 = boosting_return_value(X_train_c1.drop(['sales','kmeans'],axis=1),X_train_c1['sales'])
    r2, c2, len2 =  boosting_return_value(X_train_c2.drop(['sales','kmeans'],axis=1),X_train_c2['sales'])
    r3, c3, len3 =  boosting_return_value(X_train_c3.drop(['sales','kmeans'],axis=1),X_train_c3['sales'])

    total_error = (c0 * len0 + c1 * len1 + c2 * len2 + c3 *len3)/(len0+len1+len2+len3)

    print(f'origin error : {round(origin,2)}%\n\nCluster_0 : {round(c0,2)}%\nCluster_1 : {round(c1,2)}%\nCluster_2 : {round(c2,2)}%\nCluster_3 : {round(c3,2)}%\n\nTotal error : {round(total_error,2)}%')
    
    r0['cluster'] = 0
    r1['cluster'] = 1
    r2['cluster'] = 2
    r3['cluster'] = 3

    return pd.concat([r0,r1,r2,r3]).reset_index(drop=True)

data_path = 'data/'
perform_raw, rating, test = load_data(data_path) # perform_raw, test에 id 부여 (index > id column으로)
raw_data, y, y_km = preprocess(perform_raw,test,0.03,4) # 이상치 삭제

data = mk_trainset(raw_data)
X_train, X_test = clustering(data,y_km,y)
results = predict_return_value(data,y)

# data, X_train > ohe 되어 있음

perform_raw.merge(results,left_on='id',right_on='index',how='right').tail(10)
# perform_raw에는 붙는데 raw_data에는 안붙음,,

results = raw_data.merge(results,left_on='id',right_on='index',how='right')
results.drop(['index','sales'],axis=1,inplace=True)

#results.to_csv('results.csv',encoding='euc-kr')



