import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
robustScaler = RobustScaler()
import numpy as np



def clustering(data,y_km,train_len,test=False):
    """
    make cluster for train and val set using models from return of modeling func.
    return : clustered dataframe
    """  
    X_train = data.iloc[:train_len,:]
    X_test = data.iloc[train_len:,:]

    robustScaler.fit(np.array(X_train['sales']).reshape(-1,1))
    s = robustScaler.fit(np.array(X_train['sales']).reshape(-1,1))
    X_train['sales'] = robustScaler.transform(np.array(X_train['sales']).reshape(-1,1))

    lgb = LGBMClassifier(n_estimators=1500,learning_rate=0.04,subsample=0.8,colsample_bytree=0.5,random_state=2020,objective='multiclass')
    
    if test:
        lgb.fit(X_train.drop(['id','sales']+['is_mcode','mcode_freq','mcode_freq_gr','mcode_sales_mean','mcode_sales_std','mcode_sales_med','mcode_sales_rank','mcode_order_mean','mcode_order_med','mcode_order_rank','mcode_order_std'],axis=1),y_km)
        X_test['kmeans'] = lgb.predict(X_test.drop(['id','sales']+['is_mcode','mcode_freq','mcode_freq_gr','mcode_sales_mean','mcode_sales_std','mcode_sales_med','mcode_sales_rank','mcode_order_mean','mcode_order_med','mcode_order_rank','mcode_order_std'],axis=1))
        X_test.reset_index(drop=True,inplace=True)
        train = pd.concat([X_train,y_km],axis=1)
        train.reset_index(drop=True,inplace=True)
        return train, X_test, s
    else:      
        train_features, val_features, train_labels, val_labels = train_test_split(X_train,y_km,random_state=2020,test_size=0.08)

        # val set에 is_mcode 변수 달기
        val_1 = val_features.sample(n=1500,random_state=2020)
        val_0 = val_features.drop(val_1.index)
        val_0['is_mcode'] = 0
        val_features = pd.concat([val_0,val_1])


        # val cluster 예측
        lgb.fit(train_features.drop(['id','sales','is_mcode'],axis=1),train_labels,verbose=False)
        val_features['kmeans'] = lgb.predict(val_features.drop(['id','sales','is_mcode'],axis=1))
        val_features.reset_index(drop=True,inplace=True)
        train = pd.concat([train_features,train_labels],axis=1)
        train.reset_index(drop=True,inplace=True)

        return train, val_features, s


