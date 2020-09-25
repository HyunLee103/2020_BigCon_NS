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

    lgb = LGBMClassifier(n_estimators=2000,learning_rate=0.04,subsample=0.8,colsample_bytree=0.5,random_state=2020,objective='multiclass')
    
    if test:
        lgb.fit(X_train.drop(['id','sales'],axis=1),y_km)
        X_test['kmeans'] = lgb.predict(X_test.drop(['id','sales'],axis=1))
        X_test.reset_index(drop=True,inplace=True)
        train = pd.concat([X_train,y_km],axis=1)
        train.reset_index(drop=True,inplace=True)
        return train, X_test, s
    else:      
        train_features, val_features, train_labels, val_labels = train_test_split(X_train,y_km,random_state=2020,test_size=0.08)

        # val set에 is_mcode 변수 달기
        train_features.reset_index(drop=True,inplace=True)
        train_features['val_is_mcode'] = 1
        val_features.reset_index(drop=True,inplace=True)
        train_mcode = set(train_features['mcode'])
        val_mcode = set(val_features['mcode'])
        tem = pd.DataFrame(columns=['val_is_mcode'])
        val_features = pd.concat([val_features,tem])
        val_features['val_is_mcode'][val_features['mcode'].isin(list(train_mcode & val_mcode))] = 1
        val_features.fillna(0,inplace=True)

        # val cluster 예측
        lgb.fit(train_features.drop(['id','sales'],axis=1),train_labels,early_stopping_rounds=300,eval_set=[(val_features.drop(['id','sales'],axis=1),val_labels)],verbose=False)
        val_features['kmeans'] = lgb.predict(val_features.drop(['id','sales'],axis=1))
        val_features.reset_index(drop=True,inplace=True)
        train = pd.concat([train_features,train_labels],axis=1)
        train.reset_index(drop=True,inplace=True)

        return train, val_features, s

