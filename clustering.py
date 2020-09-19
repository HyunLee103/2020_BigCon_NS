import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV


def clustering(data,y_km,train_len):
    """
    make cluster for train and val set using models from return of modeling func.
    return : clustered dataframe
    """  
    X_train = data.iloc[:train_len,:]
    X_test = data.iloc[train_len:,:]
    train_features, val_features, train_labels, val_labels = train_test_split(X_train,y_km,random_state=2020,test_size=0.08)

    lgb = LGBMClassifier(n_estimators=2000,learning_rate=0.04,subsample=0.8,colsample_bytree=0.5,random_state=2020,objective='multiclass')
    lgb.fit(train_features.drop(['id','sales','mean_sales'],axis=1),train_labels,early_stopping_rounds=300,eval_set=[(val_features.drop(['id','sales','mean_sales'],axis=1),val_labels)],verbose=True)

    val_features['kmeans'] = lgb.predict(val_features.drop(['id','sales','mean_sales'],axis=1))

    val_features.reset_index(drop=True,inplace=True)
    train = pd.concat([train_features,train_labels],axis=1)
    train.reset_index(drop=True,inplace=True)

    return train, val_features



"""
def eval_cluster(data,y_km):
    ## mk train set
    #X_train = data.iloc[:34317,:]
    #X_test = data.iloc[34317:,:]

    X_train = data.iloc[:35379,:]
    X_test = data.iloc[35379:,:]

    ## modeling
    # gbm
    train_features, test_features, train_labels, test_labels = train_test_split(X_train, y_km,random_state=2020)
    gb_model = GradientBoostingClassifier(n_estimators=400,subsample=0.9,learning_rate=0.05,min_samples_split=0.9,criterion='mae',random_state=2020)
    gb_model.fit(train_features,train_labels)
    gb_score = accuracy_score(gb_model.predict(test_features),test_labels)

    # lgbm
    train_features, test_features, train_labels, test_labels = train_test_split(X_train, y_km,random_state=2020)
    lgb = LGBMClassifier(n_estimators=2000,learning_rate=0.04,subsample=0.8,colsample_bytree=0.55,random_state=2020,objective='multiclass')
    lgb.fit(train_features,train_labels,early_stopping_rounds = 100,eval_set = [(test_features,test_labels)],verbose=True)
    lgbm_score = accuracy_score(lgb.predict(test_features),test_labels)
    lgbm_score  
    # parameter = {'learning_rate':[0.03,0.04,0.05],
    #         'colsample_bytree':[0.7,0.8,0.9,1.0],
    #         'subsample':[0.7,0.8,0.9]}
    # model = RandomizedSearchCV(LGBMClassifier(random_state=2020,n_estimators=2000),parameter,n_iter=50 ,cv=2, n_jobs=3,random_state=2020)
    # model.fit(train_features,train_labels,early_stopping_rounds = 100,eval_set = [(test_features,test_labels)],verbose=True)
    # print(model.best_estimator_)

    # xgbm
    xgb = XGBClassifier(n_estimators=400, random_state=2020,learning_rate=0.04,objective='multi:softmax',subsample=0.9,colsample_bytree=0.9)
    xgb.fit(train_features, train_labels)
    xgb_score = accuracy_score(xgb.predict(test_features),test_labels)

    # Ensemble
    models = list()
    models.append(('gbm',GradientBoostingClassifier(n_estimators=400,subsample=0.9,learning_rate=0.05,min_samples_split=0.9,criterion='mae',random_state=2020)))
    models.append(('lgbm',LGBMClassifier(n_estimators=2000,learning_rate=0.04,subsample=0.7,colsample_bytree=0.8,random_state=2020,objective='multiclass')))
    models.append(('XGB',XGBClassifier(n_estimators=400, random_state=2020,learning_rate=0.04,objective='multi:softmax',subsample=0.9,colsample_bytree=0.9)))
    ensemble = VotingClassifier(estimators = models, voting = 'hard')

    ensemble.fit(train_features,train_labels)
    ensemble_score = accuracy_score(ensemble.predict(test_features),test_labels)

    print('gbm : {}, lgbm : {}, XGB : {}, Voting : {}'.format(gb_score,lgbm_score,xgb_score,ensemble_score))
    return lgb, ensemble

"""