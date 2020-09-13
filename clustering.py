import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV


def clustering(data,y_km,y):
    """
    make cluster for train and val set using models from return of modeling func.
    return : clustered dataframe
    """  
    X_train = data.iloc[:36250,:]
    X_test = data.iloc[36250:,:]
    X_train = pd.concat([X_train,y_km,y],axis=1)
    six = X_train[X_train['month']==6]
    others = X_train[X_train['month']!=6]
    train_sample_six = six.sample(1716,random_state=2020)
    train_sample_other = others.sample(1000,random_state=2020)
    
    X_val = pd.concat([train_sample_six,train_sample_other])
    X_train = X_train[X_train['id'].isin(set(X_train['id']) - set(X_val['id']))]

    train_features = X_train.drop(['id','kmeans','sales'],axis=1)
    train_labels = X_train['kmeans']
    val_features = X_val.drop(['id','kmeans','sales'],axis=1)
    val_labels = X_val['kmeans']
    lgb = LGBMClassifier(n_estimators=2000,learning_rate=0.045,subsample=0.8,colsample_bytree=0.6,random_state=2020,objective='multiclass')
    lgb.fit(train_features,train_labels,early_stopping_rounds=300,eval_set=[(val_features,val_labels)],verbose=True)
    

    X_val['kmeans_pred'] = lgb.predict(val_features)

    return X_train, X_val


def eval_cluster(data,y_km):
    """
    Evaluate clustering accuracy
    return : models
    """
    ## mk train set
    X_train = data.iloc[:36250,:]
    X_test = data.iloc[36250:,:]

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