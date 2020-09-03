from util import *


def modeling(data,y_km):
    """
    Evaluate clustering accuracy
    return : models
    """
    ## mk train set
    data = data.fillna(0)
    X_train = data.iloc[:36250,:]
    X_test = data.iloc[36250:,:]

    ## modeling
    # gbm
    train_features, test_features, train_labels, test_labels = train_test_split(X_train, y_km,random_state=2020)
    gb_model = GradientBoostingClassifier(n_estimators=400,subsample=0.9,learning_rate=0.05,min_samples_split=0.9,criterion='mae',random_state=2020)
    gb_model.fit(train_features,train_labels)
    gb_score = accuracy_score(gb_model.predict(test_features),test_labels)

    # lgbm
    nama_ch = {v:k for k,v in enumerate(X_train.columns)}
    X_train.columns = [nama_ch[x] for x in X_train.columns]
    train_features, test_features, train_labels, test_labels = train_test_split(X_train, y_km,random_state=2020)
    lgb = LGBMClassifier(n_estimators=2000,learning_rate=0.04,subsample=0.7,colsample_bytree=0.8,random_state=2020,objective='multiclass')
    lgb.fit(train_features,train_labels,early_stopping_rounds = 100,eval_set = [(test_features,test_labels)],verbose=True)
    lgbm_score = accuracy_score(lgb.predict(test_features),test_labels)

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



def clustering(data,y_km,y):
    """
    make cluster for train and test set using models from return of modeling func.
    return : clustered dataframe
    """  
    X_train = data.iloc[:36250,:]
    X_test = data.iloc[36250:,:]
    lgb = LGBMClassifier(n_estimators=1194,learning_rate=0.045,subsample=0.8,colsample_bytree=0.6,random_state=2020,objective='multiclass')
    nama_ch = {v:k for k,v in enumerate(X_train.columns)}
    X_train.columns = [nama_ch[x] for x in X_train.columns]
    lgb.fit(X_train,y_km)
    
    X_train = data.iloc[:36250,:]
    X_test = data.iloc[36250:,:]
    X_train = pd.concat([X_train,y],axis=1)

    X_test['kmeans'] = lgb.predict(X_test)
    X_train['kmeans'] = y_km

    X_train_c0 = X_train[X_train['kmeans']==0]
    X_train_c1 = X_train[X_train['kmeans']==1]
    X_train_c2 = X_train[X_train['kmeans']==2]

    X_test_c0 = X_test[X_test['kmeans']==0]
    X_test_c1 = X_test[X_test['kmeans']==1]
    X_test_c2 = X_test[X_test['kmeans']==2]

    return X_train_c0, X_train_c1, X_train_c2, X_test_c0, X_test_c1, X_test_c2