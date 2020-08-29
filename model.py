from util import *

def mk_trainset(data):
    data['sales_per'] = np.log1p(data['판매단가'])
    data.rename(columns={'마더코드':'mcode','상품군':'cate','노출(분)':'length_raw','상품코드':'item_code'},inplace=True)
    data = pd.get_dummies(data,columns=(['gender','pay','cate','day','hour','hour_gr','min','min_gr','len_gr','mcode_freq_gr']))
    data['mcode'] = data['mcode'].astype('str').apply(lambda x: x[3:])
    data['mcode'] = data['mcode'].astype(int)
    data['item_code'] = data['item_code'].astype('str').apply(lambda x: x[2:])
    data['item_code'] = data['item_code'].astype(int)
    data = data.drop(['방송일시','상품명','판매단가'],axis=1)
    return data

def modeling(data,y_km):
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

# excution
if '__name__'=='__main__': 

    ## data load 
    perform_raw = pd.read_csv('data/2019_performance.csv')
    rating = pd.read_csv('data/2019_rating.csv',encoding='utf-8')
    test = pd.read_csv('data/question.csv')

    perform_raw.reset_index(inplace=True)
    perform_raw.rename(columns={'index':'id'})
    test.reset_index(inplace=True)
    test.rename(columns={'index':'id'})

    data, y, y_km = preprocess(perform_raw,0.03,3)
    data = mk_trainset(data)
    lgb, ensemble = modeling(data,y,y_km)
