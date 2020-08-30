from clustering import *


def predict(X_train_c0,X_train_c1,X_train_c2):
    """
    predict '취급액' score only using train set(perform)
    return : RMAE score for each cluster
    """
    train_features, test_features, train_labels, test_labels = train_test_split(X_train_c0, X_train_c0['취급액'],random_state=2020)
    nama_ch = {v:k for k,v in enumerate(train_features.columns)}
    train_features.columns = [nama_ch[x] for x in train_features.columns]
    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= 0.7, learning_rate=0.05,n_estimators=1272)
    model_lgb.fit(train_features,train_labels)
    pred_lgb = model_lgb.predict(test_features)
    c0 = metric(test_labels,pred_lgb)


    train_features, test_features, train_labels, test_labels = train_test_split(X_train_c1, X_train_c1['취급액'],random_state=2020)
    nama_ch = {v:k for k,v in enumerate(train_features.columns)}
    train_features.columns = [nama_ch[x] for x in train_features.columns]
    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= 0.7, learning_rate=0.05,n_estimators=1272)
    model_lgb.fit(train_features,train_labels)
    pred_lgb = model_lgb.predict(test_features)
    c1 = metric(test_labels,pred_lgb)

    train_features, test_features, train_labels, test_labels = train_test_split(X_train_c2, X_train_c2['취급액'],random_state=2020)
    nama_ch = {v:k for k,v in enumerate(train_features.columns)}
    train_features.columns = [nama_ch[x] for x in train_features.columns]
    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= 0.7, learning_rate=0.05,n_estimators=1272)
    model_lgb.fit(train_features,train_labels)
    pred_lgb = model_lgb.predict(test_features)
    c2 = metric(test_labels,pred_lgb)

    print(f'c0 : {c0}, c1 : {c1}, c2 : {c2}')


# excution
if '__name__'=='__main__': 

    ## data load 
    perform_raw = pd.read_csv('data/2019_performance.csv')
    rating = pd.read_csv('data/2019_rating.csv',encoding='utf-8')
    test = pd.read_csv('data/question.csv')

    perform_raw.reset_index(inplace=True)
    perform_raw.rename(columns={'index':'id'},inplace=True)
    test.reset_index(inplace=True)
    test.rename(columns={'index':'id'},inplace=True)

    data, y, y_km = preprocess(perform_raw,0.03,3)
    data = mk_trainset(data)
    lgb, ensemble = modeling(data,y_km)
    X_train_c0, X_train_c1, X_train_c2, X_test_c0, X_test_c1, X_test_c2 = clustering(data,y_km,y)

    predict(X_train_c0, X_train_c1, X_train_c2)