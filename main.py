from sklearn.utils.validation import column_or_1d
from clustering import *


def predict(X_train_c0,X_train_c1,X_train_c2,data,y):
    """
    predict '취급액' score only using train set(perform)
    return : RMAE score for each cluster
    """
    train_features, test_features, train_labels, test_labels = train_test_split(data.iloc[:36250,:], y['sales'],random_state=2020)
    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= 0.7, learning_rate=0.05,n_estimators=1000,random_state=2020)
    model_lgb.fit(train_features,train_labels,early_stopping_rounds = 200,eval_set = [(test_features,test_labels)],verbose=False)
    pred_lgb = model_lgb.predict(test_features)
    origin = metric(test_labels,pred_lgb)

    y = X_train_c0['sales']
    X_train = X_train_c0.drop(['sales'],axis=1)
    X_train = X_train.drop(['kmeans'],axis=1)
    train_features, test_features, train_labels, test_labels = train_test_split(X_train, y,random_state=2020)
    # nama_ch = {v:k for k,v in enumerate(train_features.columns)}
    # train_features.columns = [nama_ch[x] for x in train_features.columns]
    model_lgb = LGBMRegressor(subsample= 0.8, colsample_bytree= 0.55, learning_rate=0.075,n_estimators=1000,random_state=2020)
    model_lgb.fit(train_features,train_labels,early_stopping_rounds = 200,eval_set = [(test_features,test_labels)],verbose=False)
    pred_lgb = model_lgb.predict(test_features)
    c0 = metric(test_labels,pred_lgb)
    len0 = len(test_labels)
    print(f'c0 error : {round(c0,4)}%  learning_rate : 0.077')

    y = X_train_c1['sales']
    X_train = X_train_c1.drop(['sales'],axis=1)
    X_train = X_train.drop(['kmeans'],axis=1)
    X_train = X_train.drop(['length_raw'],axis=1)
    train_features, test_features, train_labels, test_labels = train_test_split(X_train, y,random_state=2020)
    # nama_ch = {v:k for k,v in enumerate(train_features.columns)}
    # train_features.columns = [nama_ch[x] for x in train_features.columns]
    model_lgb = LGBMRegressor(subsample= 0.8, colsample_bytree= 0.55, learning_rate=0.05,n_estimators=1000,random_state=2020)
    model_lgb.fit(train_features,train_labels,early_stopping_rounds = 200,eval_set = [(test_features,test_labels)],verbose=False)
    pred_lgb = model_lgb.predict(test_features)
    c1 = metric(test_labels,pred_lgb)
    print(c1)

    y = X_train_c2['sales']
    X_train = X_train_c2.drop(['sales'],axis=1)
    X_train = X_train.drop(['kmeans'],axis=1)
    train_features, test_features, train_labels, test_labels = train_test_split(X_train, y,random_state=2020)
    # nama_ch = {v:k for k,v in enumerate(train_features.columns)}
    # train_features.columns = [nama_ch[x] for x in train_features.columns]
    model_lgb = LGBMRegressor(subsample= 0.8, colsample_bytree= 0.8, learning_rate=0.05,n_estimators=1500,random_state=2020)
    model_lgb.fit(train_features,train_labels,early_stopping_rounds = 200,eval_set = [(test_features,test_labels)],verbose=False)
    pred_lgb = model_lgb.predict(test_features)
    c2 = metric(test_labels,pred_lgb)
    len2 = len(test_labels)
    total_error = (c0 * len0 + c1 * len1 + c2 * len2)/(len0+len1+len2)

    print(f'origin : {round(origin,4)}, c0 : {round(c0,4)}%, c1 : {round(c1,4)}%, c2 : {round(c2,4)}%\n Total error {round(total_error,4)}%')



# excution
if '__name__'=='__main__': 

    perform_raw.reset_index(inplace=True)
    perform_raw.rename(columns={'index':'id'},inplace=True)
    test.reset_index(inplace=True)
    test.rename(columns={'index':'id'},inplace=True)

    data, y, y_km = preprocess(perform_raw,0.03,3)
    data = mk_trainset(data)
    lgb, ensemble = modeling(data,y_km)
    X_train_c0, X_train_c1, X_train_c2, X_test_c0, X_test_c1, X_test_c2 = clustering(data,y_km,y)

    predict(X_train_c0, X_train_c1, X_train_c2,data,y)

