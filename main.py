from lightgbm.callback import early_stopping, reset_parameter
from pandas.core.arrays import categorical
from pandas.io.pytables import Term
from sklearn import dummy
from make_var import make_variable
from util import load_data,preprocess,mk_trainset, metric
from clustering import clustering
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from dim_reduction import train_AE,by_AE,by_PCA
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier



def boosting(X,y,X_val,y_val,robustScaler,col_sample=0.6,lr=0.04,iter=50000,inference=True):

    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    model_lgb.fit(X,y,early_stopping_rounds = 500,eval_set = [(X_val,y_val)],verbose=False)
    if inference:
        pred_lgb = model_lgb.predict(X_val)
        res = pd.concat([y_val.reset_index(drop=True),pd.DataFrame(pred_lgb,columns=['pred'])],axis=1)
    else:
        pred_lgb = model_lgb.predict(X)
        res = pd.concat([y.reset_index(drop=True),pd.DataFrame(pred_lgb,columns=['pred'])],axis=1)

    # print(res)
    real = robustScaler.inverse_transform(np.array(res['sales']).reshape(-1,1))
    pred = robustScaler.inverse_transform(np.array(res['pred']).reshape(-1,1))
    print(real.shape,pred.shape)

    return metric(real,pred), len(res), pd.DataFrame({'real':real.flatten(), 'pred':pred.flatten()},columns=['real','pred']),model_lgb



def boosting_2(pop,inference,robustScaler,col_sample=0.6,lr=0.04,iter=50000,test=True):
    y = pop['sales']
    X = pop.drop(['id','sales','kmeans'],axis=1)
    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    
    if test==True:
        model_lgb.fit(X,y)
        pred_lgb = model_lgb.predict(inference.drop(['id','sales','kmeans'],axis=1))
        res = pd.concat([inference.reset_index(drop=True),pd.DataFrame(pred_lgb,columns=['pred'])],axis=1)
        real = robustScaler.inverse_transform(np.array(res['sales']).reshape(-1,1))
        pred = robustScaler.inverse_transform(np.array(res['pred']).reshape(-1,1))

        return metric(real,pred), pd.DataFrame({'real':real.flatten(), 'pred':pred.flatten()},columns=['real','pred']), model_lgb
     
    else:
        X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=2020)
        model_lgb.fit(X_train,y_train,early_stopping_rounds = 1000,eval_set = [(X_val,y_val)],verbose=False)
        pred_lgb = model_lgb.predict(X_val)
        res = pd.concat([y_val.reset_index(drop=True),pd.DataFrame(pred_lgb,columns=['pred'])],axis=1)
        real = robustScaler.inverse_transform(np.array(res['sales']).reshape(-1,1))
        pred = robustScaler.inverse_transform(np.array(res['pred']).reshape(-1,1))

        return metric(real,pred), pd.DataFrame({'real':real.flatten(), 'pred':pred.flatten()},columns=['real','pred']), model_lgb


def predict(X_train,val,k,robustScaler,col_sample=0.6,lr=0.04,iter=50000,inference=True):
    """
    predict '취급액' score only using train set(perform)
    return : RMAE score for each cluster
    """
    origin, originlen, tmp, model = boosting(X_train.drop(['id','sales','kmeans'],axis=1),X_train['sales'],val.drop(['id','sales','kmeans'],axis=1),val['sales'],robustScaler,col_sample,lr,iter,inference)
    print(f'origin error : {round(origin,2)}%\n')

    
    sum = 0
    total_len = 0
    for i in range(k):
        train_tem = X_train[X_train['kmeans']==i]
        val_tem = val[val['kmeans']==i]

        score,len,pred,model_cluster = boosting(train_tem.drop(['sales','kmeans','id'],axis=1),train_tem['sales'],val_tem.drop(['sales','kmeans','id'],axis=1),val_tem['sales'],robustScaler,col_sample,lr,iter,inference)
        if inference==True:
            results = pd.concat([val_tem.reset_index(drop=True),pred],axis=1)
        else:
            results = pd.concat([train_tem.reset_index(drop=True),pred],axis=1)

        results['MAPE'] = pred.apply(lambda x: metric(x['real'],x['pred']),axis=1)

        if i == 0:
            fin_results = results.copy()

        else:
            fin_results = pd.concat([fin_results,results])
            
        sum += (score * len)
        total_len += len
        print(f'Cluster_{i} : {round(score,2)}%\n')
    print(f'Total error : {round(sum/total_len,2)}%')

    return fin_results, model


def second_predict(input,robustScaler,train,val):
    neg = input.iloc[:5000,:]
    pos = input.iloc[5000:,:].sample(n=5000,random_state=2020)
    population = pd.concat([neg,pos]).reset_index(drop=True)

    result = predict(train,val,3,robustScaler,inference=True,iter=10000)
    result_0 = result[result['kmeans']==0]

    score,res,model_lgb = boosting_2(population.iloc[:,:-3],result_0.iloc[:,:-3],robustScaler,0.55,0.04,1500,True)
    result_0['pred_eva'] = (res['pred']*0.4 + result_0['pred']*0.6)
    # print(f'second_fit cluster0  : {score}')
    fin_score = (metric(result_0['pred_eva'], result_0['real'])*1412 + result[result['kmeans']!=0]['MAPE'].sum())/2746
    print(f'final error  : {round(fin_score,2)}%')
    return model_lgb


# excution
if __name__=='__main__': 
    data_path = 'data/'
    perform_raw, rating, test_raw = load_data(data_path,trend=False,weather=False)
    train_var, test_var = make_variable(perform_raw,test_raw,rating)
    raw_data, y_km, train_len= preprocess(train_var,test_var,0.03,3,inner=False) # train, test 받아서 쓰면 돼
    data = mk_trainset(raw_data,categorical=True)
    train, val, robustScaler = clustering(data,y_km,train_len)
    
    tem_result, clf = predict(train,val,3,robustScaler,inference=True,iter=10000)
    clf = second_predict(tem_result,robustScaler,train,val)






"""
k = tem_result.iloc[:,:-3].drop(['id','sales','kmeans'],axis=1)

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# sorted(zip(clf.feature_importances_, X.columns), reverse=True)
feature_imp = pd.DataFrame(sorted(zip(model_lgb.feature_importances_,val_features.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()

plt.savefig('lgbm_importances-01.png')





from sklearn.inspection import permutation_importance
r = permutation_importance(model_lgb, val_features, val_labels,
                            n_repeats=100,
                            random_state=0)

for i in r.importances_mean.argsort():
    # if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
    print(f"{val_features.columns[i]:<8}"
        f"{r.importances_mean[i]:.3f}"
        f" +/- {r.importances_std[i]:.3f}")
r.importances_mean.argsort()[::-1]
len(r.importances_mean)



"""