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
from dim_reduction import by_PCA
import pandas as pd
import seaborn as sns
import numpy as np
import joblib
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from sklearn.preprocessing import RobustScaler
robustScaler = RobustScaler()

def boosting(X,y,X_val,robustScaler,col_sample=0.6,lr=0.04,iter=1500,inference=True):
    
    X_val1 = X_val[X_val['is_mcode']==1]
    X_val0 = X_val[X_val['is_mcode']==0]
    y_val1 = X_val1['sales']
    y_val0 = X_val0['sales']

    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    model_lgb1 = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    model_lgb0 = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)

    if inference:
        model_lgb1.fit(X.drop(drop1_+['is_mcode'],axis=1),y,verbose=False)
        model_lgb0.fit(X.drop(drop0_+['is_mcode','mcode_freq','mcode_freq_gr','mcode_sales_mean','mcode_sales_std','mcode_sales_med','mcode_sales_rank','mcode_order_mean','mcode_order_med','mcode_order_rank','mcode_order_std'],axis=1),y,verbose=False)

        pred_lgb1 = model_lgb1.predict(X_val1.drop(drop1_+['is_mcode','sales'],axis=1))
        res1 = pd.concat([y_val1.reset_index(drop=True),pd.DataFrame(pred_lgb1,columns=['pred'])],axis=1)
        real1 = robustScaler.inverse_transform(np.array(res1['sales']).reshape(-1,1))
        pred1 = robustScaler.inverse_transform(np.array(res1['pred']).reshape(-1,1))
        print(f'real1 : {real1.shape},pred1 : {pred1.shape}')

        pred_lgb0 = model_lgb0.predict(X_val0.drop(drop0_+['is_mcode','sales','mcode_freq','mcode_freq_gr','mcode_sales_mean','mcode_sales_std','mcode_sales_med','mcode_sales_rank','mcode_order_mean','mcode_order_med','mcode_order_rank','mcode_order_std'],axis=1))
        res0 = pd.concat([y_val0.reset_index(drop=True),pd.DataFrame(pred_lgb0,columns=['pred'])],axis=1)
        real0 = robustScaler.inverse_transform(np.array(res0['sales']).reshape(-1,1))
        pred0 = robustScaler.inverse_transform(np.array(res0['pred']).reshape(-1,1))
        print(f'real0 : {real0.shape},pred0 : {pred0.shape}')

        score = (metric(real1,pred1) + metric(real0 , pred0))/2
        print(len(res1),round(metric(real1,pred1),2),len(res0),round(metric(real0 , pred0),2))
        length = len(res0) + len(res1)
        real = np.concatenate((real1,real0))
        pred = np.concatenate((pred1,pred0))
        return score, length, pd.DataFrame({'real':real.flatten(), 'pred':pred.flatten()},columns=['real','pred']),model_lgb0,model_lgb1

    else:
        model_lgb.fit(X.drop(['is_mcode'],axis=1),y)
        pred_lgb = model_lgb.predict(X.drop(['is_mcode'],axis=1))
        res = pd.concat([y.reset_index(drop=True),pd.DataFrame(pred_lgb,columns=['pred'])],axis=1)
        real = robustScaler.inverse_transform(np.array(res['sales']).reshape(-1,1))
        pred = robustScaler.inverse_transform(np.array(res['pred']).reshape(-1,1))
        print(real.shape,pred.shape)

        return metric(real,pred), len(res), pd.DataFrame({'real':real.flatten(), 'pred':pred.flatten()},columns=['real','pred']),model_lgb0, model_lgb1



# def boosting_2(pop,inference,robustScaler,col_sample=0.6,lr=0.04,iter=1500,test=True):
#     y = pop['sales']
#     X = pop.drop(['id','sales','kmeans'],axis=1)
#     model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    
#     if test==True:
#         model_lgb.fit(X,y)
#         pred_lgb = model_lgb.predict(inference.drop(['id','sales','kmeans'],axis=1))
#         res = pd.concat([inference.reset_index(drop=True),pd.DataFrame(pred_lgb,columns=['pred'])],axis=1)
#         real = robustScaler.inverse_transform(np.array(res['sales']).reshape(-1,1))
#         pred = robustScaler.inverse_transform(np.array(res['pred']).reshape(-1,1))

#         return metric(real,pred), pd.DataFrame({'real':real.flatten(), 'pred':pred.flatten()},columns=['real','pred']), model_lgb
     
#     else:
#         X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=2020)
#         model_lgb.fit(X_train,y_train,early_stopping_rounds = 1000,eval_set = [(X_val,y_val)],verbose=False)
#         pred_lgb = model_lgb.predict(X_val)
#         res = pd.concat([y_val.reset_index(drop=True),pd.DataFrame(pred_lgb,columns=['pred'])],axis=1)
#         real = robustScaler.inverse_transform(np.array(res['sales']).reshape(-1,1))
#         pred = robustScaler.inverse_transform(np.array(res['pred']).reshape(-1,1))

#         return metric(real,pred), pd.DataFrame({'real':real.flatten(), 'pred':pred.flatten()},columns=['real','pred']), model_lgb


def predict(X_train,val,k,robustScaler,col_sample=0.6,lr=0.04,iter=1500,inference=True):
    """
    predict '취급액' score only using train set(perform)
    return : RMAE score for each cluster
    """
    origin, originlen, tmp, model0,model1 = boosting(X_train.drop(['id','sales','kmeans'],axis=1),X_train['sales'],val.drop(['id','kmeans'],axis=1),robustScaler,col_sample,lr,iter,inference)
    print(f'origin error : {round(origin,2)}%\n')

    
    sum = 0
    total_len = 0
    for i in range(k):
        train_tem = X_train[X_train['kmeans']==i]
        val_tem = val[val['kmeans']==i]

        score,len,pred,_,_ = boosting(train_tem.drop(['sales','kmeans','id'],axis=1),train_tem['sales'],val_tem.drop(['kmeans','id'],axis=1),robustScaler,col_sample,lr,iter,inference)
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

    return fin_results, model0, model1



# def second_predict(input,robustScaler,train,val):
#     neg = input.iloc[:5000,:]
#     pos = input.iloc[5000:,:].sample(n=5000,random_state=2020)
#     population = pd.concat([neg,pos]).reset_index(drop=True)

#     result,model_l,_ = predict(train,val,3,robustScaler,inference=True,iter=1500)
#     # result_0 = result[result['kmeans']==0]

#     score,res,model_lgb = boosting_2(population.iloc[:,:-3],result.iloc[:,:-3],robustScaler,0.55,0.04,1500,True)
#     result.reset_index(drop=True, inplace=True)
#     result.reset_index(drop=True, inplace=True)
#     result['pred_eva'] = (res['pred']*0.5 + result['pred']*0.5)
#     print(metric(result['real'],result['pred_eva']))
#     # print(f'second_fit cluster0  : {score}')
#     # fin_score = (metric(result_0['pred_eva'], result_0['real'])*1467 + result[result['kmeans']!=0]['MAPE'].sum())/2746
#     # print(f'final error  : {round(fin_score,2)}%')
#     return model_lgb

def final_test(train,test,k,robustScaler,col_sample=0.6,lr=0.04,iter=1500):

    test0 = test[test['is_mcode']==0]
    test1 = test[test['is_mcode']==1]

    model_total0 = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    model_total1 = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)

    model_total0.fit(train.drop(drop0_+['id','sales','kmeans','is_mcode','mcode_freq','mcode_freq_gr','mcode_sales_mean','mcode_sales_std','mcode_sales_med','mcode_sales_rank','mcode_order_mean','mcode_order_med','mcode_order_rank','mcode_order_std'],axis=1),train['sales'],verbose=False)
    pred_total0 = model_total0.predict(test0.drop(drop0_+['id','sales','kmeans','is_mcode','mcode_freq','mcode_freq_gr','mcode_sales_mean','mcode_sales_std','mcode_sales_med','mcode_sales_rank','mcode_order_mean','mcode_order_med','mcode_order_rank','mcode_order_std'],axis=1))
    pred_total0 = robustScaler.inverse_transform(pred_total0.reshape(-1,1))
    res0 = pd.concat([test0.reset_index(drop=True),pd.DataFrame(pred_total0,columns=['pred'])],axis=1)

    model_total1.fit(train.drop(drop1_+['id','sales','kmeans','is_mcode'],axis=1),train['sales'],verbose=False)
    pred_total1 = model_total1.predict(test1.drop(drop1_+['id','sales','kmeans','is_mcode'],axis=1))
    pred_total1 = robustScaler.inverse_transform(pred_total1.reshape(-1,1))
    res1 = pd.concat([test1.reset_index(drop=True),pd.DataFrame(pred_total1,columns=['pred'])],axis=1)

    total_pred = pd.concat([res0,res1])

    for i in range(k):
        train_tem = train[train['kmeans']==i]
        test_tem = test[test['kmeans']==i]
        
        test_tem_0 = test_tem[test_tem['is_mcode']==0]
        test_tem_1 = test_tem[test_tem['is_mcode']==1]

        best_params = joblib.load(f'best_lgbm_params_{i+1}.pkl')

        model_cluster0 = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample,**best_params,random_state=2020)
        model_cluster1 = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample,**best_params,random_state=2020)

        model_cluster0.fit(train_tem.drop(drop0_+['id','sales','kmeans','is_mcode','mcode_freq','mcode_freq_gr','mcode_sales_mean','mcode_sales_std','mcode_sales_med','mcode_sales_rank','mcode_order_mean','mcode_order_med','mcode_order_rank','mcode_order_std'],axis=1),train_tem['sales'],verbose=False)
        pred_cluster0 = model_cluster0.predict(test_tem_0.drop(drop0_+['id','sales','kmeans','is_mcode','mcode_freq','mcode_freq_gr','mcode_sales_mean','mcode_sales_std','mcode_sales_med','mcode_sales_rank','mcode_order_mean','mcode_order_med','mcode_order_rank','mcode_order_std'],axis=1))
        pred_cluster0 = robustScaler.inverse_transform(pred_cluster0.reshape(-1,1))
        res_tem0 = pd.concat([test_tem_0.reset_index(drop=True),pd.DataFrame(pred_cluster0,columns=['pred'])],axis=1)

        model_cluster1.fit(train_tem.drop(drop1_+['id','sales','kmeans','is_mcode'],axis=1),train_tem['sales'],verbose=False)
        pred_cluster1 = model_cluster1.predict(test_tem_1.drop(drop1_+['id','sales','kmeans','is_mcode'],axis=1))
        pred_cluster1 = robustScaler.inverse_transform(pred_cluster1.reshape(-1,1))
        res_tem1 = pd.concat([test_tem_1.reset_index(drop=True),pd.DataFrame(pred_cluster1,columns=['pred'])],axis=1)

        cluster_pred_tem = pd.concat([res_tem0,res_tem1])

        if i ==0:
            cluster_pred = cluster_pred_tem.copy()
        else:
            cluster_pred = pd.concat([cluster_pred,cluster_pred_tem])
    
    return total_pred.sort_values(by='id').reset_index(drop=True), cluster_pred.sort_values(by='id').reset_index(drop=True)


# excution
if __name__=='__main__': 
    data_path = 'data/'
    perform_raw, rating, test_raw = load_data(data_path,trend=False,weather=False,query=False)
    train_var, test_var = make_variable(perform_raw,test_raw,rating)
    raw_data, y_km, train_len= preprocess(train_var,test_var,0.03,3,inner=False) 
    data = mk_trainset(raw_data,categorical=True) # lgbm만 categorical = True, 나머지 모델은 False -> one-hot encoding
    train, val, robustScaler = clustering(data,y_km,train_len,test=True) # test 할때만 test = True

    # permutation으로 날릴 변수들 
    # lgbm 기준이라서 one-hot 안된 카테고리 변수들이 있음, 다른 모델 랜덤 서치 돌릴 때는 
    # 해당 변수들은 이름이 없을테니(ex. min -> min_0, min_1, min_2 ...)
    # 알아서 에러나는거 보고 빼던가 미리 카테고리 변수는 drop 리스트에서 빼 놓으셈

    # 그리고 변수 drop은 0,1 모델 기준으로 한거라 cluster 기준으로 랜덤서치하는 거랑 안 맞을 수 있음
    # 혜린이한테는 일단 두 리스트 교집합으로 하라 했는데 더 좋은 방법 있음 생각해서 시도 ㄱㄱ

    drop1_  = ['min_sales_med',  'min_sales_std',  'day_sales_rank', 'min_sales_rank',
    'min_order_rank', 'cate_sales_rank', 'cate_order_rank', 'cate_order_med',
    'cate_sales_med', 'prime', 'min_order_std', 'min_sales_mean',
    'day_order_rank', 'cate_order_std', 'min_order_med', 'min_order_mean',
    'cate_sales_mean', 'cate_sales_std', 'day_order_std', 'rating',
    'day_sales_med', 'min', 'day_order_med']
    drop0_ = ['min_order_med', 'day_order_rank', 'min_sales_rank', 'min_sales_med',
    'day_sales_rank', 'min_order_rank', 'cate_order_std', 'cate_sales_rank',
    'min_sales_std', 'min_order_std', 'min', 'cate_order_rank',
    'min_order_mean', 'rating', 'min_sales_mean', 'prime', 'cate_order_med',
    'day_order_med']

    # val 코드(test할 땐 실행 X)
    # tem_result,model0,model1 = predict(train,val,3,robustScaler,inference=True,iter=2000) 
    
    # 테스트 코드 
    """
    total : 클러스터 안나누고 한번에 돌린 결과
    cluster : 클러스터별로 따로 모델돌린거 합친 결과
    """
    total, cluster = final_test(train,val,3,robustScaler,0.6,0.04,2000)
    #total.to_csv('final_predict.csv')


# def feature_select(val,mcode,var,model):
#     val_sel = val[val['is_mcode']==mcode]
#     val_sel.reset_index(drop=True,inplace=True)
#     val_features = val_sel.drop(var,axis=1)
#     r = permutation_importance(model, val_features, val_sel['sales'],
#                             n_repeats=30,
#                             random_state=0)
#     drop = []
#     for i in r.importances_mean.argsort():
#     # print(i)
#     # if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
#         drop.append(val_features.columns[i])
#         print(f"{val_features.columns[i]:<8}"
#             f"{r.importances_mean[i]:.3f}"
#             f" +/- {r.importances_std[i]:.3f}")
#     return drop







