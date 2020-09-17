from lightgbm.callback import early_stopping
from util_hye import load_data, preprocess_hye, mk_trainset_hye, metric, scoring
from clustering import clustering
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
#from dim_reduction import train_AE,by_AE,by_PCA
import pandas as pd
import numpy as np

# six 무슨 의미?

def boosting(X,y,X_val,y_val,col_sample=0.6,lr=0.04,iter=1500,six=True):
    
    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    model_lgb.fit(X,y,early_stopping_rounds = 500,eval_set = [(X_val,y_val)],verbose=False)
    pred_lgb = model_lgb.predict(X_val)
    
    res = pd.concat([y_val.reset_index(drop=True),pd.DataFrame(pred_lgb,columns=['pred'])],axis=1)
    res['score'] = res.apply(lambda x : scoring(x['sales'],x['pred']),axis=1)
    return res['score'].mean(), len(res)

def predict(X_train,val,k,col_sample=0.6,lr=0.04,iter=1500,six=True):
    """
    predict '취급액' score only using train set(perform)
    return : RMAE score for each cluster
    """
    X_train = X_train[X_train['sales']!=0]

    origin, originlen = boosting(X_train.drop(['id','sales','kmeans'],axis=1),X_train['sales'],val.drop(['sales','kmeans','id','kmeans_pred'],axis=1),val['sales'],col_sample,lr,iter,six)
    print(f'origin error : {round(origin,2)}%\n')

    sum = 0
    total_len = 0

    for i in range(k):
        train_tem = X_train[X_train['kmeans']==i]
        val_tem = val[val['kmeans_pred']==i]
        score,len = boosting(train_tem.drop(['sales','kmeans','id'],axis=1),train_tem['sales'],val_tem.drop(['sales','kmeans','id','kmeans_pred'],axis=1),val_tem['sales'],col_sample,lr,iter,six)
        sum += (score * len)
        total_len += len
        print(f'Cluster_{i} : {round(score,2)}%\n')
    print(f'Total error : {round(sum/total_len,2)}%')

def mk_sid_df(data,y,y_km):
    # data - after mk_trainset_hye
    # sales - 취급액, y_km - kmeans 결과

    # col - id, mcode, item_code, cate, show_id, length, month, season, day, holiday, hour, min, mcode_freq, pcode_freq, pcode_count, rating_byshow
    data = pd.concat([data,y,y_km],axis=1)
    sid_icode = pd.DataFrame(data.groupby('show_id')['item_code'].apply(lambda x: list(set(x)))).reset_index()
 
    sid_df = pd.DataFrame({
        col:np.repeat(sid_icode['show_id'].values,sid_icode['item_code'].str.len())
        for col in sid_icode.columns.drop('item_code')
    }).assign(**{'item_code': np.concatenate(sid_icode['item_code'].values)})
    # show_id, item_code 
    # len - 13837

    # 변수 생성
    # 1. mcode merge
    codes = data[['mcode','item_code']]
    codes.drop_duplicates(["item_code"],inplace=True)

    sid_df = pd.merge(sid_df,codes,on='item_code',how='left')

    # 2. cate - 모든 방송은 한 카테고리 물품 판매하는 것 확인함.
    cates = data[['cate','item_code']]
    cates.drop_duplicates(["item_code"],inplace=True)

    sid_df = pd.merge(sid_df,cates,on='item_code',how='left')

    # 3. length - 단순 합.
    sid_len = pd.DataFrame(data.groupby('show_id')['length'].apply(list)).reset_index()
    sid_len.head(3)

    sid_len['length'].str.len()/sid_icode['item_code'].str.len()


    lengths = data[['']]

    # 3. length
    
    


# excution
#if __name__=='__main__': 
data_path = 'data/'
perform_raw, rating, test = load_data(data_path)
raw_data, y, y_km = preprocess_hye(perform_raw,test,0.03,3) # del outlier drop
data = mk_trainset_hye(raw_data) # one hot encoding X, 불필요한 column만 drop

data,y,y_km = mk_sid_df(data,y,y_km)

y_km.head(3)


    # raw_data > train+test mk_var 거친 후. (원핫 인코딩 X)




    #data = mk_trainset(raw_data)
    #train, val = clustering(data,y_km,y)
    #predict(train,val,5)

# 과정
# 방송별로 모은 df > 시간은 어떻게 표시? / 상품 여러개 - 다른것. 마더코드, 상품코드, 방송 빈도?, 여성, 남성~ 
# ㄴ 변수 생성 후 모을지? df 만들어놓고 변수 만들어갈지?

# 방송별 df 꼴) 방송 ID column 有 > 판매 상품 개수만큼 row. 변수 column 쭉. 방송 순서 관련 column은 drop. 시간, 분은 어떻게 처리?
# >> 시간 같은 건 평균내서?  ex. 3 3 4 면 3.3 (연속형)

import pickle

with open('NOT_FOR_GIT/2019_byshow.pkl', 'rb') as f:
    data = pickle.load(f)

