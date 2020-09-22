#-*-coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.reshape.reshape import get_dummies
import seaborn as sns
plt.rc('font', family='NanumBarunGothic')
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from make_var_func import mk_var, mk_stat_var
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter



## load data
def load_data(data_path,weather=False,trend=False):
    perform_raw = pd.read_csv(os.path.join(f'{data_path}','2019_performance.csv'))
    test = pd.read_csv(os.path.join(f'{data_path}','question.csv'))
    rating = pd.read_csv(os.path.join(f'{data_path}','2019_rating.csv'),encoding='utf-8')

    if weather:
        weather_train = pd.read_csv(os.path.join(f'{data_path}','weather_data.csv'))
        weather_test = pd.read_csv(os.path.join(f'{data_path}','weather_data_test.csv'))
        perform_raw = pd.concat([perform_raw,weather_train],axis=1)
        test = pd.concat([test,weather_test],axis=1)
    if trend:
        trend_train = pd.read_csv(os.path.join(f'{data_path}','shopping_trend.csv'))
        trend_test = pd.read_csv(os.path.join(f'{data_path}','shopping_trend_test.csv'))
        perform_raw = pd.concat([perform_raw,trend_train],axis=1)
        test = pd.concat([test,trend_test],axis=1)

    perform_raw.reset_index(inplace=True)
    perform_raw.rename(columns={'index':'id'},inplace=True)

    test.reset_index(inplace=True)
    test.rename(columns={'index':'id'},inplace=True)
    test['id'] = test['id'] + 37372

    return perform_raw, rating, test

def mk_sid(perform_raw,test_raw):
    
    def preprocess(name):

        # 예외 처리
        name = name.replace('휴롬퀵스퀴저','휴롬 퀵스퀴저')
        name = name.replace('장수흙침대','장수 흙침대')
        name = name.replace('1+1 국내제조', '국내제조')

        words = ['보국미니히터', '우아미', '갓바위', '두씽', '해뜰찬', '법성포굴비', '쥐치포', '공간아트', '르젠','쿠쿠']

        # name 이 words 안에 있다면 name 전체를 위의 word로 대체.
        for word in words:
            if (word in name):
                name = word

        if ('삼성' in name) and ('도어' in name):
            name = 'tmp'

        brands = ['프라다','구찌','버버리','코치','마이클코어스','톰포드', '페라가모', '생로랑']

        for brand in brands:
            name = name.replace(brand,'명품')

        # 상품명 처리
        words = ['LG전자','삼성','LG','(일)3인용', '(일)4인용', '(무)3인용', '(무)4인용', '(삼성카드 6월 5%)',\
        '[1세트]','[2세트]','[SET]','[풀패키지]','[실속패키지]', '(점보특대형)','(점보형)', '(중형)',\
        '(퀸+퀸)','(킹+싱글)','(퀸+싱글)','(킹사이즈)','(퀸사이즈)','(더블사이즈)','(싱글사이즈)','(싱글+싱글)','(더블+더블)','(더블+싱글)',\
        '(점보)','(특대)','(대형)','더커진','(1등급)467L_','(1등급)221L_','1세트 ','2세트 ','5세트 ','19년 신제품 ',\
        'K-SWISS 남성','K-SWISS 여성','(퀸)','(싱글)','[무이자]','[일시불]','(무)','(일)','무)','일)','무이자','일시불']

        for word in words:
            name = name.replace(word,'')
        
        name = name.strip()

        return name.split(' ')[0]

    perform_raw['istrain'] = 1
    test_raw['istrain'] = 0

    data = pd.concat([perform_raw,test_raw])

    tmp = data['상품명'].map(preprocess)
    names = tmp.tolist()

    # 전날 새벽, 다음날 아침 같은 물건 파는 경우 有 - id 31149~31153
    names[30399] = '레이프릴1'
    names[30400] = '레이프릴1'

    ids = [0]

    for i, name in enumerate(names[1:],1):
        prior = names[i-1]

        if prior == name:
            ids.append(ids[-1])
        else:
            ids.append(ids[-1]+1)

    data['show_id'] = np.array(ids)

    perform_raw = data[data['istrain'] == 1]
    test_raw = data[data['istrain'] == 0]
    
    return perform_raw.drop('istrain',axis=1), test_raw.drop('istrain',axis=1)


## detect outlier
"""
x : dataset
p : ratio of deleted outlier(0 ~ 1)
return : dataset that dropped outlier
"""
def del_outlier(x,p):
    clf=IsolationForest(contamination=float(p),
                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=2020, verbose=0)
    tem = x['취급액'].reset_index()
    clf.fit(tem)
    tem['anomaly'] = clf.predict(tem)
    return x[tem['anomaly']==1]


## clustering
"""
x : dataset
k : # of cluster
return : 기존 dataset에 kmeans라는 컬럼 추가한 df
"""
def km_clust(x, k, var=['취급액'],inner=3):

    Z = x[var].reset_index()
    km = KMeans(n_clusters=k,random_state=2020)
    km.fit(Z)
    Z['kmeans'] = km.labels_

    # inner cluster
    if inner!=0:
        object = Z[Z['kmeans']==1]
        K = Z[Z['kmeans']==1]['취급액'].reset_index()
        left = Z[Z['kmeans']!=1]
        km = KMeans(n_clusters=inner,random_state=2020)
        km.fit(K)
        object['labels'] = km.labels_
        object['kmeans'][object['labels']==0] = 1
        object['kmeans'][object['labels']==1] = 3
        object['kmeans'][object['labels']==2] = 4    
        object.drop(['labels'],axis=1,inplace=True) 
        object.set_index('index')
        left.set_index('index')
        res = pd.concat([object.set_index('index'),left.set_index('index')]).sort_index()

        return pd.concat([x,res['kmeans']],axis=1)
    else:
        return pd.concat([x,Z['kmeans']],axis=1)

def preprocess(perform,question,drop_rate,k,inner=False):
    data = pd.concat([perform,question])
    data.reset_index(inplace=True,drop=True)
    var = mk_var(data)
    data = var()
    train = data.iloc[:37372,:]
    test = data.iloc[37372:,:]
    test.reset_index(inplace=True,drop=True)
    train.reset_index(inplace=True,drop=True)

    train = train[train['취급액']!=0]
    train.reset_index(inplace=True,drop=True)

    #train = del_outlier(train,drop_rate)
    #train.reset_index(inplace=True,drop=True)

    sales_by_sid = train.groupby(['show_id','상품코드'])['취급액'].sum().reset_index(name='sales_by_sid')
    km_by_sid = km_clust(sales_by_sid,k,['sales_by_sid'],inner=inner)
    train = pd.merge(train,sales_by_sid,on=['show_id','상품코드'], how='left')

    train.rename(columns={'취급액':'sales'},inplace=True)
    test.rename(columns={'취급액':'sales'},inplace=True)

    #train = train.drop(['kmeans'],axis=1)

    return train, test, km_by_sid[['kmeans']], len(train)

def mk_statistics_var(train,test):
    stat_var = mk_stat_var(train,test)
    data = stat_var()

    #data.fillna(0, inplace=True) # test set 무형 data 변수
    return data


def mk_sid_df(data,train_len):

    data['istrain'] = 0
    data.iloc[:train_len,data.columns.get_loc('istrain')] = 1

    sid_icode = pd.DataFrame(data.groupby('show_id')['상품코드'].apply(lambda x: list(set(x)))).reset_index()

    sid_df = pd.DataFrame({
        col:np.repeat(sid_icode['show_id'].values,sid_icode['상품코드'].str.len())
        for col in sid_icode.columns.drop('상품코드')
    }).assign(**{'상품코드': np.concatenate(sid_icode['상품코드'].values)})

    sid_train = data[['show_id','istrain']].drop_duplicates('show_id')
    sid_df = pd.merge(sid_df,sid_train,on='show_id',how='left') # show_id, 상품코드 col 존재
        
    # raw_data의 변수 > sid_df에게 갖다 붙히기.
    # ㄴ 'hour_gr', 'id',
    #    'len_gr', 'min', 'min_gr',
    #    'rating',
    #    'sales', 'sales_by_sid', 'show_id', 'show_norm_order',
    #    'show_norm_order_gr', 'show_order', '노출(분)', '마더코드', '방송일시',
    #    '상품군', '상품명', '상품코드', '판매단가', 'day_sales_mean',
    #    'day_sales_std', 'day_sales_med', 'day_sales_rank', 'hour_sales_mean',
    #    'hour_sales_std', 'hour_sales_med', 'hour_sales_rank', 'min_sales_mean',
    #    'min_sales_std', 'min_sales_med', 'min_sales_rank', 'istrain'

    # 1. 상품코드 기준 단순 merge (=상품코드가 같으면 모두 같은 것들)

    item_code = data[['상품코드','마더코드','상품군','gender','pcode_freq','pay','set','special','판매단가',\
                'cate_sales_mean','cate_sales_std','cate_sales_med','cate_sales_rank','cate_price_mean',\
                'cate_price_std','cate_price_med','cate_price_rank','mcode_freq', 'mcode_freq_gr']]
    item_code = item_code.drop_duplicates('상품코드')

    sid_df = pd.merge(sid_df,item_code,on='상품코드',how='left') # 13618 row

    # 2. show_id 기준 단순 merge (= show_id가 같으면 모두 같은 것들)
    sid = data[['show_id','pcode_count','rating_byshow']]
    sid = sid.drop_duplicates('show_id')

    sid_df = pd.merge(sid_df,sid,on='show_id',how='left')

    # 3. length sum
    length = data.groupby(['show_id','상품코드'])['length'].sum()
    sid_df = pd.merge(sid_df,length,on=['show_id','상품코드'],how='left')

    # 4. show_id, 상품코드로 groupby 후 더 많은 것 택하기
    for col in ['day','holiday','hour','hour_prime','month','season']:
        dat = data.groupby(['show_id','상품코드'])[col].apply(lambda x: Counter(list(x)).most_common()[0][0])
        sid_df = pd.merge(sid_df, dat, on=['show_id','상품코드'], how='left')

    # 5. day, hour
    day = data[['day','day_sales_mean','day_sales_std', 'day_sales_med', 'day_sales_rank']]
    day.drop_duplicates('day', inplace = True)

    sid_df = pd.merge(sid_df, day, on='day', how='left')

    hour = data[['hour','hour_sales_mean','hour_sales_std', 'hour_sales_med', 'hour_sales_rank']]
    hour.drop_duplicates('hour', inplace =True)
    
    sid_df = pd.merge(sid_df, hour, on='hour', how='left')

    # 6. 학습될 값 붙이기
    sales_by_sid = data[['show_id','상품코드','sales_by_sid']]
    sales_by_sid.drop_duplicates(['show_id','상품코드'], inplace = True)

    sid_df =pd.merge(sid_df,sales_by_sid, on = ['show_id','상품코드'], how='left')
    
    return data[['show_id','상품코드','sales','show_norm_order','length']], sid_df
    

def mk_trainset(data,dummy = ['gender','pay','set','special','cate','day','hour']):
    """
    select feature to make train set 
    arg : data, dummy(list that make it one-hot-encoding)
    return : data
    """
    data['sales_per'] = np.log1p(data['판매단가'])
    data.rename(columns={'마더코드':'mcode','상품군':'cate','상품코드':'item_code'},inplace=True)
    # data['sales'] = np.log1p(data['sales'])
    encoder = LabelEncoder()
    encoder.fit(data['cate'])
    data['cate'] = encoder.transform(data['cate'])

    all_cate = ['gender','pay','set','special','cate','mcode_freq_gr','day','hour','month','season']
    left_cate = [x for x in all_cate if x not in dummy]

    if dummy:
        data = pd.get_dummies(data,columns=(dummy))

    if left_cate != []:
        for var in left_cate:
            data[var] = data[var].astype('category')

    #data['mcode'] = data['mcode'].astype('str').apply(lambda x: x[3:])
    #data['mcode'] = data['mcode'].astype(int)
    #data['item_code'] = data['item_code'].astype('str').apply(lambda x: x[2:])
    #data['item_code'] = data['item_code'].astype(int)
    data = data.drop(['판매단가','mcode'],axis=1)

    return data


def metric(real, pred):
    tem = np.abs((real - pred)/real)
    return tem.mean() * 100

def s_metric(real, pred):
    tem = np.abs(pred - real)/((np.abs(pred) + np.abs(real))/2)
    return tem.mean() * 100


def feature_impo(model,data):
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,data.columns)), columns=['Value','Feature'])
    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
    plt.savefig('lgbm_importances-01.png')


def scoring(sales,pred):
    if sales==0:
        score = s_metric(sales,pred)
        return score
    else:
        score = metric(sales,pred)
        return score


"""
def kmeans(data,k,var=['sales'],visual=False):

    train_km = pd.concat([data.iloc[:36250,:],y],axis=1)
    Z = train_km[var].reset_index()

    km = KMeans(n_clusters=k,random_state=2020)
    km.fit(Z)
    df = pd.DataFrame(Z)
    df['kmeans'] = km.labels_
    df = inner_cluster(df,3,1)
    y_km = df['kmeans']

    if visual:
        pca3 = PCA(n_components=3)
        data_pca3 = pca3.fit_transform(scaled_Z)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_pca3[:,0], data_pca3[:,1], data_pca3[:,2], c=df['kmeans'], s=60, edgecolors='white')
        ax.set_title('3D of Target distribution by diagnosis')
        ax.set_xlabel('pc1')
        ax.set_ylabel('pc2')
        ax.set_zlabel('pc3')
        plt.show()

    return y_km
"""