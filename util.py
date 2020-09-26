#-*-coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.reshape.reshape import get_dummies
import seaborn as sns
plt.rc('font', family='NanumBarunGothic')
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.decomposition import PCA
import pandas as pd



## load data
def load_data(data_path,weather=False,trend=False,query=False):
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
    perform_raw['is_mcode'] = 1

    test.reset_index(inplace=True)
    test.rename(columns={'index':'id'},inplace=True)
    test['id'] = test['id'] + 37372

    train_mcode = set(perform_raw['마더코드'])
    test_mcode = set(test['마더코드'])
    df = pd.DataFrame(columns=['is_mcode'])
    test = pd.concat([test,df])
    test['is_mcode'][test['마더코드'].isin(list(train_mcode & test_mcode))] = 1
    test.fillna(0,inplace=True)

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
    tem = x['sales'].reset_index()
    clf.fit(tem)
    tem['anomaly'] = clf.predict(tem)
    return x[tem['anomaly']==1]


## clustering
"""
x : dataset
k : # of cluster
return : 기존 dataset에 kmeans라는 컬럼 추가한 df
"""
def km_clust(x, k, var=['sales'],inner=3):
    # mean = x[var].mean()
    # std = x[var].std()
    # x[var] = (x[var] - x[var].mean())/x[var].std()
    Z = x[var].reset_index()
    km = KMeans(n_clusters=k,random_state=2020)
    km.fit(Z)
    Z['kmeans'] = km.labels_

    # inner cluster
    if inner!=0:
        object = Z[Z['kmeans']==1]
        K = Z[Z['kmeans']==1]['sales'].reset_index()
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

    perform.reset_index(inplace=True,drop=True)
    question.reset_index(inplace=True,drop=True)

    train = perform[perform['sales']!=0]
    train.reset_index(inplace=True,drop=True)

    train = del_outlier(train,drop_rate)
    train.reset_index(inplace=True,drop=True)

    train = km_clust(train,k,inner=inner)
    print(train['kmeans'].value_counts())

    y_km = train[['kmeans']]
    train = train.drop(['kmeans'],axis=1)

    raw_data = pd.concat([train,question]).reset_index(drop=True)

    return raw_data, y_km, len(train)




def mk_trainset(data,dummy = ['gender','pay'],categorical=True):
    """
    select feature to make train set 
    arg : data, dummy(list that make it one-hot-encoding),catetgorical(True for lgbm, False for other models)
    return : data
    """

    encoder = LabelEncoder()
    encoder.fit(data['cate'])
    data['cate'] = encoder.transform(data['cate'])
    all_cate = ['day','hour','min','mcode_freq_gr','s_order','gender','pay','cate']
    left_cate = [x for x in all_cate if x not in dummy]

    if categorical:
        pass
    else:
        dummy = all_cate
        left_cate = []

    if dummy:
        data = pd.get_dummies(data,columns=(dummy))

    if left_cate != []:
        for var in left_cate:
            data[var] = data[var].astype('category')

    data = data.drop(['방송일시','상품명','판매단가'],axis=1)

    return data


def metric(real, pred):
    tem = np.abs((real - pred)/real)
    return tem.mean() * 100


def feature_impo(model,data):
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,data.columns)), columns=['Value','Feature'])
    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
    plt.savefig('lgbm_importances-01.png')






