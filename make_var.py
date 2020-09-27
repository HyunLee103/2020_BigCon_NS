import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re


def mk_sid(data): # 상품명 기준
    def preprocess(name):
        # 예외 처리

        if ('보국미니히터' in name) or ('우아미' in name) or ('갓바위' in name) or ('두씽' in name)\
            or ('해뜰찬' in name) or ('법성포굴비' in name) or ('쥐치포' in name) or ('공간아트' in name)\
            or ('르젠' in name) or ('쿠쿠' in name):
            name = 'tmp'

        if ('삼성' in name) and ('도어' in name):
            name = 'tmp'

        brands = ['프라다','구찌','버버리','코치','마이클코어스','톰포드', '페라가모', '생로랑', ]

        for brand in brands:
            name = name.replace(brand,'명품')

        name = name.replace('휴롬퀵스퀴저','휴롬 퀵스퀴저')
        name = name.replace('장수흙침대','장수 흙침대')
        name = name.replace('1+1 국내제조', '국내제조')

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

    tmp = data['상품명'].map(preprocess)

    names = tmp.tolist()
    ids = [0]

    for i, name in enumerate(names[1:],1):
        prior = names[i-1]

        if prior == name:
            ids.append(ids[-1])
        else:
            ids.append(ids[-1]+1)

    return np.array(ids)

class mk_var():

    def __init__(self,perform_raw,test_raw,rating):
        self.train = perform_raw.copy()
        self.test = test_raw.copy()
        self.rating = rating.copy()

        # preprocess #

        # str to datetime
        self.train['방송일시'] = self.train['방송일시'].map(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M'))
        self.test['방송일시'] = self.test['방송일시'].map(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M'))
        
        self.rating['시간대'] = self.rating['시간대'].map(lambda x: datetime.strptime(x,'%H:%M').time())
        self.rating= self.rating.set_index('시간대')
        self.rating.columns = pd.to_datetime(self.rating.columns,format='%Y-%m-%d')
        self.rating.columns = self.rating.columns.date
        self.rating.reset_index(inplace=True)
        

        # 노출(분) 수정 > 덮어쓰기
        self.train['노출(분)'] = self.train['노출(분)'].replace(0, method='ffill')
        self.test['노출(분)'] = self.test['노출(분)'].replace(0, method='ffill')

        # make istrain col
        self.train['istrain'] = 1
        self.test['istrain'] = 0

        self.data = pd.concat([self.train,self.test])
        self.data['show_id'] = mk_sid(self.data)


    def mk_rating(self):

        # train - 해당 시간대 시청률 사용
        # test - 요일,시간 groupby

        tr = self.train[['방송일시','노출(분)']].copy()
        te = self.test[['방송일시','노출(분)']].copy()

        time_idx = dict(zip(self.rating.시간대,self.rating.index))

        # 1. train
        def cal_ratio(date,length):

            if (date.date().year == 2020):
                return 0 # 마지막 방송이 20년 1월 1일 자정 넘김

            else:
                dayrating = self.rating.loc[:,['시간대',date.date()]]

                start_idx = time_idx[date.time()]
                end = date + timedelta(minutes=length)
                end_idx = time_idx[end.time()]

                return self.rating.loc[start_idx:end_idx, date.date()].mean()

        tr['rating'] = tr.apply(lambda x: cal_ratio(x['방송일시'],x['노출(분)']),axis=1)

        # 2. test

        rating_tmp = self.rating.copy().set_index('시간대').transpose()
        rating_tmp = rating_tmp.rename_axis('방송일시').reset_index()

        rating_tmp['방송요일'] = rating_tmp['방송일시'].map(lambda x: x.weekday())

        del rating_tmp['방송일시']

        day_time = rating_tmp.groupby('방송요일').apply(np.mean).iloc[:,:-1]
        
        day_time = day_time.transpose()
        day_time.reset_index(inplace=True)

        time_idx = dict(zip(day_time.시간대,day_time.index))
        
        def cal_ratio(date,length):

            dayrating = day_time.loc[:,date.weekday()]
            
            start_idx = time_idx[date.time()]
            end = date + timedelta(minutes=length)
            end_idx = time_idx[end.time()] 

            return dayrating[start_idx:end_idx].mean()

        te['rating'] = te.apply(lambda x: cal_ratio(x['방송일시'],x['노출(분)']),axis=1)

        dat = pd.concat([tr,te])

        return dat['rating']

    def make_rating_var(self):

        sid_rating = self.data[['show_id','rating']]
        sid_rating = sid_rating.groupby('show_id')['rating'].mean().reset_index()
        sid_rating.rename(columns = {'rating':'sid_rating'}, inplace=True)

        return pd.merge(self.data, sid_rating, on='show_id',how='left')

    # 1. 방송일시) month, day, holiday,
    def make_datetime_var(self):
        def mk_day():
            return self.data['방송일시'].map(lambda x: x.weekday())

        def mk_holiday():
            ko_holiday = ['2019-01-01', '2019-02-04', '2019-02-05', '2019-02-06','2019-03-01', '2019-05-06', '2019-06-06', '2019-08-15','2019-09-13', '2019-09-12', '2019-10-03', '2019-10-09', '2019-12-25', '2020-06-06']

            def check_holiday(date):
                if date.weekday() > 4:
                    return 1

                elif str(date.date()) in ko_holiday:
                    return 1

                else:
                    return 0

            return self.data['방송일시'].map(check_holiday)

        def mk_hour():
            return self.data['방송일시'].map(lambda x: x.hour)

        # def mk_hour_group():

        def mk_prime():

            def check_prime(day,hour):
                if 21<= hour <= 23:
                    return 1

                elif (5 <= day <=6) and (18 <= hour <= 20):
                    return 1

                elif (5 <= day <= 6) and (10<= hour <= 11):
                    return 1

                else:
                    return 0

            return self.data['방송일시'].map(lambda x: check_prime(x.weekday(),x.hour))

        def mk_min():
            return self.data['방송일시'].map(lambda x: x.minute)
            
        self.data['day'] = mk_day()
        self.data['holiday'] = mk_holiday()
        self.data['hour'] = mk_hour()
        self.data['prime'] = mk_prime()
        self.data['min'] = mk_min()
    
        return self.data

    def make_mcode_var(self):

        def mk_mcode_freq():
            info = self.data[self.data['istrain']==1][['show_id','마더코드']]
            tmp = info.groupby('show_id')['마더코드'].apply(lambda x: list(set(x))).reset_index(name='mcode')

            mcode_freq = dict(pd.Series(sum([mcode for mcode in tmp.mcode],[])).value_counts())

            return self.data['마더코드'].map(lambda x: mcode_freq.get(x))

        def mk_mcode_freq_gr():

            def check_freq(freq):
                if freq <= 5:
                    return 0 # 4225개

                elif 5< freq <= 17:
                    return 1 # 6157개

                elif 17< freq <= 30:
                    return 2 # 6197개
                
                elif 30< freq <= 51:
                    return 3 # 7334개

                elif 51< freq <= 130 :
                    return 4 # 6469개
    
                else:
                    return 5 # 6990개

            return self.data['mcode_freq'].map(check_freq)

        self.data['mcode_freq'] = mk_mcode_freq()
        self.data['mcode_freq_gr'] = mk_mcode_freq_gr()

        return self.data

    def make_icode_var(self):

        def mk_icode_count():
            tmp = self.data.groupby('show_id')['상품코드'].apply(lambda x: len(set(x)))
            icode_count = dict(tmp)

            return self.data['show_id'].map(lambda x: icode_count[x])

        self.data['icode_count'] = mk_icode_count()

        return self.data

    def make_iname_var(self):

        def mk_gender():
            def check_gender(iname):
                if ('여성' in iname) or ('여자' in iname):
                    return 0
                elif '남성' in iname:
                    return 1
                else:
                    return 2
                
            return self.data['상품명'].map(check_gender)

        def mk_pay():
            def check_pay(iname):
                if ('일시불' in iname) or ('일)' in iname): # (일), 일) 모두 커버
                    return 0
                elif ('무이자' in iname) or ('무)' in iname):
                    return 1
                else:
                    return 2

            return self.data['상품명'].map(check_pay)

        def mk_set():
            def check_set(iname):
                regex = re.compile('\d+(?![단|도|년|분|구|인|형|등])[가-힣]')
                regex_2 = re.compile('\d박스')

                if ('세트' in iname) or ('SET' in iname) or ('패키지' in iname) or ('+' in iname) \
                   or ('&' in iname) or (regex.search(iname)) or (regex_2.search(iname)): 
                    return 1

                else:
                    return 0

            return self.data['상품명'].map(check_set)

        self.data['gender'] = mk_gender()
        self.data['pay'] = mk_pay()
        self.data['set'] = mk_set()

        return self.data

    def make_order_var(self):

        def mk_order():
            sid_date = self.data.groupby('show_id')['방송일시'].apply(lambda x: sorted(list(set(x)))).reset_index(name='방송일시')
            sid_date = sid_date['방송일시'].map(lambda x: {time:i for i, time in enumerate(x,1)})

            date_order = {time:dic[time] for dic in sid_date for time in dic}

            return self.data['방송일시'].map(lambda x: date_order[x])

        def mk_normorder():
            sid_date = self.data.groupby('show_id')['방송일시'].apply(lambda x: sorted(list(set(x)))).reset_index(name='방송일시')
            sid_date = sid_date['방송일시'].map(lambda x: {time:i/len(x) for i, time in enumerate(x,1)})

            date_order = {time:dic[time] for dic in sid_date for time in dic}

            return self.data['방송일시'].map(lambda x: date_order[x])

        self.data['s_order'] = mk_order()
        self.data['s_normorder'] = mk_normorder()

        return self.data


    def make_cate_stat(self):
        sales = self.train.groupby('상품군')['취급액'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'cate_sales_mean', 'std':'cate_sales_std', '50%':'cate_sales_med'},inplace=True)
        sales['cate_sales_rank'] = sales['cate_sales_mean'].rank(ascending=False,method='dense')

        sales.fillna(0,inplace=True)

        return pd.merge(self.data,sales,on='상품군', how='left')

    def make_day_stat(self):
        sales = self.data[self.data['istrain']==1].groupby('day')['취급액'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'day_sales_mean', 'std':'day_sales_std', '50%':'day_sales_med'},inplace=True)
        sales['day_sales_rank'] = sales['day_sales_mean'].rank(ascending=False,method='dense')

        sales.fillna(0,inplace=True)

        return pd.merge(self.data,sales,on='day',how='left')

    def make_hour_stat(self):
        sales = self.data[self.data['istrain']==1].groupby('hour')['취급액'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'hour_sales_mean', 'std':'hour_sales_std', '50%':'hour_sales_med'},inplace=True)
        sales['hour_sales_rank'] = sales['hour_sales_mean'].rank(ascending=False,method='dense')

        sales.fillna(0,inplace=True)

        return pd.merge(self.data,sales,on='hour',how='left')

    def make_min_stat(self):

        sales = self.data[self.data['istrain']==1].groupby('min')['취급액'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'min_sales_mean', 'std':'min_sales_std', '50%':'min_sales_med'},inplace=True)
        sales['min_sales_rank'] = sales['min_sales_mean'].rank(ascending=False,method='dense')

        sales.fillna(0,inplace=True)

        return pd.merge(self.data,sales,on='min',how='left')

    def make_mcode_stat(self):

        sales = self.train.groupby('마더코드')['취급액'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'mcode_sales_mean', 'std':'mcode_sales_std', '50%':'mcode_sales_med'},inplace=True)
        sales['mcode_sales_rank'] = sales['mcode_sales_mean'].rank(ascending=False,method='dense')

        sales.fillna(0,inplace=True)

        return pd.merge(self.data,sales,on='마더코드',how='left')

    def make_order_stat(self):

        tmp = self.data[self.data['istrain']==1]
        tmp['주문량'] = tmp.apply(lambda x: x['취급액']/x['판매단가'], axis=1)

        sales = tmp.groupby('상품군')['주문량'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'cate_order_mean', 'std':'cate_order_std', '50%':'cate_order_med'},inplace=True)
        sales['cate_order_rank'] = sales['cate_order_mean'].rank(ascending=False,method='dense')
        
        sales.fillna(0,inplace=True)
    
        self.data = pd.merge(self.data,sales,on='상품군',how='left')

        sales = tmp.groupby('day')['주문량'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'day_order_mean', 'std':'day_order_std', '50%':'day_order_med'},inplace=True)
        sales['day_order_rank'] = sales['day_order_mean'].rank(ascending=False,method='dense')
        
        sales.fillna(0,inplace=True)
    
        self.data = pd.merge(self.data,sales,on='day',how='left')

        sales = tmp.groupby('hour')['주문량'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'hour_order_mean', 'std':'hour_order_std', '50%':'hour_order_med'},inplace=True)
        sales['hour_order_rank'] = sales['hour_order_mean'].rank(ascending=False,method='dense')
        
        sales.fillna(0,inplace=True)

        self.data = pd.merge(self.data,sales,on='hour',how='left')

        sales = tmp.groupby('min')['주문량'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'min_order_mean', 'std':'min_order_std', '50%':'min_order_med'},inplace=True)
        sales['min_order_rank'] = sales['min_order_mean'].rank(ascending=False,method='dense')
        
        sales.fillna(0,inplace=True)
    
        self.data = pd.merge(self.data,sales,on='min',how='left')

        sales = tmp.groupby('마더코드')['주문량'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'mcode_order_mean', 'std':'mcode_order_std', '50%':'mcode_order_med'},inplace=True)
        sales['mcode_order_rank'] = sales['mcode_order_mean'].rank(ascending=False,method='dense')
        
        sales.fillna(0,inplace=True)
    
        self.data = pd.merge(self.data,sales,on='마더코드',how='left')

        return self.data

    def make_code_to_var(self):

        self.data['마더코드'] = self.data['마더코드'].astype(int)
        self.data['마더코드'] = self.data['마더코드'].astype('str').apply(lambda x: x[3:])
        self.data['마더코드'] = self.data['마더코드'].astype(int)

        self.data['상품코드'] = self.data['상품코드'].astype(int)
        self.data['상품코드'] = self.data['상품코드'].astype('str').apply(lambda x: x[2:])
        self.data['상품코드'] = self.data['상품코드'].astype(int)

        return self.data

    def make_salespower(self):

        self.data['salespower'] = self.data.apply(lambda x: x['s_normorder'] * x['노출(분)'], axis=1)

        return self.data
    
    def __call__(self):

        self.data['rating'] = self.mk_rating()
        self.data['price_log'] = self.data['판매단가']

        self.data = self.make_rating_var()

        self.data = self.make_datetime_var()
        self.data = self.make_mcode_var()
        self.data = self.make_icode_var()
        self.data = self.make_iname_var()
        self.data = self.make_order_var()
        
        self.data = self.make_cate_stat()
        self.data = self.make_day_stat()
        self.data = self.make_min_stat()
        self.data = self.make_mcode_stat()

        self.data = self.make_order_stat()

        self.data = self.make_code_to_var()

        self.data = self.make_salespower()

        self.data.rename(columns={'노출(분)':'length','취급액':'sales','상품군':'cate','마더코드':'mcode', '상품코드':'icode'},inplace=True)

        tr = self.data[self.data['istrain']==1]
        te = self.data[self.data['istrain']==0]

        del tr['istrain']
        del te['istrain']

        del tr['show_id']
        del te['show_id']

        return tr, te

def make_variable(perform_raw,test_raw,rating):
    
    make_var = mk_var(perform_raw,test_raw,rating)

    return make_var()
