import numpy as np
from numpy.lib.utils import info
import pandas as pd
from datetime import datetime, timedelta
import copy
import re

from torch import triangular_solve

class mk_var():

    def __init__(self,data):
        self.data = data.copy()
        self.rating = pd.read_csv('data/2019_rating.csv',encoding='utf-8')
        
        # str to datetime
        self.data['방송일시'] = self.data['방송일시'].map(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M'))
        self.rating['시간대'] = self.rating['시간대'].map(lambda x: datetime.strptime(x,'%H:%M').time())
        
        self.data['length'] = copy.deepcopy(self.data['노출(분)']).replace(0,method='ffill')

        self.info = self.mk_addtional_info()

    def mk_addtional_info(self):

        def mk_show_id(): # 상품명 기준
            def preprocess(name):

                # 예외 처리
                
                name = name.replace('휴롬퀵스퀴저','휴롬 퀵스퀴저')
                name = name.replace('장수흙침대','장수 흙침대')
                name = name.replace('1+1 국내제조', '국내제조')

                if ('보국미니히터' in name) or ('우아미' in name) or ('갓바위' in name) or ('두씽' in name)\
                    or ('해뜰찬' in name) or ('법성포굴비' in name) or ('쥐치포' in name) or ('공간아트' in name)\
                    or ('르젠' in name) or ('쿠쿠' in name):
                    name = 'tmp'

                if ('삼성' in name) and ('도어' in name):
                    name = 'tmp'
    
                brands = ['프라다','구찌','버버리','코치','마이클코어스','톰포드', '페라가모', '생로랑', ]

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

            tmp = self.data['상품명'].map(preprocess)

            names = tmp.tolist()
            ids = [0]

            for i, name in enumerate(names[1:],1):
                prior = names[i-1]

                if prior == name:
                    ids.append(ids[-1])
                else:
                    ids.append(ids[-1]+1)

            return np.array(ids)
        
        def mk_rating():

            time_idx = dict(zip(self.rating.시간대,self.rating.index)) 

            def cal_ratio(date,length):
               
                if (date.date().year == 2020):
                   return 0

                else:
                    rating_for_day = self.rating.loc[:,['시간대',str(date.date())]]
            
                    start_idx = time_idx[date.time()]
                    end = date + timedelta(minutes=length)
                    end_idx = time_idx[end.time()]

                return self.rating.loc[start_idx:end_idx, str(date.date())].mean()

            return self.data.apply(lambda x: cal_ratio(x['방송일시'],x['length']),axis=1)

        def mk_rating_byshow(info):

            if 'rating' in info.columns:
                pass
            else:
                self.data['rating'] = mk_rating()

            show_rating = dict(info.groupby('show_id')['rating'].mean())
                
            return info['show_id'].map(lambda x: show_rating[x])
            
        info = pd.DataFrame()
        info['show_id'] = mk_show_id()
        info['rating'] = mk_rating()
        info['rating_byshow'] = mk_rating_byshow(info)

        return info

    def mk_datetime_var(self):

        def mk_month():
            return self.data['방송일시'].map(lambda x: x.month)

        def mk_season():

            def month_grouping(month):
                if 3<= month <6: # 봄
                    return 0

                elif 6<= month <9: # 여름
                    return 1

                elif 9<= month <12: # 가을
                    return 2

                else: # 겨울
                    return 3

            return self.data['방송일시'].map(lambda x: month_grouping(x.month))

        def mk_day():
            return self.data['방송일시'].map(lambda x: x.weekday())

        def mk_holiday():

            ko_holiday = ['2019-01-01', '2019-02-04', '2019-02-05', '2019-02-06','2019-03-01', '2019-05-06', '2019-06-06', '2019-08-15','2019-09-13', '2019-09-12', '2019-10-03', '2019-10-09', '2019-12-25']

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

        def mk_hour_group():

            def hour_grouping(hour): # 주문량 기준으로 인간지능 그룹핑
                if 1 <= hour < 7: # 1, 2, 6
                    return 0
                elif 7 <= hour < 9:
                    return 1
                elif 9 <= hour < 15:
                    return 2
                elif 15 <= hour < 19:
                    return 3
                else:
                    return 4
            
            return self.data['방송일시'].map(lambda x: hour_grouping(x.hour))

        def mk_hour_prime():
            # prime 시간대 - 오후 9시,10시,11시 + 주말 오후 6시,7시,8시 + 일요일 오전 10시,11시
            def hour_prime(day,hour):
            
                if 21 <= hour <= 23:
                    return 1

                elif (5 <= day <=6) and (18 <= hour <= 20):
                    return 1

                elif (5 <= day <= 6) and (10<= hour <= 11):
                    return 1

                else:
                    return 0

            return self.data['방송일시'].map(lambda x: hour_prime(x.weekday(),x.hour))   
        
        def mk_min():
            return self.data['방송일시'].map(lambda x: x.minute)

        def mk_min_group():

            def min_grouping(min):
                if min <20:
                    return 0
                elif min == 20:
                    return 1
                else:
                    return 2

            return self.data['방송일시'].map(lambda x: min_grouping(x.minute))
        
        self.data['month'] = mk_month()
        self.data['season'] = mk_season()
        self.data['day'] = mk_day()
        self.data['holiday'] = mk_holiday()
        self.data['hour'] = mk_hour()
        self.data['hour_gr'] = mk_hour_group()
        self.data['hour_prime'] = mk_hour_prime()
        self.data['min'] = mk_min()
        self.data['min_gr'] = mk_min_group()

        return self.data

    def mk_length_var(self):

        def mk_length_grouping():

            def length_grouping(length):
                if length < 20:
                    return 0
                elif length == 20:
                    return 1
                else:
                    return 2

            return self.data['length'].map(length_grouping)

        self.data['len_gr'] = mk_length_grouping()

        return self.data

    def mk_mcode_var(self):

        def mk_mcode_freq():
            self.info['mcode'] = copy.deepcopy(self.data['마더코드'])
            tmp = self.info.groupby('show_id')['mcode'].apply(lambda x: list(set(x))).reset_index(name='mcode')
            mcode_freq = dict(pd.Series(sum([mcode for mcode in tmp.mcode],[])).value_counts())

            return self.data['마더코드'].map(lambda x: mcode_freq[x])

        def mk_mcode_freq_grouping():

            def freq_grouping(freq):
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

            return self.data['mcode_freq'].map(freq_grouping)
        
        self.data['mcode_freq'] = mk_mcode_freq()
        self.data['mcode_freq_gr'] = mk_mcode_freq_grouping()

        return self.data

    def mk_pcode_var(self):

        def mk_pcode_freq():
            self.info['pcode'] = copy.deepcopy(self.data['상품코드'])
            tmp = self.info.groupby('show_id')['pcode'].apply(lambda x: list(set(x))).reset_index(name='pcode')
            pcode_freq = dict(pd.Series(sum([pcode for pcode in tmp.pcode],[])).value_counts())

            return self.data['상품코드'].map(lambda x: pcode_freq[x])

        def mk_pcode_count():
            tmp = self.info.groupby('show_id')['pcode'].apply(lambda x: len(set(x)))
            pcode_count = dict(tmp)

            return self.info['show_id'].map(lambda x: pcode_count[x])

        self.data['pcode_freq'] = mk_pcode_freq()
        self.data['pcode_count'] = mk_pcode_count()

        return self.data

    def mk_rating_var(self):

        def mk_rating():
            return copy.deepcopy(self.info['rating'])

        def mk_rating_byshow():
            return copy.deepcopy(self.info['rating_byshow'])

        self.data['rating'] = mk_rating()
        self.data['rating_byshow'] = mk_rating_byshow()

        return self.data

    def mk_pname_var(self):

        def mk_gender():
            def check_gender(pname):
                if ('여성' in pname) or ('여자' in pname):
                    return 0
                elif '남성' in pname:
                    return 1
                else:
                    return 2
                
            return self.data['상품명'].map(check_gender)

        def mk_pay():
            def check_pay(pname):
                if ('일시불' in pname) or ('일)' in pname): # (일), 일) 모두 커버
                    return 0
                elif ('무이자' in pname) or ('무)' in pname):
                    return 1
                else:
                    return 2

            return self.data['상품명'].map(check_pay)

        def mk_set():
            def check_set(pname):

                regex = re.compile('\d+(?![단|도|년|분|구|인|형|등])[가-힣]')
                regex_2 = re.compile('\d박스')

                if ('세트' in pname) or ('SET' in pname) or ('패키지' in pname) or ('+' in pname) \
                   or ('&' in pname) or (regex.search(pname)) or (regex_2.search(pname)): 
                    return 1

                else:
                    return 0

            return self.data['상품명'].map(check_set)
        
        def mk_special():
            def check_special(pname):
                if ('할인'in pname ) or ('스페셜' in pname) or ('초특가' in pname) or ('단하루' in pname) or ('파격찬스' in pname) or ('최저가' in pname):
                    return 1
                else:
                    return 0

            return self.data['상품명'].map(check_special)

        self.data['gender'] = mk_gender()
        self.data['pay'] = mk_pay()
        self.data['set'] = mk_set()
        self.data['special'] = mk_special()

        return self.data

    def mk_order_var(self):

        self.info['방송일시'] = copy.deepcopy(self.data['방송일시'])

        def mk_order():
            id_date = self.info.groupby('show_id')['방송일시'].apply(lambda x: sorted(list(set(x)))).reset_index(name='방송일시')
            id_date = id_date['방송일시'].map(lambda x: {time:i for i, time in enumerate(x,1)})

            date_order = {time:dic[time] for dic in id_date for time in dic}

            return self.data['방송일시'].map(lambda x: date_order[x])

        def mk_norm_order():
            id_date = self.info.groupby('show_id')['방송일시'].apply(lambda x: sorted(list(set(x)))).reset_index(name='방송일시')
            id_date = id_date['방송일시'].map(lambda x: {time:i/len(x) for i, time in enumerate(x,1)})

            date_order = {time:dic[time] for dic in id_date for time in dic}

            return self.data['방송일시'].map(lambda x: date_order[x])

        def mk_norm_order_gr():
            
            def order_grouping(order):

                # 3,4,5,6,8등분 존재

                if order < 0.34:
                    return 0

                elif 0.34 < order <= 0.8:
                    return 1

                else:
                    return 2

            return self.data['show_norm_order'].map(order_grouping)

        self.data['show_order'] = mk_order()
        self.data['show_norm_order'] = mk_norm_order()
        self.data['show_norm_order_gr'] = mk_norm_order_gr()

        return self.data

    def __call__(self):

        self.data = self.mk_datetime_var()
        self.data = self.mk_length_var()
        self.data = self.mk_mcode_var()
        self.data = self.mk_pcode_var()
        self.data = self.mk_rating_var()
        self.data = self.mk_pname_var()
        self.data = self.mk_order_var()

        self.data.fillna(0,inplace=True)

        return self.data

# ~ 별 평균, 분산, 비율, 중위수, rank
class mk_stat_var():
    def __init__(self,train,test):
        self.train = train.copy() # 34317
        self.test = test.copy() # 2891
        self.data = pd.concat([train,test]).reset_index(drop=True) # 37208

    def mk_cate_var(self):

        sales = self.train.groupby('상품군')['sales'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'cate_sales_mean', 'std':'cate_sales_std', '50%':'cate_sales_med'},inplace=True)
        sales['cate_sales_rank'] = sales['cate_sales_mean'].rank(ascending=False,method='dense')

        price = self.data.groupby('상품군')['판매단가'].describe()[['mean','std','50%']]
        price.rename(columns={'mean':'cate_price_mean', 'std':'cate_price_std', '50%':'cate_price_med'},inplace=True)
        price['cate_price_rank'] = price['cate_price_mean'].rank(ascending=False,method='dense')

        self.data = pd.merge(self.data,sales,on='상품군',how='left')
        self.data = pd.merge(self.data,price,on='상품군',how='left')
        
        return self.data


    def mk_day_var(self):

        sales = self.train.groupby('day')['sales'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'day_sales_mean', 'std':'day_sales_std', '50%':'day_sales_med'},inplace=True)
        sales['day_sales_rank'] = sales['day_sales_mean'].rank(ascending=False,method='dense')

        return pd.merge(self.data,sales,on='day', how='left')

    def mk_hour_var(self):

        sales = self.train.groupby('hour')['sales'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'hour_sales_mean', 'std':'hour_sales_std', '50%':'hour_sales_med'},inplace=True)
        sales['hour_sales_rank'] = sales['hour_sales_mean'].rank(ascending=False,method='dense')

        return pd.merge(self.data,sales,on='hour',how='left')

    
    def mk_min_var(self):

        sales = self.train.groupby('min')['sales'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'min_sales_mean', 'std':'min_sales_std', '50%':'min_sales_med'},inplace=True)
        sales['min_sales_rank'] = sales['min_sales_mean'].rank(ascending=False,method='dense')

        return pd.merge(self.data, sales, on='min', how='left')
    

    def mk_mcode_var(self):

        sales = self.train.groupby('마더코드')['sales'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'mcode_sales_mean', 'std':'mcode_sales_std', '50%':'mcode_sales_med'},inplace=True)
        sales['mcode_sales_rank'] = sales['mcode_sales_mean'].rank(ascending=False,method='dense')

        return pd.merge(self.data, sales, on='마더코드', how='left')

    def mk_pcode_var(self):

        sales = self.train.groupby('상품코드')['sales'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'pcode_sales_mean', 'std':'pcode_sales_std', '50%':'pcode_sales_med'},inplace=True)
        sales['pcode_sales_rank'] = sales['pcode_sales_mean'].rank(ascending=False,method='dense')

        return pd.merge(self.data, sales, on='상품코드', how='left')

    
    def mk_order_var(self):

        sales = self.train.groupby('show_norm_order_gr')['sales'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'order_sales_mean', 'std':'order_sales_std', '50%':'order_sales_med'},inplace=True)
        sales['order_sales_rank'] = sales['order_sales_mean'].rank(ascending=False,method='dense')

        return pd.merge(self.data,sales, on='show_norm_order_gr', how='left')

    def mk_price_var(self):

        def mk_price_gr(price):

            if price <= 39900:
                return 0

            elif 40000 <= price <=69900:
                return 1

            elif 70000 <= price <= 99900:
                return 2

            elif 100000 <= price <= 149900:
                return 3

            elif 150000 <= price <= 299900:
                return 4

            else:
                return 5

        self.train['price_gr'] = self.train['판매단가'].map(mk_price_gr)

        sales = self.train.groupby('price_gr')['sales'].describe()[['mean','std','50%']]
        sales.rename(columns={'mean':'pr_sales_mean', 'std':'pr_sales_std', '50%':'pr_sales_med'},inplace=True)
        sales['pr_sales_rank'] = sales['pr_sales_mean'].rank(ascending=False,method='dense')

        self.data['price_gr'] = self.data['판매단가'].map(mk_price_gr)

        results = pd.merge(self.data,sales, on='price_gr', how='left')

        del self.train['price_gr']
        del self.data['price_gr']

        return results

    def __call__(self):
        self.data = self.mk_cate_var()
        self.data = self.mk_day_var()
        self.data = self.mk_hour_var()
        self.data = self.mk_min_var()
        self.data = self.mk_mcode_var()
        self.data = self.mk_pcode_var()
        self.data = self.mk_order_var()
        self.data = self.mk_price_var()

        self.data.fillna(0,inplace=True)

        return self.data
