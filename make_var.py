import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import copy

class mk_var():

    def __init__(self):
        self.data = pd.read_csv('data/2019_performance.csv', encoding='utf-8')
        self.rating = pd.read_csv('data/2019_rating.csv',encoding='utf-8')
        
        # str to datetime
        self.data['방송일시'] = self.data['방송일시'].map(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M'))
        self.rating['시간대'] = self.rating['시간대'].map(lambda x: datetime.strptime(x,'%H:%M').time())
        
        self.data['length'] = copy.deepcopy(self.data['노출(분)']).replace(0,method='ffill')

        self.info = self.mk_addtional_info()

    def mk_addtional_info(self):
        
        def mk_order():
            return round(self.data['취급액']/self.data['판매단가'])

        def mk_show(): # 상품명 기준
            def preprocess(name):
                name = name.replace('무이자','')
                name = name.replace('일시불','')
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
        info['order'] = mk_order()
        info['show_id'] = mk_show()
        info['rating'] = mk_rating()
        info['rating_byshow'] = mk_rating_byshow(info)

        return info

    def mk_datetime_var(self):

        def mk_day():
            return self.data['방송일시'].map(lambda x: x.weekday())

        def mk_holiday():
            return self.data['방송일시'].map(lambda x: 1 if x.weekday() > 4 else 0)

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
        
        self.data['day'] = mk_day()
        self.data['holiday'] = mk_holiday()
        self.data['hour'] = mk_hour()
        self.data['hour_gr'] = mk_hour_group()
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

    def __call__(self):

        self.data = self.mk_datetime_var()
        self.data = self.mk_length_var()
        self.data = self.mk_mcode_var()
        self.data = self.mk_pcode_var()
        self.data = self.mk_rating_var()

        return self.data

var = mk_var()
data = var()

