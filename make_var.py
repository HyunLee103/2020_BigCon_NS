import numpy as np
import pandas as pd
from datetime import datetime

# 추가 정보 데이터프레임 따로 생성. performance(data) 데이터프레임과 인덱스로 매칭
class mk_additional_info():
    '''
    input: perform, ratio
    return: (__call__) additional info dataframe (주문량,방송ID,시청률, )
    '''

    def __init__(self,data,ratio):
        self.data = data 
        self.ratio = ratio
        self.info = pd.DataFrame()

    def mk_order(self):
        return round(self.data['취급액']/self.data['판매단가'])

    def mk_showid(self):
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

    '''
    def mk_ratio(self):
        pass
    '''

    def __call__(self): # 모든 함수 실행

        self.info['주문량'] = self.mk_order()
        self.info['방송ID'] = self.mk_showid()
        #self.info['시청률'] = self.mk_ratio()

        return self.info
 

class mk_datetime_var():
    
    '''
    input: perform
    return: (__call__) 요일, 휴일 여부, 방송시간-시, 방송시간-분
    '''

    def __init__(self,data):
        # str to datetime
        self.data = data

    def str_to_datetime(self):
        return self.data['방송일시'].map(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M'))
        
    def mk_weekday(self): # 요일
        return self.data['방송일시'].map(lambda x: x.weekday())

    def mk_holiday(self): # 휴일 여부
        return self.data['방송일시'].map(lambda x: 1 if x.weekday() >4 else 0)

    def mk_hour(self,grouping=False): # 방송 시간 - 시

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
        
        if grouping:
            return self.data['방송일시'].map(lambda x: hour_grouping(x.hour))
        
        else:
            return self.data['방송일시'].map(lambda x: x.hour)

    def mk_min(self,grouping=False): # 방송 시간 - 분
        
        def min_grouping(min): 
            return min # 그룹핑 기준 정해야

        if grouping:
            return self.data['방송일시'].map(lambda x: min_grouping(x.minute))
        
        else:
            return self.data['방송일시'].map(lambda x: x.minute)

    def __call__(self):

        self.data['방송일시'] = self.str_to_datetime()
        self.data['방송요일'] = self.mk_weekday()
        self.data['휴일'] = self.mk_holiday()
        self.data['방송시각'] = self.mk_hour(grouping=True)
        self.data['방송분'] = self.mk_min(grouping=True)

        return self.data


class mk_showtime_var():
    
    '''
    input: perform
    return: (__call__) 노출(분)_그룹
    '''

    def __init__(self,data):
        self.data = data
        self.data['노출(분)_수정'] = self.data['노출(분)'].replace(0,method='ffill')

    def grouping(self):

        def showtime_grouping(showtime): # '노출(분) > 24개'
            if showtime < 20:
                return 0
            elif showtime == 20:
                return 1
            else:
                return 2

        return self.data['노출(분)_수정'].map(showtime_grouping)


    def __call__(self):
        self.data['노출(분)_그룹'] = self.grouping()

        del self.data['노출(분)_수정']

        return self.data

data = pd.read_csv('data/2019_performance.csv', encoding='utf-8') # 실제론 test와 concat해야
ratio = pd.read_csv('data/2019_rating.csv',encoding='utf-8')

additional_info = mk_additional_info(data,ratio)
info = additional_info()

date = mk_datetime_var(data)
date() # ~ = date() 할당 안해도 data 변해있음. copy() 사용해야?



