import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import copy

# 추가 정보 데이터프레임 따로 생성. performance(data) 데이터프레임과 인덱스로 매칭
class mk_additional_info():

    '''
    input: perform, rating
    return: (__call__) additional info dataframe (주문량,방송ID,시청률,시청률_방송별)
    '''

    def __init__(self,data,rating):
        self.data = data 
        self.rating = rating
        self.info = pd.DataFrame()

    def mk_order(self):
        return round(self.data['취급액']/self.data['판매단가'])

    def str_to_datetime(self,date):

        if date:
            return self.data['방송일시'].map(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M'))

        else: # for rating['시간대]
            return self.rating['시간대'].map(lambda x: datetime.strptime(x,'%H:%M').time())

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
    
    def mk_rating(self,by_show=False):
        self.data['노출(분)_수정'] = copy.deepcopy(self.data['노출(분)']).replace(0,method='ffill')

        def cal_ratio(date,long):

            time_idx = dict(zip(self.rating.시간대,rating.index)) 
            # key - 시간대(str), value - 인덱스

            # 예외처리
            if (date.date().year == 2020):
                return 0

            else:
                rating_for_day = self.rating.loc[:,['시간대',str(date.date())]]

                start_idx = time_idx[date.time()]
                end = date + timedelta(minutes = long)
                end_idx = time_idx[end.time()]

            return self.rating.loc[start_idx:end_idx,str(date.date())].mean()

        if by_show:
            if '시청률' in self.info.columns:
                pass

            else:
                self.info['시청률'] = self.data.apply(lambda x: cal_ratio(x['방송일시'],x['노출(분)_수정']) ,axis=1)
            
            
            showid_rating = dict(self.info.groupby('방송ID')['시청률'].mean())

            return self.info['방송ID'].map(lambda x: showid_rating[x])

        else:
            return self.data.apply(lambda x: cal_ratio(x['방송일시'],x['노출(분)_수정']) ,axis=1)
        
    def __call__(self): # 모든 함수 실행

        self.data['방송일시'] = self.str_to_datetime(date=True)
        self.rating['시간대'] = self.str_to_datetime(date=False)

        self.info['주문량'] = self.mk_order()
        self.info['방송ID'] = self.mk_showid()
        self.info['시청률'] = self.mk_rating(by_show=False)
        self.info['시청률_방송별'] = self.mk_rating(by_show=True)

        return self.info

class mk_datetime_var():
    
    '''
    input: perform
    return: (__call__) 요일, 휴일 여부, 방송시간-시, 방송시간-분
    '''

    def __init__(self,data):
        self.data = data
        
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

        self.data['방송요일'] = self.mk_weekday()
        self.data['휴일'] = self.mk_holiday()
        self.data['방송시각'] = self.mk_hour(grouping=False)
        self.data['방송분'] = self.mk_min(grouping=True)

        return self.data


class mk_showtime_var():
    '''
    input: perform
    return: (__call__) 노출(분)_그룹
    '''

    def __init__(self,data):
        self.data = data
        self.data['노출(분)_수정'] = copy.deepcopy(self.data['노출(분)']).replace(0,method='ffill')

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
        self.data['노출(분)_그룹'] = self.data['노출(분)_수정'].map(self.grouping())

        return self.data

class mk_mcode_var():
    '''
    input: perform
    return: (__call__) 마더코드_빈도(=마더코드 기준 방송빈도)
    '''

    def __init__(self,data,info):
        self.data = data
        self.info = info

    def mk_mcode_freq(self): 
        # 마더코드에 따른 방송횟수.(유사 브랜드파워) 
        # 한 방송에 마더코드 여러개일 수 있음. > 둘 다 카운팅 되어야

        self.info['마더코드'] = copy.deepcopy(self.data['마더코드'])
        tmp = self.info.groupby('방송ID')['마더코드'].apply(lambda x: list(set(x))).reset_index(name='마더코드')
        mcode_showfreq = dict(pd.Series(sum([mcode for mcode in tmp.마더코드],[])).value_counts())

        return self.data['마더코드'].map(lambda x: mcode_showfreq[x])

    def __call__(self):
        self.data['마더코드_빈도'] = self.mk_mcode_freq()

        del info['마더코드']
        
        return self.data

class mk_pcode_var():
    '''
    input: perform
    return: (__call__) 상품코드_빈도(=상품코드 기준 방송빈도)
    '''
    def __init__(self,data,info):
        self.data = data
        self.info = info

    def mk_pcode_freq(self):
        self.info['상품코드'] = copy.deepcopy(self.data['상품코드'])
        tmp = self.info.groupby('방송ID')['상품코드'].apply(lambda x: list(set(x))).reset_index(name='상품코드')
        pcode_showfreq = dict(pd.Series(sum([pcdoe for pcdoe in tmp.상품코드],[])).value_counts())

        return self.data['상품코드'].map(lambda x: pcode_showfreq[x])

    def __call__(self):
        self.data['상품코드_빈도'] = self.mk_pcode_freq()

        del info['상품코드']

        return self.data

class mk_rating_var():
    '''
    input: perform
    return: (__call__) 시청률, 방송시청률
    '''
    def __init__(self,data,info):
        self.data = data
        self.info = info

    def mk_rating(self):
        return copy.deepcopy(self.info['시청률'])

    def mk_rating_byshow(self):
        return copy.deepcopy(self.info['시청률_방송별'])

    def __call__(self):

        self.data['시청률'] = self.mk_rating()
        self.data['시청률_방송별'] = self.mk_rating_byshow()

        return self.data


# 사용
data = pd.read_csv('data/2019_performance.csv', encoding='utf-8') # 실제론 test와 concat해야
rating = pd.read_csv('data/2019_rating.csv',encoding='utf-8')

additional_info = mk_additional_info(data,rating)
info = additional_info()
info

date = mk_datetime_var(data)
date() # ~ = date() 할당 안해도 data 변해있음. copy.deepcopy(data) 사용해야?

# test셋에 마더코드, 상품 코드 겹치는 것 적어서 걱정. 
# 상품명 전처리 거친 걸로 다시 빈도 뽑아볼수도? (브랜드, 상품명)

showtime = mk_showtime_var(data)
showtime()

mcode = mk_mcode_var(data,info)
mcode()

pcode = mk_pcode_var(data,info)
pcode()

ratings = mk_rating_var(data,info)
ratings()

# 다른 변수와 시청률 관계
day_rating = pd.DataFrame(data.groupby('방송요일')['시청률'].mean().reset_index())
day_rating.plot.scatter(x='방송요일', y='시청률')

hour_rating = pd.DataFrame(data.groupby('방송시각')['시청률'].mean()).reset_index()
hour_rating.plot.scatter(x='방송시각', y='시청률')

min_rating = pd.DataFrame(data.groupby('방송분')['시청률'].mean()).reset_index()
min_rating.plot.scatter(x='방송분', y='시청률')

mfreq_rating = pd.DataFrame(data.groupby('마더코드_빈도')['시청률'].mean()).reset_index()
mfreq_rating[mfreq_rating['마더코드_빈도']<500].plot.scatter(x='마더코드_빈도',y='시청률',c='DarkBlue')

pfreq_rating = pd.DataFrame(data.groupby('상품코드_빈도')['시청률'].mean()).reset_index()
pfreq_rating[pfreq_rating['상품코드_빈도']<500].plot.scatter(x='상품코드_빈도',y='시청률',c='DarkBlue')

order_rating = pd.DataFrame(info.groupby('주문량')['시청률'].mean()).reset_index()
order_rating.plot.scatter(x='주문량',y='시청률')
order_rating[order_rating['주문량']<6000].plot.scatter(x='주문량',y='시청률')
order_rating[order_rating['주문량']<3000].plot.scatter(x='주문량',y='시청률')

info.groupby('방송ID')['시청률'].mean().describe()
info[info['시청률_방송별']> 0.08] # 손질 오징어, 진공 스텐 냄비
