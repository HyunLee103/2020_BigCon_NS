from calendar import month
from datetime import timedelta
from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus
import urllib
import requests
import json
import pandas as pd
import calendar
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import re
from ast import literal_eval

class EXTERNAL_DATA:
    def __init__(self):
        self.weather_key = 'ebq6gyrIdY0I%2BPEZjn3N091jOpB%2FfVGJhrnVGLJ9QGlyRIRGtemhB6dXYgXghbXYshun%2BXyNhEhKuFhAk2CtYA%3D%3D'
        self.client_id = ["BU2RiBkSkYGYoL4AsKks", 'X_E9NaAQh7lDbUmH1pzq']
        self.client_secret = ["h9zoGYRyUg", '9kccWvY9o9']
        self.perform_raw = pd.read_csv(r"data\2019_performance.csv")
        self.perform_raw['id'] = [i for i in range(self.perform_raw)]
        self.test = pd.read_csv(r"data\question.csv")

        self.perform_raw['상품명전처리'] = self.perform_raw['상품명'].apply(self.cleansing)
        self.perform_raw['검색용'] = self.perform_raw['상품명전처리'].apply(self.for_query)

        # self.test['상품명전처리'] = self.test['상품명'].apply(self.cleansing)
        # self.test['검색용'] = self.test['상품명전처리'].apply(self.for_query)

    # perform['방송일시'] 에다가 적용해서 사용하면 됨 / 새로운 col으로 만들어서 합칠 것
    def mk_datetime_hour(self, col):
        return datetime.strftime(datetime.strptime(col, '%Y-%m-%d %H:%M'), '%Y-%m-%d %H')

    def mk_datetime_day(self, col):
        return datetime.strftime(datetime.strptime(col, '%Y-%m-%d %H:%M'), '%Y-%m-%d')


    def cleansing(self, data):

        data = re.sub('\[페플럼제이\]', '페플럼제이', data)
        data = re.sub('\(일시불\)', '', data)
        data = re.sub('\(무이자\)', '', data)
        data = re.sub('일시불', '', data)
        data = re.sub('무이자', '', data)
        data = re.sub('\(세일20%\)', '', data)
        data = re.sub('\(1등급\)', '', data)
        data = re.sub('\(일\)', '', data)
        data = re.sub('\(무\)', '', data)
        data = re.sub('일\)', '', data)
        data = re.sub('\(세', '', data)
        data = re.sub('무\)', '', data)
        data = re.sub('\(쿠\)', '', data)
        data = re.sub('\(.*\)', '', data)
        data = re.sub('\[.*\]', '', data)
        data = re.sub('\[.*\］', '', data)
        for word in ['대', '통', '종', '세트', 'kg', 'g', '인용', '팩', '차', '미', '포', '구', 'L', '봉', '병', '매', '장', ',', '박스', '-', '개', '%', 'P', '마리', '단', '주', 'M']:
            data = re.sub(f'\d\d\d{word}', '', data)
        for word in ['대', '통', '종', '세트', 'kg', 'g', '인용', '팩', '차', '미', '포', '구', 'L', '봉', '병', '매', '장', ',', '박스', '-', '개', '%', 'P', '마리', '단', '주', 'M']:
            data = re.sub(f'\d\d{word}', '', data)
        for word in ['대', '통', '종', '세트', 'kg', 'g', '인용', '팩', '차', '미', '포', '구', 'L', '봉', '병', '매', '장', ',', '박스', '-', '개', '%', 'P', '마리', '단', '주', 'M']:
            data = re.sub(f'\d\.\d{word}', '', data)
        for word in ['대', '통', '종', '세트', 'kg', 'g', '인용', '팩', '차', '미', '포', '구', 'L', '봉', '병', '매', '장', ',', '박스', '-', '개', '%', 'P', '마리', '단', '주', 'M']:
            data = re.sub(f'\d{word}', '', data)
        for word in ['세트']:
            data = re.sub(f'{word}\d', '', data)
        data = re.sub('국내제조', '', data)
        data = re.sub('무료설치', '', data)
        data = re.sub('한세트', '', data)
        data = re.sub('기초세트', '', data)
        data = re.sub(' 총 ', '', data)
        data = re.sub('_', ' ', data)
        data = re.sub('한국인', ' ', data)
        data = re.sub('1000', ' ', data)
        data = re.sub('800', ' ', data)
        data = re.sub('풀세트', '', data)
        data = re.sub('세트', '', data)
        data = re.sub('패키지', '', data)
        data = re.sub('[0-9][0-9]\+[0-9][0-9]', '', data)
        data = re.sub('[0-9][0-9]\+[0-9]', '', data)
        data = re.sub('[0-9][0-9]\+', '', data)
        data = re.sub('[0-9]\+[0-9]', '', data)
        data = re.sub('[0-9]\+', '', data)
        data = re.sub('[0-9]\+', '', data)
        data = re.sub('\+.*', '', data)
        data = re.sub('\&.*', '', data)
        data = re.sub('x', '', data)
        data = re.sub(' ,', '', data)
        data = re.sub('프리미엄형', '', data)
        data = re.sub('싱글사이즈', '', data)
        data = re.sub('사이즈', '', data)
        data = re.sub('기본형 ', '', data)
        data = re.sub('中사이즈', '', data)
        data = re.sub('大사이즈', '', data)
        data = re.sub('시즌.*', '', data)
        data = re.sub('\*.*', '', data)
        data = re.sub('S/S', '', data)
        data = re.sub('F/W', '', data)
        data = re.sub(' SS', '', data)
        data = re.sub(' RX', '', data)
        data = re.sub(' SK', '', data)
        data = re.sub(' S', '', data)
        data = re.sub(' K', '', data)
        data = re.sub(' Q', '', data)
        data = re.sub('오리지널', '', data)
        data = re.sub(' 퀸', '', data)
        data = re.sub(' 킹', '', data)
        data = re.sub(' 싱글', '', data)
        data = re.sub(' 슈퍼싱글', '', data)
        data = re.sub('스페셜', '', data)
        data = re.sub('점보특대형', '', data)
        data = re.sub('특대형', '', data)
        data = re.sub('대형', '', data)
        data = re.sub('소형', '', data)
        data = re.sub('초특가\)', '', data)
        data = re.sub('가\)', '', data)
        data = re.sub('뉴 ', '', data)
        data = re.sub(' 플러스', '', data)
        data = re.sub(' 시그니처플러스', '', data)
        data = re.sub('!!', '', data)
        data = re.sub(' !', '', data)
        data = re.sub('2019년', '', data)
        data = re.sub('2019', '', data)
        data = re.sub('19년', '', data)
        data = re.sub('19', '', data)
        data = re.sub('Fall', '', data)
        data = re.sub('2020년', '', data)
        data = re.sub('2020', '', data)
        data = re.sub('20년', '', data)
        data = re.sub('단하루', '', data)
        data = re.sub('더블팩', '', data)
        data = re.sub('싱글팩', '', data)
        data = re.sub('\"', '', data)
        data = re.sub('SET', '', data)
        data = re.sub('18K', '', data)
        data = re.sub('24K', '', data)
        data = re.sub('14K', '', data)
        data = re.sub('  ', ' ', data)

        return data.strip()

    def for_query(self, data):

        data = data.split()
        if len(data) > 1:
            data = data[0] + ' ' + data[-1]
        else:
            data = data[0]

        return data

    def weather_api(self, typ='train'):

        url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'    
        cities = ['108', '112', '133', '143', '152', '156', '159']

        def query(start_date, end_date, day_num, city_num):
            queryParams = '?' + urlencode({ \
                    quote_plus('ServiceKey') : self.weather_key,\
                    quote_plus('pageNo') : '1', \
                    quote_plus('numOfRows') : day_num * 24, \
                    quote_plus('dataType') : 'JSON', \
                    quote_plus('dataCd') : 'ASOS', \
                    quote_plus('dateCd') : 'HR', \
                    quote_plus('startDt') : start_date, \
                    quote_plus('startHh') : '01', \
                    quote_plus('endDt') : end_date, \
                    quote_plus('endHh') : '01', \
                    quote_plus('stnIds') : city_num })
            
            return queryParams

        if typ == 'train':
            weather_data = pd.DataFrame()
            for city_num in cities:
                city_data = pd.DataFrame()
                for month in range(1, 14):
                    try:
                        start_date = (date(2019,month,1)).strftime("%Y%m%d")
                        end_date = (date(2019,month,1)+ relativedelta(months=1)).strftime("%Y%m%d")
                    except:            
                        start_date = '20200101'
                        end_date = '20200102'

                    day_num = calendar.monthrange(int(start_date[:4]), int(start_date[4:6]))[1]
                    queryParams = query(start_date, end_date, day_num, city_num)
                    req = urllib.request.Request(url + unquote(queryParams))
                    response_body = urlopen(req, timeout=60).read() # get bytes data ## ss : 일조, 'hm' : 습도, 'rn' : 강수, 'ta' : 온도
                    data = pd.DataFrame(json.loads(response_body)['response']['body']['items']['item'])[['tm', 'ss', 'hm', 'rn', 'ta']]	# convert bytes data to json data
                    data = data.set_index(['tm'])
                    data.index.apply(self.mk_datetime)
                    city_data = pd.concat([city_data, data])
                city_data.columns = [col + '_' + city_num for col in city_data.columns if col != 'tm']
                weather_data = pd.concat([weather_data, city_data], axis=1)

            return weather_data

        else:
            weather_data = pd.DataFrame()
            for city_num in cities:
                city_data = pd.DataFrame()
                for month in range(6, 8):
                    start_date = (date(2020,month,1)).strftime("%Y%m%d")
                    end_date = (date(2020,month,1)+ relativedelta(months=1)).strftime("%Y%m%d")

                    day_num = calendar.monthrange(int(start_date[:4]), int(start_date[4:6]))[1]
                    queryParams = query(start_date, end_date, day_num, city_num)
                    req = urllib.request.Request(url + unquote(queryParams))
                    response_body = urlopen(req, timeout=60).read() # get bytes data ## ss : 일조, 'hm' : 습도, 'rn' : 강수, 'ta' : 온도
                    data = pd.DataFrame(json.loads(response_body)['response']['body']['items']['item'])[['tm', 'ss', 'hm', 'rn', 'ta']]	# convert bytes data to json data
                    data = data.set_index(['tm'])
                    data.index.apply(self.mk_datetime)
                    city_data = pd.concat([city_data, data])
                city_data.columns = [col + '_' + city_num for col in city_data.columns if col != 'tm']
                weather_data = pd.concat([weather_data, city_data], axis=1)

            return weather_data

    def naver_query_trend_api(self):
        
        url = "https://openapi.naver.com/v1/datalab/search"

        def mk_body(start_date, end_date, keyword):
            
            keyword = '삼성전자'
            body = "{\"startDate\":\"" + start_date + "\", \
            \"endDate\":\"" + end_date + "\", \
            \"timeUnit\":\"date\", \
            \"keywordGroups\":[{\"groupName\":\"쇼핑\",\"keywords\":[ \"" + keyword + "\"]}]}"

            return body

        trend_63 = []
        trend_28 = []
        trend_7 = []
        for date, query in zip(self.perform_raw['방송일시'], self.perform_raw['검색용']):
            start_date = datetime.strftime(datetime.strptime(date, '%Y-%m-%d %H:%M') - timedelta(63), '%Y-%m-%d')
            end_date = datetime.strftime(datetime.strptime(date, '%Y-%m-%d %H:%M') - timedelta(7), '%Y-%m-%d')
            body = mk_body(start_date, end_date, keyword = query)

            request = urllib.request.Request(url)
            request.add_header("X-Naver-Client-Id", self.client_id)
            request.add_header("X-Naver-Client-Secret", self.client_secret)
            request.add_header("Content-Type","application/json")
            response = urllib.request.urlopen(request, data=body.encode("utf-8"))
            rescode = response.getcode()

            if(rescode==200):
                response_body = response.read()
                ratio = pd.DataFrame(literal_eval(response_body.decode('utf-8'))['results'][0]['data'])
                trend_63.append(ratio['ratio'].mean())
                trend_28.append(ratio.ratio[28:].mean())
                trend_7.append(ratio.ratio[-7:].mean())

            else:
                print("Error Code:" + rescode)

        final = pd.DataFrame(trend_63, trend_28, trend_7, columns=['trend_63', 'trend_28', 'trend_7'])

        return final

    def naver_shopping_trend_api(self):

        url = "https://openapi.naver.com/v1/datalab/shopping/categories"

        category_dict = {'패션의류' : '50000000',
                        '패션잡화' : '50000001',
                        '화장품/미용' : '50000002',
                        '디지털/가전' : '50000003',
                        '가구/인테리어' : '50000004',
                        '출산/육아' : '50000005',
                        '식품' : '50000006',
                        '스포츠/레저' : '50000007',
                        '생활건강' : '50000008',
                        '여가/생활편의' : '50000009',
                        '면세점' : '50000010'}

        def mk_body(category_dict, category):
            
            body = "{\"startDate\":\"2019-01-01\", \
                    \"endDate\":\"2020-01-02\",\
                    \"timeUnit\":\"date\",\
                    \"category\":[{\"name\":\"" + category + "\",\"param\":[\"" + category_dict[category] + "\"]}]}"

            return body

        final = pd.DataFrame()
        for category in category_dict.keys():
            body = mk_body(category_dict, category)
            request = urllib.request.Request(url)
            request.add_header("X-Naver-Client-Id", self.client_id)
            request.add_header("X-Naver-Client-Secret", self.client_secret)
            request.add_header("Content-Type","application/json")
            response = urllib.request.urlopen(request, data=body.encode("utf-8"))
            rescode = response.getcode()
            if(rescode==200):
                response_body = response.read()
                cate_ratio = pd.DataFrame(literal_eval(response_body.decode('utf-8'))['results'][0]['data'])
                cate_ratio = cate_ratio.set_index(['period'])
                final = pd.concat([final, cate_ratio], axis=1)

            else:
                print("Error Code:" + rescode)
        
        final.index = cate_ratio.index
        final.columns = [f'category_ratio_{category_dict[category][-2:]}' for category in category_dict.keys()]

        return final

ex = EXTERNAL_DATA()
ex.naver_shopping_trend_api()
ex.naver_query_trend_api()

ex.perform_raw['검색용'].unique().__len__()


def naver_query_trend_api():

    url = "https://openapi.naver.com/v1/datalab/search"

    def mk_body(keyword):
        
        keyword = '삼성전자'
        body = "{\"startDate\":\"2019-01-01\", \
        \"endDate\":\"2019-01-31\", \
        \"timeUnit\":\"date\", \
        \"keywordGroups\":[{\"groupName\":\"쇼핑\",\"keywords\":[ \"" + keyword + "\"]}],\
        \"device\":\"pc\"}"

        return body

    ## Naver 검색어 트렌드 api

    # 기간 조절 필요
    # TimeUnit 조절 필요 ( 일간 주간 월간 가능 )
    # 검색 환경, 연령, 성별
    for query in train['검색용']:
        body = mk_body(query)
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)
        request.add_header("Content-Type","application/json")
        response = urllib.request.urlopen(request, data=body.encode("utf-8"))
        rescode = response.getcode()
        if(rescode==200):
            response_body = response.read()
            print(response_body.decode('utf-8'))
            print(json.loads(response_body.decode('utf-8'))['results'][0]['data'])
            print(pd.DataFrame(response_body.decode('utf-8')['data']))
        else:
            print("Error Code:" + rescode)



ratio = pd.DataFrame(literal_eval(response_body.decode('utf-8'))['results'][0]['data'])
ratio.ratio.mean()

category_dict.keys()




response = urllib.request.urlopen(request, data=body.encode("utf-8"))
response.read()
body.encode('utf-8')

body = "{\"startDate\":\"2017-08-01\", \
    \"endDate\":\"2017-09-30\", \
    \"timeUnit\":\"month\",\
    \"category\":[{\"name\":\"패션의류\",\"param\":[\"50000000\"]},\
    {\"name\":\"화장품/미용\",\"param\":[\"50000002\"]}],\
        \"device\":\"pc\",\"ages\":[\"20\",\"30\"],\"gender\":\"f\"}";
naver_query_trend_api()


client_id = "BU2RiBkSkYGYoL4AsKks"
client_secret = "h9zoGYRyUg"



naver_query_trend_api()


"""
train['상품명전처리'] = train['상품명'].apply(cleansing)
train[train['상품명전처리'] == ''].index
train['검색용'] = train['상품명전처리'].apply(for_query)

train[['상품명', '상품명전처리', '검색용']].to_excel('test.xlsx')





## 리뷰를 긁어오는 것도 생각해볼 수 있을 듯
"""


def cleansing(data):

    data = re.sub('\[페플럼제이\]', '페플럼제이', data)
    data = re.sub('\(일시불\)', '', data)
    data = re.sub('\(무이자\)', '', data)
    data = re.sub('일시불', '', data)
    data = re.sub('무이자', '', data)
    data = re.sub('\(세일20%\)', '', data)
    data = re.sub('\(1등급\)', '', data)
    data = re.sub('\(일\)', '', data)
    data = re.sub('\(무\)', '', data)
    data = re.sub('일\)', '', data)
    data = re.sub('\(세', '', data)
    data = re.sub('무\)', '', data)
    data = re.sub('\(쿠\)', '', data)
    data = re.sub('\(.*\)', '', data)
    data = re.sub('\[.*\]', '', data)
    data = re.sub('\[.*\］', '', data)
    for word in ['대', '통', '종', '세트', 'kg', 'g', '인용', '팩', '차', '미', '포', '구', 'L', '봉', '병', '매', '장', ',', '박스', '-', '개', '%', 'P', '마리', '단', '주', 'M']:
        data = re.sub(f'\d\d\d{word}', '', data)
    for word in ['대', '통', '종', '세트', 'kg', 'g', '인용', '팩', '차', '미', '포', '구', 'L', '봉', '병', '매', '장', ',', '박스', '-', '개', '%', 'P', '마리', '단', '주', 'M']:
        data = re.sub(f'\d\d{word}', '', data)
    for word in ['대', '통', '종', '세트', 'kg', 'g', '인용', '팩', '차', '미', '포', '구', 'L', '봉', '병', '매', '장', ',', '박스', '-', '개', '%', 'P', '마리', '단', '주', 'M']:
        data = re.sub(f'\d\.\d{word}', '', data)
    for word in ['대', '통', '종', '세트', 'kg', 'g', '인용', '팩', '차', '미', '포', '구', 'L', '봉', '병', '매', '장', ',', '박스', '-', '개', '%', 'P', '마리', '단', '주', 'M']:
        data = re.sub(f'\d{word}', '', data)
    for word in ['세트']:
        data = re.sub(f'{word}\d', '', data)
    data = re.sub('국내제조', '', data)
    data = re.sub('무료설치', '', data)
    data = re.sub('한세트', '', data)
    data = re.sub('기초세트', '', data)
    data = re.sub(' 총 ', '', data)
    data = re.sub('_', ' ', data)
    data = re.sub('한국인', ' ', data)
    data = re.sub('1000', ' ', data)
    data = re.sub('800', ' ', data)
    data = re.sub('풀세트', '', data)
    data = re.sub('세트', '', data)
    data = re.sub('패키지', '', data)
    data = re.sub('[0-9][0-9]\+[0-9][0-9]', '', data)
    data = re.sub('[0-9][0-9]\+[0-9]', '', data)
    data = re.sub('[0-9][0-9]\+', '', data)
    data = re.sub('[0-9]\+[0-9]', '', data)
    data = re.sub('[0-9]\+', '', data)
    data = re.sub('[0-9]\+', '', data)
    data = re.sub('\+.*', '', data)
    data = re.sub('\&.*', '', data)
    data = re.sub('x', '', data)
    data = re.sub(' ,', '', data)
    data = re.sub('프리미엄형', '', data)
    data = re.sub('싱글사이즈', '', data)
    data = re.sub('사이즈', '', data)
    data = re.sub('기본형 ', '', data)
    data = re.sub('中사이즈', '', data)
    data = re.sub('大사이즈', '', data)
    data = re.sub('시즌.*', '', data)
    data = re.sub('\*.*', '', data)
    data = re.sub('S/S', '', data)
    data = re.sub('F/W', '', data)
    data = re.sub(' SS', '', data)
    data = re.sub(' RX', '', data)
    data = re.sub(' SK', '', data)
    data = re.sub(' S', '', data)
    data = re.sub(' K', '', data)
    data = re.sub(' Q', '', data)
    data = re.sub('오리지널', '', data)
    data = re.sub(' 퀸', '', data)
    data = re.sub(' 킹', '', data)
    data = re.sub(' 싱글', '', data)
    data = re.sub(' 슈퍼싱글', '', data)
    data = re.sub('스페셜', '', data)
    data = re.sub('점보특대형', '', data)
    data = re.sub('특대형', '', data)
    data = re.sub('대형', '', data)
    data = re.sub('소형', '', data)
    data = re.sub('초특가\)', '', data)
    data = re.sub('가\)', '', data)
    data = re.sub('뉴 ', '', data)
    data = re.sub(' 플러스', '', data)
    data = re.sub(' 시그니처플러스', '', data)
    data = re.sub('!!', '', data)
    data = re.sub(' !', '', data)
    data = re.sub('2019년', '', data)
    data = re.sub('2019', '', data)
    data = re.sub('19년', '', data)
    data = re.sub('19', '', data)
    data = re.sub('Fall', '', data)
    data = re.sub('2020년', '', data)
    data = re.sub('2020', '', data)
    data = re.sub('20년', '', data)
    data = re.sub('단하루', '', data)
    data = re.sub('더블팩', '', data)
    data = re.sub('싱글팩', '', data)
    data = re.sub('\"', '', data)
    data = re.sub('SET', '', data)
    data = re.sub('18K', '', data)
    data = re.sub('24K', '', data)
    data = re.sub('14K', '', data)
    data = re.sub('  ', ' ', data)

    return data.strip()

def for_query(data):

    data = data.split()
    if len(data) > 1:
        data = data[0] + ' ' + data[-1]
    else:
        data = data[0]

    return data

perform['상품명전처리'] = perform['상품명'].apply(cleansing)
perform['검색명'] = perform['상품명전처리'].apply(for_query)
perform['day'] = perform['방송일시'].apply(lambda x : datetime.strftime(datetime.strptime(x, '%Y-%m-%d %H:%M'), '%Y-%m-%d'))

perform[['day', '검색명']].drop_duplicates()

perform[['상품명', '상품명전처리', '검색명']].to_excel("test.xlsx")

perform = pd.read_csv(r"data\2019_performance.csv")