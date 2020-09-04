from datetime import timedelta
from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus
import urllib
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
        self.client_ids = ["BU2RiBkSkYGYoL4AsKks", 'X_E9NaAQh7lDbUmH1pzq', 'J8x7ATegg_xhEYHPPVlx', 'xnLyB7AdYFHBLCT9XRU5', 'ZQ3P1B2c1UPf2NQJ5qwz', 'wdG7ibZrncM9nVHLsDj0', 'C4wH6U786JdGW6Mq8Bsr', 'v7c2vfKDgh8ns7uqTXsA']
        self.client_secrets = ["h9zoGYRyUg", '9kccWvY9o9', 'vXIB9LCfwo', 'uInvwFD9nR', '6X1zL82hWt', '_58eNSboSg', 'iizcleAcOz', 'dKClhuja8t']
        self.query_api_use = 0
        self.shop_api_use = 0
        self.client_num = 0
        self.perform_raw = pd.read_csv(r"data\2019_performance.csv")
        self.perform_raw['id'] = [i for i in range(len(self.perform_raw['상품명']))]
        self.test = pd.read_csv(r"data\question.csv")

        self.perform_raw['상품명전처리'] = self.perform_raw['상품명'].apply(self.cleansing)
        self.perform_raw['검색용'] = self.perform_raw['상품명전처리'].apply(self.for_query)

        # self.test['상품명전처리'] = self.test['상품명'].apply(self.cleansing)
        # self.test['검색용'] = self.test['상품명전처리'].apply(self.for_query)

    def client(self, api):
        if api == 'query':
            if self.query_api_use > 950:
                self.client_num += 1
                self.query_api_use = 0
            client_id = self.client_ids[self.client_num]
            client_secret = self.client_secrets[self.client_num]
            self.query_api_use += 1

        elif api == 'shop':
            if self.shop_api_use > 950:
                self.client_num += 1
                self.shop_api_use = 0
            client_id = self.client_ids[self.client_num]
            client_secret = self.client_secrets[self.client_num]
            self.shop_api_use += 1

        else:
            print('점심 나가서 먹을꺼 같애')

        return client_id, client_secret

    def naver_connect(self, url, client_id, client_secret, body):

        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)
        request.add_header("Content-Type","application/json")
        response = urllib.request.urlopen(request, data=body.encode("utf-8"))
        rescode = response.getcode()
        
        return rescode, response.read()

    # perform['방송일시'] 에다가 적용해서 사용하면 됨 / 새로운 col으로 만들어서 합칠 것
    def mk_datetime_hour(self, col):
        return datetime.strftime(datetime.strptime(col, '%Y-%m-%d %H:%M'), '%Y-%m-%d %H')

    def mk_datetime_day(self, col):
        return datetime.strftime(datetime.strptime(col, '%Y-%m-%d %H:%M'), '%Y-%m-%d')

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
                    data['tm'] = data['tm'].apply(self.mk_datetime_hour)
                    data = data.set_index(['tm'])
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
                    data['tm'] = data['tm'].apply(self.mk_datetime_hour)
                    data = data.set_index(['tm'])
                    city_data = pd.concat([city_data, data])
                city_data.columns = [col + '_' + city_num for col in city_data.columns if col != 'tm']
                weather_data = pd.concat([weather_data, city_data], axis=1)

            return weather_data

    def naver_query_trend_api(self):
        
        url = "https://openapi.naver.com/v1/datalab/search"
        client_id, client_secret = self.client('query')

        def mk_body(start_date, end_date, keyword):
            
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
            
            rescode, response_body = self.naver_connect(url, client_id, client_secret, body)
            
            if(rescode==200):
                ratio = pd.DataFrame(literal_eval(response_body.decode('utf-8'))['results'][0]['data'])
                trend_63.append(ratio['ratio'].mean())
                trend_28.append(ratio.ratio[28:].mean())
                trend_7.append(ratio.ratio[-7:].mean())

            else:
                print("Error Code:" + rescode)
            break

        final = pd.DataFrame(trend_63, trend_28, trend_7, columns=['trend_63', 'trend_28', 'trend_7'])
        #TODO
        # perform_raw 에서 unique한 것들만 남긴 다음에
        # 걔네를 merge해서 사용해야 그러면 날짜로 unique를 남겨야 겠고만 그러면 날짜 self.mkdatetimeday사용
        # 그리고나서 return 하면 되나?

        return final

    def naver_shopping_trend_api(self):

        url = "https://openapi.naver.com/v1/datalab/shopping/categories"
        client_id, client_secret = self.client('shop')

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

            rescode, response_body = self.naver_connect(url, client_id, client_secret, body)

            if(rescode==200):
                cate_ratio = pd.DataFrame(literal_eval(response_body.decode('utf-8'))['results'][0]['data'])
                cate_ratio = cate_ratio.set_index(['period'])
                final = pd.concat([final, cate_ratio], axis=1)

            else:
                print("Error Code:" + rescode)
        
        final.index = cate_ratio.index
        final.columns = [f'category_ratio_{category_dict[category][-2:]}' for category in category_dict.keys()]

        return final

    def cleansing(self, data):

        data = re.sub('\[페플럼제이\]', '페플럼제이', data)
        for word in ['\(일시불\)', '\(무이자\)', '일시불', '무이자', '\(세일20%\)', '\(1등급\)', '\(일\)', '\(무\)', '일\)', '\(세', '무\)', '\(쿠\)', '\(.*\)', '\[.*\]', '\[.*\］']:
            data = re.sub(word, '', data)
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
        for word in ['국내제조', '무료설치', '한세트', '기초세트', ' 총 ', '_', '한국인', '1000', '800', '풀세트', '세트', '패키지', '[0-9][0-9]\+[0-9][0-9]', \
            '[0-9][0-9]\+[0-9]', '[0-9][0-9]\+', '[0-9]\+[0-9]', '[0-9]\+', '\+.*', '\&.*', 'x', ' ,', '프리미엄형','싱글사이즈',\
            '사이즈', '기본형 ', '中사이즈', '大사이즈', '시즌.*', '\*.*', 'S/S', 'F/W', ' SS', ' RX', ' SK', ' S', \
            ' K', ' Q', '오리지널', ' 퀸', ' 킹', ' 싱글', ' 슈퍼싱글','스페셜', '점보특대형', '특대형',\
            '대형',  '소형', '초특가\)','가\)', '뉴 ', ' 플러스', ' 시그니처플러스', '!!', ' !', '2019년', '2019', '19년', '19', \
            'Fall', '2020년', '2020', '20년', '단하루', '더블팩', '싱글팩', '\"', 'SET', '18K', '24K', '14K', '  ']:
            data = re.sub(word, '', data)

        return data.strip()

    def for_query(self, data):

        data = data.split()
        if len(data) > 1:
            data = data[0] + ' ' + data[-1]
        else:
            data = data[0]

        return data

ex = EXTERNAL_DATA()
ex.weather_api()
ex.naver_shopping_trend_api()
ex.naver_query_trend_api()

"""
## 리뷰를 긁어오는 것도 생각해볼 수 있을 듯
## 코스피 추가하고 ( 시간대가 근데 그게 안되서 좀 별로일지도 )
## 추가할만한 경제지표를 추가해보는 것도 나쁘지 않을 듯
"""