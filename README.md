# NS 홈쇼핑 매출액 예측
NS Shop+편성데이터(NS홈쇼핑)를 활용하여 방송편성표에 따른 판매실적을 예측하고, 최적 수익을 고려한 요일별/ 시간대별 / 카테고리별 편성 최적화 방안 제시


## Process
날씨정보, 검색어 트랜드 정보 반영 전처리

    load_data(data_path, trend = True, weather = True)
Feature Engineering

    make_variable(perform_raw,test_raw,rating)
