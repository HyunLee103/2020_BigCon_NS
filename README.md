# NS 홈쇼핑 매출액 예측
NS Shop+편성데이터(NS홈쇼핑)를 활용하여 방송편성표에 따른 판매실적을 예측하고, 최적 수익을 고려한 요일별/ 시간대별 / 카테고리별 편성 최적화 방안 제시


## Process

    # 날씨정보, 검색어 트랜드 정보 반영 전처리
    load_data(data_path, trend = True, weather = True)  
    
    # Feature Engineering
    make_variable(perform_raw, test_raw, rating)  
    
    # 이상치 제거 및 클러스터링
    preprocess(train_var, test_var, outlier_rate, # of cluster)
학습 데이터 셋 형태 구축
    
    mk_trainset, clutering
학습 및 inference
    
    final_test(train, val, 3, hyperparmeters)

## Usage
'data/'에 데이터 넣은 뒤

    !python main.py 

## Requirements
- pandas
- numpy
- sklearn
- joblib
- seaborn
- lightgbm

## Contributor
