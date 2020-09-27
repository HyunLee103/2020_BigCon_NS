import numpy as np
import datetime
import joblib

# Machine Learning Modeling
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.bagging import BaggingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingRegressor, StackingRegressor

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

from util import load_data,preprocess,mk_trainset, metric
from make_var import make_variable
from clustering import clustering

# 자기가 실행할 모델 파라미터 관련 코드 실행 (그냥 다한다! : >)
# real implement에서 lightgbm 제외하고는 mk_trainset의 category False, PCA True로 설정.
# random_search의 모델 명, 파라미터 명 설정. (random_search 함수 내 파라미터 저장 이름도 확인. 덮어쓰기 안되게!)
# 파라미터 로드는 맨 아래 방법처럼.

n_estimators = [int(x) for x in range(10000,50000,5000)]
importance_type = ['split','gain']
lambda_l1 = sp_randFloat()
lambda_l2 = sp_randFloat()
max_depth =  sp_randInt(3, 30)
depth = sp_randInt(3, 30)
min_child_samples = sp_randInt(1,7)
min_data_in_leaf = sp_randInt(1,7)
min_sum_hessian_in_leaf = sp_randInt(1,10)
num_leaves = sp_randInt(10,50)
bagging_fraction = sp_randFloat()
feature_fraction = sp_randFloat()
learning_rate = sp_randFloat()
max_bin = sp_randInt(low=0, high=30)
min_gain_to_split = sp_randFloat()
max_leaf_nodes = sp_randInt(10,50)
min_samples_leaf = sp_randInt(2,30)
min_samples_split = sp_randInt(2,30)
min_weight_fraction_leaf = sp_randFloat()
l2_leaf_reg =sp_randFloat()
border_count = sp_randInt(1,30)
ctr_border_count= sp_randInt(1,30)
gamma = sp_randFloat()
min_child_weight =  np.linspace(0.1, 4, 10)

max_samples = sp_randInt(1,10)
max_features = sp_randInt(10,50)
bagging_temperature = sp_randInt(0, 5)

# Lightgbm
lgbm_params ={'n_estimators': n_estimators,
              'importance_type':importance_type,
              'lambda_l1':lambda_l1,
              'lambda_l2':lambda_l2,
              'max_depth':max_depth,
              'min_child_samples':min_child_samples,
              'min_data_in_leaf':min_data_in_leaf,
              'min_sum_hessian_in_leaf':min_sum_hessian_in_leaf,
              'num_leaves':num_leaves,
              'bagging_fraction':bagging_fraction,
              'feature_fraction':feature_fraction,
              'max_bin':max_bin,
              'learning_rate':learning_rate,
              'min_gain_to_split':min_gain_to_split
             }

model_lgbm = LGBMRegressor(boosting_type='gbdt',random_state=2020)

# Randomforest
rf_params = {'max_depth':max_depth,
             'max_leaf_nodes': max_leaf_nodes,
             'min_samples_leaf':min_samples_leaf,
             'min_samples_split':min_samples_split,
             'n_estimators':n_estimators,
             'min_weight_fraction_leaf':min_weight_fraction_leaf
            }

model_rf = RandomForestRegressor(bootstrap=True,random_state=2020)

# Catboost
catb_params = {'learning_rate':learning_rate,
               'depth':depth,
               'l2_leaf_reg':l2_leaf_reg,
               'border_count':border_count,
            #    'ctr_border_count':ctr_border_count,
               'n_estimators':n_estimators,
               'bagging_temperature' : bagging_temperature,
               'min_data_in_leaf' : min_data_in_leaf
              }

model_catb = CatBoostRegressor(random_state=2020)


# XGBoost
xgb_params = {'max_depth':max_depth,
              'learning_rate':learning_rate,
              'gamma':gamma,
              'min_child_weight':min_child_weight,
              'n_estimators':n_estimators
             }

model_xgb = XGBRegressor(booster='gbtree', objective='reg:squarederror',random_state=2020)
 

# Bagging
base_estimator = [model_lgbm, model_catb, model_rf, model_xgb]

bagging_params ={ 'base_estimator' : base_estimator,
            'n_estimators': n_estimators,
            'max_samples' : max_samples,
            'max_features' : max_features
             }

model_bagging = BaggingRegressor()

def random_search(model, params, X_train, y_train, X_val, y_val):
    print('-'*100)
    start_time = datetime.datetime.now()
    print('Start Time : {}'.format(start_time))
    
    rnd_search = RandomizedSearchCV(model,
                                   param_distributions=params,
                                   n_iter = 500,
                                   cv = 2,
                                   scoring ='neg_mean_absolute_error',
                                   verbose =2,
                                   n_jobs=2,
                                   random_state=2020)
    
    search = rnd_search.fit(X_train, y_train)
    print('Best Estimator : {}'.format(search.best_estimator_))
    print('Best Params : {}'.format(search.best_params_))

    print('Save Best Params...')
    joblib.dump(search.best_params_, f'best_params.pkl', compress = 1)

    end_time = datetime.datetime.now()

    print('End Time : {}'.format(start_time))
    print('Duration Time : {}'.format(end_time-start_time))


# real implement
data_path = 'data/'
perform_raw, rating, test_raw = load_data(data_path,trend=False,weather=False)
train_var, test_var = make_variable(perform_raw,test_raw,rating)
raw_data, y_km, train_len= preprocess(train_var,test_var,0.03,3,inner=False) # train, test 받아서 쓰면 돼
data = mk_trainset(raw_data,categorical=False,PCA=True) # lightgbm 제외하고는 categorical False로
train, val, robustScaler = clustering(data,y_km,train_len)

drop0_ = ['min_order_med', 'day_order_rank', 'min_sales_rank', 'min_sales_med',
'day_sales_rank', 'min_order_rank', 'cate_order_std', 'cate_sales_rank',
'min_sales_std', 'min_order_std', 'min', 'cate_order_rank',
'min_order_mean', 'rating', 'min_sales_mean', 'prime', 'cate_order_med',
'day_order_med']

drop1_  = ['min_sales_med',  'min_sales_std',  'day_sales_rank', 'min_sales_rank',
'min_order_rank', 'cate_sales_rank', 'cate_order_rank', 'cate_order_med',
'cate_sales_med', 'prime', 'min_order_std', 'min_sales_mean',
'day_order_rank', 'cate_order_std', 'min_order_med', 'min_order_mean',
'cate_sales_mean', 'cate_sales_std', 'day_order_std', 'rating',
'day_sales_med', 'min', 'day_order_med']

drop = list(set(drop0_).intersection(drop1_))

train_0_y = train[train['kmeans'] == 0]['sales']
train_1_y = train[train['kmeans'] == 1]['sales']
train_2_y = train[train['kmeans'] == 2]['sales']

train_0 = train[train['kmeans'] == 0].drop(['sales']+list(set(train.columns).intersection(drop)), axis=1)
train_1 = train[train['kmeans'] == 1].drop(['sales']+list(set(train.columns).intersection(drop)), axis=1)
train_2 = train[train['kmeans'] == 2].drop(['sales']+list(set(train.columns).intersection(drop)), axis=1)

val_0_y = val[val['kmeans'] == 0]['sales']
val_1_y = val[val['kmeans'] == 1]['sales']
val_2_y = val[val['kmeans'] == 2]['sales']

val_0 = val[val['kmeans'] == 0].drop(['sales']+list(set(val.columns).intersection(drop)), axis=1)
val_1 = val[val['kmeans'] == 1].drop(['sales']+list(set(val.columns).intersection(drop)), axis=1)
val_2 = val[val['kmeans'] == 2].drop(['sales']+list(set(val.columns).intersection(drop)), axis=1)

random_search(model_xgb, xgb_params, train_0, train_0_y, val_0, val_0_y)

# best params load
best_params = joblib.load('model_best_params.pkl')
model_lgbm = LGBMRegressor(boosting_type='gbdt',params=best_params)

