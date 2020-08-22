import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='NanumBarunGothic')
from matplotlib import font_manager
import datetime
from sklearn.model_selection import cross_val_score
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

## data load 
perform = pd.read_csv('data/2019_performance.csv')
rating = pd.read_csv('data/2019_rating.csv')
test = pd.read_csv('data/question.csv')

## detect outlier
"""
x : dataset
p : ratio of deleted outlier(0 ~ 1)
return : dataset that dropped outlier
"""
def del_outlier(x,p):
    clf=IsolationForest(contamination=float(p),
                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=2020, verbose=0)
    tem = x['취급액'].reset_index()
    clf.fit(tem)
    tem['anomaly'] = clf.predict(tem)
    sns.boxplot(tem[tem['anomaly']==1]['취급액'])
    dist(tem[tem['anomaly']==1]['취급액'])
    return train[tem['anomaly']==1]
