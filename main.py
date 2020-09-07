from util import  load_data, preprocess, mk_trainset, metric
from clustering import clustering
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from dim_reduction import train_AE,by_AE,by_PCA

def boosting(X,y,col_sample=0.6,lr=0.04,iter=1500):
    train_features, test_features, train_labels, test_labels = train_test_split(X, y,random_state=2020)
    model_lgb = LGBMRegressor(subsample= 0.7, colsample_bytree= col_sample, learning_rate=lr,n_estimators=iter,random_state=2020)
    model_lgb.fit(train_features,train_labels,early_stopping_rounds = 200,eval_set = [(test_features,test_labels)],verbose=False)
    pred_lgb = model_lgb.predict(test_features)
    return metric(test_labels,pred_lgb), len(test_labels)


def predict(data,y,k):
    for i in range(k):
        globals()[f'X_train_c{i}'] = X_train[X_train['kmeans']==i]
        globals()[f'X_test_c{i}'] = X_test[X_test['kmeans']==i]
    """
    predict '취급액' score only using train set(perform)
    return : RMAE score for each cluster
    """
    origin, originlen = boosting(data.drop(['id'],axis=1).iloc[:34317,:],y['sales'])
    c0, len0 = boosting(X_train_c0.drop(['sales','kmeans','id'],axis=1),X_train_c0['sales'])
    c1, len1 = boosting(X_train_c1.drop(['sales','kmeans','id'],axis=1),X_train_c1['sales'])
    c2, len2 =  boosting(X_train_c2.drop(['sales','kmeans','id'],axis=1),X_train_c2['sales'])
    c3, len3 =  boosting(X_train_c3.drop(['sales','kmeans','id'],axis=1),X_train_c3['sales'])

    total_error = (c0 * len0 + c1 * len1 + c2 * len2 + c3 *len3)/(len0+len1+len2+len3)

    print(f'origin error : {round(origin,2)}%\n\nCluster_0 : {round(c0,2)}%\nCluster_1 : {round(c1,2)}%\nCluster_2 : {round(c2,2)}%\nCluster_3 : {round(c3,2)}%\n\nTotal error : {round(total_error,2)}%')

# excution
if __name__=='__main__': 
    data_path = 'data/'
    perform_raw, rating, test = load_data(data_path)
    data, y, y_km = preprocess(perform_raw,test,0.03,4)
    data = mk_trainset(data)

    # lgb, ensemble = modeling(data,y_km)  # only use for tunning cluster model
    
    # train_AE(data,30,40) # only use for training AE
    
    #data = by_AE(data,'AE') 
    data = by_PCA(data,0.9)

    X_train, X_test = clustering(data,y_km,y)
    predict(data,y,4)

"""
<cluster 정확도 cv>

kfold = KFold(n_splits=5,shuffle=True,random_state=2020)
res = cross_val_score(lgb,X_train,y_km,cv=kfold)
print(f'Accuracy : {res}\nAverage acc : {np.array(res).mean()}')


<시각화>

plt.rcParams["figure.figsize"] = (10,6)

sns.boxplot(perform_raw['취급액'])

sns.boxplot(y)

Z = y.rename(columns={'sales':'취급액'}).reset_index()
km = KMeans(n_clusters=3)
km.fit(Z)
df = pd.DataFrame(Z)
df['kmeans'] = km.labels_
colormap = { 0: 'red', 1: 'green', 2: 'blue'}
colors = df.apply(lambda row: colormap[row.kmeans], axis=1)
ax = df.plot(kind='scatter', x=0, y=1, alpha=0.1, s=300, c=colors)

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(Z)
    kmeanModel.fit(Z)
    distortions.append(sum(np.min(cdist(Z, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / Z.shape[0])
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

sns.boxplot(X_train_c0['sales'])
plt.title('Cluster 0')
plt.xlabel('취급액')
sns.boxplot(X_train_c1['sales'])
plt.title('Cluster 1')
plt.xlabel('취급액')
sns.boxplot(X_train_c2['sales'])
plt.title('Cluster 2')
plt.xlabel('취급액')
"""