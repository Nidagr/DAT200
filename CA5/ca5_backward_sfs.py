# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 10:44:21 2021

@author: Nida

The linear regression models in combination with datasetwith only most 
important features. We try backward sequential feature
selection with linear regression.

best r2 score: 0.6929599980299123 with random forest


Think it might be better to use LASSO to select features as this is the model
that got the best r2score when performing regression directly on the 
preprocessed data.

best r2 score: 0.7038797302476771 with random forest
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt

#--------------------------The data-------------------------------------------
data = pd.read_pickle('train4.pkl')
data.replace('missing', np.nan, inplace=True)
# make sure data is numeric
objects=[]
for i in data.columns:
    if data.loc[:,i].dtype == object:
        objects.append(i)

data[objects] = data[objects].apply(pd.to_numeric, errors='ignore')

# test data
test = pd.read_pickle('test4.pkl')
test = test.replace('missing',np.nan)
# make sure data is numeric
ob=[]
for i in test.columns:
    if test.loc[:,i].dtype == object:
        ob.append(i)

test[ob] = test[ob].apply(pd.to_numeric, errors='ignore')

# remove the features from training and test data that had 930 missing values
liste = data.isnull().sum()
remove=[]
for i, val in enumerate(liste):
    if val > 1:
        remove.append('demo_{}'.format(i))

data = data.drop(remove,axis=1)
# we had one missing value in column demo_25, replace with mean of column
data=data.fillna(data.mean())

l = test.isnull().sum()
re=[]
for i, val in enumerate(l):
    if val > 1: 
        re.append('demo_{}'.format(i))

test = test.drop(re,axis=1)

#---------------------------------------------------------------------------


# Split in training and test set
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


lr = make_pipeline(StandardScaler(),Lasso(alpha=3))
#Start the sequential feature selector. It is backward selection (forward=False), removes one by one until we are left with
#one feature(k_features=1). Use our model svc to choose features, and choose the features based on accuracy (scoring)
sfs = SFS(lr, 
           k_features=1, 
           forward=False, 
           floating=False, 
           verbose=0,
           scoring='r2',
           cv=10)
# Fit the models
sfs = sfs.fit(X_train, y_train)

# This dictionary contains results from all compuations, that is, metrics from models with 8 until 1 features
metricDict = sfs.get_metric_dict()

# # Initialise plot
# fig = plot_sfs(sfs.get_metric_dict(), kind='std_dev',figsize=(14,6))


# plt.title('Sequential Backward Selection (w. StdDev)')
# plt.grid()
# plt.show()
# ma=0
# number_features=103
# for num,dic in metricDict.items():
#     if dic['avg_score']>ma:
#         ma = dic['avg_score']
#         number_features = num

# print(number_features,ma)

# for num_features,dic in metricDict.items():
#     print('Number of features: ',num_features,',score: ',dic['avg_score'])

"""
If you print out the average scores and examine the plot closely, we see that
we get the best average score when using 36 features. Note that the average 
score does not differ much for having 10 more or less features. 

In the version with LASSO in SFS we get the best avg_score for 21 features.
Number of features:  21 ,score:  0.6961660948265187
Use this


When chossing features based on entire dataset we get the best score for 14
features. The scores estimates got better here, but worse on kaggle. Seems
like it overfits the data, which makes sense, we stick with the features
selected on part of the data.
"""

f = list(metricDict[21]['feature_idx'])

sfs_X_train = X_train.iloc[:,f]
sfs_X_test = X_test.iloc[:,f]


# # Linear regression
# pipe = make_pipeline(StandardScaler(),LinearRegression())

# gs = pipe.fit(sfs_X_train, y_train)

# y_pred=gs.predict(sfs_X_test)

# print('r2 score: ', r2_score(y_test,y_pred))

"""
r2 score:  0.651161483787867

lasso version
r2 score:  0.6484004858599917
"""
# LASSO model

# pipe = make_pipeline(StandardScaler(),Lasso())
# alpha = np.arange(0.5,3.1,0.1)
# param_grid = {'lasso__alpha':alpha}
# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(sfs_X_train,y_train)

# y_pred = gs.predict(sfs_X_test)

# print('r2 score:',r2_score(y_test,y_pred))


# reg = gs.best_estimator_
# # only include 21 features
# sfs_X = X.iloc[:,f]
# sfs_test = test.iloc[:,f]
# reg.fit(sfs_X,y)
# predicted = reg.predict(sfs_test)
# # # dataframe of predictions
# id_col=[i for i in range(0,sfs_test.shape[0])]
# dict_val={'Id': id_col,'Predicted': predicted}
# results=pd.DataFrame(dict_val)
# results=results.set_index('Id')
# results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA5\\results_04.csv')
"""
r2 score: 0.6585517009994151

lasso ver
r2 score: 0.6578313948846529

14 features 
r2 score: 0.7042263541102736
"""

# Ridge model


# pipe = make_pipeline(StandardScaler(),Ridge())

# param_grid={'ridge__alpha':[0.5,1,1.5,10,15,20,25,30,50,100,150,200,250]}

# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(sfs_X_train,y_train)

# y_pred = gs.predict(sfs_X_test)

# print('r2 score:',r2_score(y_test,y_pred))
"""
r2 score: 0.6533435411492612

lasso ver
r2 score: 0.661271754675643

14 features
r2 score: 0.7040777630332128
"""

# Elastic Net

# pipe = make_pipeline(StandardScaler(),ElasticNet())
# param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
# param_range2 = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
# param_grid = [{'elasticnet__alpha': param_range, 
#                 'elasticnet__l1_ratio': param_range2}]


# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(sfs_X_train,y_train)

# y_pred = gs.predict(sfs_X_test)

# print('r2 score:',r2_score(y_test,y_pred))

"""
r2 score: 0.6511646985928126

lasso ver
r2 score: 0.6637636319302347

14 features
r2 score: 0.7040145333447374
"""
# Decision Tree regressor

# pipe = DecisionTreeRegressor(random_state=1)
# d = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# param_grid={'max_depth':d}
# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(sfs_X_train,y_train)

# y_pred = gs.predict(sfs_X_test)

# print('r2 score:',r2_score(y_test,y_pred))
"""
r2 score: 0.4971744464952669

lasso ver
r2 score: 0.4770590342810411

14 features
r2 score: 0.3917112875872687
"""
# Random forest regressor

# pipe = RandomForestRegressor(random_state=1)

# n_trees = [10,50,100,150]

# param_grid={'max_depth': d,'n_estimators':n_trees}
# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(sfs_X_train,y_train)

# y_pred = gs.predict(sfs_X_test)

# print('r2 score:',r2_score(y_test,y_pred))

# reg = gs.best_estimator_
# # only include 21 features
# sfs_X = X.iloc[:,f]
# sfs_test = test.iloc[:,f]
# reg.fit(sfs_X,y)
# predicted = reg.predict(sfs_test)
# # # dataframe of predictions
# id_col=[i for i in range(0,sfs_test.shape[0])]
# dict_val={'Id': id_col,'Predicted': predicted}
# results=pd.DataFrame(dict_val)
# results=results.set_index('Id')
# results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA5\\results_03.csv')


"""
r2 score: 0.6929599980299123

lasso ver
r2 score: 0.7038797302476771

selecting from all of training data:
    r2 score: 0.6944050959221091
"""
# Polynomial model



# pipe = make_pipeline(StandardScaler(),PolynomialFeatures(),LinearRegression())

# param_grid={'polynomialfeatures__degree':[2,3,4]}

# gs = GridSearchCV(estimator=pipe,
#                   param_grid=param_grid,
#                   scoring='r2',
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(sfs_X_train,y_train)
# y_pred = gs.predict(sfs_X_test)

# print('r2 score:',r2_score(y_test,y_pred))
"""
r2 score: -6.703220498588623

lasso ver
r2 score: -13.84035933812451

14 features
r2 score: 0.618940362569981
"""
