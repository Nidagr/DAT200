# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:07:50 2021

@author: Nida

Lets do sequential forward selection with LASSO regression (best model in
our regression directly on preprocessed data pipeline). 


Best r2 score: 0.7003944381657514 for random forest.
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
#Start the sequential feature selector. It is forward selection (forward=True), 
# removes one by one until we are left with
#all features (k_features=102). Use our linear to choose features, and choose 
# the features based on r2 (scoring)
sfs = SFS(lr, 
           k_features=102, 
           forward=True, 
           floating=False, 
           verbose=0,
           scoring='r2',
           cv=10)
# Fit the models
sfs = sfs.fit(X_train, y_train)

# This dictionary contains results from all compuations, that is, metrics from models with 8 until 1 features
metricDict = sfs.get_metric_dict()

# Initialise plot
# fig = plot_sfs(sfs.get_metric_dict(), kind='std_dev',figsize=(14,6))


# plt.title('Sequential Backward Selection (w. StdDev)')
# plt.grid()
# plt.show()


for num_features,dic in metricDict.items():
    print('Number of features: ',num_features,',score: ',dic['avg_score'])
"""
Printing the above shows that we get the best avg_score when using 32 features.
"""

f = list(metricDict[32]['feature_idx'])

sfs_X_train = X_train.iloc[:,f]
sfs_X_test = X_test.iloc[:,f]


# Linear regression
pipe = make_pipeline(StandardScaler(),LinearRegression())

gs = pipe.fit(sfs_X_train, y_train)

y_pred=gs.predict(sfs_X_test)

print('r2 score: ', r2_score(y_test,y_pred))

"""
r2 score:  0.6633554650657565
"""
# LASSO model

pipe = make_pipeline(StandardScaler(),Lasso())
alpha = np.arange(0.5,3.1,0.1)
param_grid = {'lasso__alpha':alpha}
gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='r2', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(sfs_X_train,y_train)

y_pred = gs.predict(sfs_X_test)

print('r2 score:',r2_score(y_test,y_pred))


"""
r2 score: 0.6691020626843105
"""
# Ridge model


pipe = make_pipeline(StandardScaler(),Ridge())

param_grid={'ridge__alpha':[0.5,1,1.5,10,15,20,25,30,50,100,150,200,250]}

gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='r2', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(sfs_X_train,y_train)

y_pred = gs.predict(sfs_X_test)

print('r2 score:',r2_score(y_test,y_pred))

"""
r2 score: 0.6746562747083799
"""


# Elastic Net

pipe = make_pipeline(StandardScaler(),ElasticNet())
param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
param_range2 = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
param_grid = [{'elasticnet__alpha': param_range, 
                'elasticnet__l1_ratio': param_range2}]


gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='r2', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(sfs_X_train,y_train)

y_pred = gs.predict(sfs_X_test)

print('r2 score:',r2_score(y_test,y_pred))

"""
r2 score: 0.6754359008865889
"""
# Decision Tree regressor

pipe = DecisionTreeRegressor(random_state=1)
d = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]

param_grid={'max_depth':d}
gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='r2', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(sfs_X_train,y_train)

y_pred = gs.predict(sfs_X_test)

print('r2 score:',r2_score(y_test,y_pred))


"""
r2 score: 0.5219099571196113
"""
# Random forest regressor

pipe = RandomForestRegressor(random_state=1)

n_trees = [10,50,100,150]

param_grid={'max_depth': d,'n_estimators':n_trees}
gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='r2', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(sfs_X_train,y_train)

y_pred = gs.predict(sfs_X_test)

print('r2 score:',r2_score(y_test,y_pred))

"""
r2 score: 0.7003944381657514
"""
# Polynomial model



pipe = make_pipeline(StandardScaler(),PolynomialFeatures(),LinearRegression())

param_grid={'polynomialfeatures__degree':[2,3,4]}

gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='r2',
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(sfs_X_train,y_train)
y_pred = gs.predict(sfs_X_test)

print('r2 score:',r2_score(y_test,y_pred))

"""
r2 score: -32.64784797907673
"""
