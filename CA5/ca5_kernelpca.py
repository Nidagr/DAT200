# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:22:53 2021

@author: Nida

KernelPCA in combination with the models.

best r2 score: 0.6662802527312128
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
from sklearn.decomposition import KernelPCA

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
# The number of components we want to test KernelPCA for
n_comp = [2,3,4,5,6]


# Split in training and test set
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

pipe = make_pipeline(StandardScaler(),KernelPCA(random_state=0),
                     LinearRegression())


param_grid={'kernelpca__n_components':n_comp,'kernelpca__kernel':
            ['linear','rbf','poly','cosine','sigmoid','precomputed']}


gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='r2', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train)
y_pred=gs.predict(X_test)
print('r2 score: ', r2_score(y_test,y_pred))

"""
r2 score:  0.5843884112126231
"""

# Linear regression with LASSO

pipe = make_pipeline(StandardScaler(),KernelPCA(random_state=0),Lasso())
alpha = np.arange(0.5,3.1,0.1)
param_grid={'kernelpca__n_components':n_comp,'kernelpca__kernel':
            ['linear','rbf','poly','cosine','sigmoid','precomputed'],
            'lasso__alpha':alpha}
    
gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='r2', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train,y_train)
y_pred = gs.predict(X_test)
print('r2 score:',r2_score(y_test,y_pred))

"""
r2 score: 0.5821984418218642
"""

pipe = make_pipeline(StandardScaler(),KernelPCA(random_state=0),Ridge())


param_grid={'kernelpca__n_components':n_comp,'kernelpca__kernel':
            ['linear','rbf','poly','cosine','sigmoid','precomputed'],
            'ridge__alpha':[0.5,1,1.5,10,15,20,25,30,50,100,150,200,250]}
gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='r2', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train,y_train)

y_pred = gs.predict(X_test)

print('r2 score:',r2_score(y_test,y_pred))

"""
r2 score: 0.5831289531608413
"""
# Linear regression with Elastic Net.
pipe = make_pipeline(StandardScaler(),KernelPCA(random_state=0),ElasticNet())
param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
param_range2 = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
param_grid={'kernelpca__n_components':n_comp,'kernelpca__kernel':
            ['linear','rbf','poly','cosine','sigmoid','precomputed'],
            'elasticnet__alpha': param_range, 
                'elasticnet__l1_ratio': param_range2}


gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='r2', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train,y_train)

y_pred = gs.predict(X_test)

print('r2 score:',r2_score(y_test,y_pred))


"""
r2 score: 0.5835544136241804
"""
pipe = make_pipeline(StandardScaler(),KernelPCA(random_state=0),
                     DecisionTreeRegressor(random_state=1))
d = [2,3,4,5,6,7,8,9,10]

param_grid={'kernelpca__n_components':n_comp,'kernelpca__kernel':
            ['linear','rbf','poly','cosine','sigmoid','precomputed'],
            'decisiontreeregressor__max_depth':d}

gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='r2', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train,y_train)

y_pred = gs.predict(X_test)

print('r2 score:',r2_score(y_test,y_pred))

"""
r2 score: 0.5043041586404535
"""



pipe = make_pipeline(StandardScaler(),KernelPCA(random_state=0),
                     RandomForestRegressor(random_state=1))

n_trees = [10,50,100,150]
param_grid={'kernelpca__n_components':n_comp,'kernelpca__kernel':
            ['linear','rbf','poly','cosine','sigmoid','precomputed'],
            'randomforestregressor__max_depth': d,
            'randomforestregressor__n_estimators':n_trees}
    
gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='r2', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train,y_train)

y_pred = gs.predict(X_test)

print('r2 score:',r2_score(y_test,y_pred))

"""
r2 score: 0.6289231972425622
"""
# Polynomial model

pipe = make_pipeline(StandardScaler(),KernelPCA(random_state=0),
                     PolynomialFeatures(),LinearRegression())
param_grid={'kernelpca__n_components':n_comp,'kernelpca__kernel':
            ['linear','rbf','poly','cosine','sigmoid','precomputed'],
            'polynomialfeatures__degree':[2,3,4]}

gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='r2',
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train,y_train)
y_pred = gs.predict(X_test)

print('r2 score:',r2_score(y_test,y_pred))

"""
r2 score: 0.6662802527312128
"""

