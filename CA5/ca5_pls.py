# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 10:32:45 2021

@author: Nida

Regression models in combination with PLS.

PLS is compresion of data and then least squares regression (OLS) on the 
new data. So we do not check pls in combination with other models, not like
with PCA.

Got score 0.6661015913181108.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression


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

# Do not need to include StandardScaler as PLS scales data for you when 
# scale=True
pipe = PLSRegression(scale=True)


param_grid = {'n_components':[2,3,4,5,6]}


gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='r2', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train)
y_pred=gs.predict(X_test)
print('r2 score: ', r2_score(y_test,y_pred))

"""
r2 score:  0.6661015913181108

If scale = False then:
r2 score:  0.37979775132882965
"""




















