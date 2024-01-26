# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 09:50:22 2021

@author: Nida

Regression models in combination with PCA.

We got the best estimate for r2 score 0.666280241201215 when using PCA
with polynomial regression model.
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
from sklearn.decomposition import PCA

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
# The number of components we want to test PCA for
n_comp = [2,3,4,5,6]

# Split in training and test set
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


# PCA with Linear Regression

# pipe = make_pipeline(StandardScaler(),PCA(random_state=0),LinearRegression())

# param_grid = {'pca__n_components':n_comp}

# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(X_train, y_train)
# y_pred=gs.predict(X_test)
# print('r2 score: ', r2_score(y_test,y_pred))
"""
r2 score:  0.5619602575028693
"""

# PCA with LASSO 
# pipe = make_pipeline(StandardScaler(), PCA(random_state=0),Lasso())
# alpha = np.arange(0.5,3.1,0.1)
# param_grid = {'pca__n_components':n_comp,'lasso__alpha':alpha}


# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(X_train, y_train)
# y_pred=gs.predict(X_test)
# print('r2 score: ', r2_score(y_test,y_pred))

"""
r2 score:  0.5616444132202846
"""

# PCA with Ridge
# pipe = make_pipeline(StandardScaler(), PCA(random_state=0),Ridge())

# param_grid = {'pca__n_components':n_comp,
#               'ridge__alpha':[0.5,1,1.5,10,15,20,25,30,50,100,150,200,250]}

# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(X_train, y_train)
# y_pred=gs.predict(X_test)
# print('r2 score: ', r2_score(y_test,y_pred))

"""
r2 score:  0.5618781234429253

Exactly same as LASSO. Makes sense as PCA makes new components maybe not 
removed by LASSO.
"""

# PCA with Elastic Net

# pipe = make_pipeline(StandardScaler(), PCA(random_state=0),ElasticNet())
# param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
# param_range2 = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

# param_grid = {'pca__n_components':n_comp,
#               'elasticnet__alpha': param_range, 
#                'elasticnet__l1_ratio': param_range2}

# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(X_train, y_train)
# y_pred=gs.predict(X_test)
# print('r2 score: ', r2_score(y_test,y_pred))
"""
r2 score:  0.561387311291281
"""

# PCA with Decision Tree Regression


# pipe = make_pipeline(StandardScaler(), PCA(random_state=0),DecisionTreeRegressor())
d = [2,3,4,5,6,7,8,9,10]
# param_grid = {'pca__n_components':n_comp,'decisiontreeregressor__max_depth':d}


# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(X_train, y_train)
# y_pred=gs.predict(X_test)
# print('r2 score: ', r2_score(y_test,y_pred))

"""
r2 score:  0.38713314228032214
"""

# PCA with Random Forest
# pipe = make_pipeline(StandardScaler(), PCA(random_state=0),RandomForestRegressor())
# param_grid = {'pca__n_components':n_comp,'randomforestregressor__max_depth':d,
#               'randomforestregressor__n_estimators':[10,50,100,150]}

# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(X_train, y_train)
# y_pred=gs.predict(X_test)
# print('r2 score: ', r2_score(y_test,y_pred))
"""
r2 score:  0.5832489111294795
"""

# PCA with polynomial regression
pipe = make_pipeline(StandardScaler(), PCA(random_state=0),
                     PolynomialFeatures(),LinearRegression())


param_grid={'pca__n_components':n_comp, 'polynomialfeatures__degree':[2,3,4]}


gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='r2', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train)
y_pred=gs.predict(X_test)
print('r2 score: ', r2_score(y_test,y_pred))

"""
r2 score:  0.666280241201215
"""





