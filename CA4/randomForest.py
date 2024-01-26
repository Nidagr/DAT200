# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:26:14 2021

@author: Nida

CA4 Random Forests
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

train_data=pd.read_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA4\\train.csv')
X=train_data[train_data.columns[:6]]
y=train_data[train_data.columns[6]]
column_names=list(train_data.columns)
# change string values for y to be 0,1,2
le = LabelEncoder()
y = le.fit_transform(y)
le.transform(['non_habitable', 'potentially_habitable','very_habitable'])

data=X
data['target']=y
"""
We try three different ways of handling missing values
- drop rows with missing values
- KNNImputer
- IterativeImputer

"""

# Drop rows with missing values
drop_na=data.dropna(axis=0)

# Impute missing values using the KNNImputer
# the dataset is large, hence I choose a large number of neighbors to impute 
# from

knn=KNNImputer(n_neighbors=20)
knn_imputed=knn.fit_transform(data)
knn_imputed=pd.DataFrame(knn_imputed,columns=column_names)

# Impute missing values using IterativeImputer
it=IterativeImputer(random_state=1)
iterative_imputed=it.fit_transform(data)
iterative_imputed=pd.DataFrame(iterative_imputed,columns=column_names)

RF_pipe = make_pipeline(RandomForestClassifier(random_state=1))
param_grid=[{'randomforestclassifier__criterion':['gini','entropy'],
    'randomforestclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None],
    'randomforestclassifier__n_estimators':[25,50,75,100,125]}]

inner_segments = 5
outer_segments = 5

gs = GridSearchCV(estimator=RF_pipe, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=inner_segments,
                  n_jobs=-1)

# for the dataset where we dropped rows with missing vlaues
X_na=drop_na[drop_na.columns[:6]]
y_na=drop_na[drop_na.columns[6]]
#outer loop
scores_1 = cross_val_score(gs, X_na, y_na, 
                         scoring='accuracy', cv=outer_segments)

"""
CV accuracy: 

Try with the knn imputed dataset.
"""
X_knn=knn_imputed[knn_imputed.columns[:6]]
scores_2 = cross_val_score(gs, X_knn, y, 
                          scoring='accuracy', cv=outer_segments)

"""
CV accuracy:

Try with iterative imputed dataset.
"""
X_iter=iterative_imputed[iterative_imputed.columns[:6]]
scores_3 = cross_val_score(gs, X_iter, y, 
                          scoring='accuracy', cv=outer_segments)


"""
CV accuracy: 


"""
print('CV accuracy for drop_na: %.3f +/- %.3f' % (np.mean(scores_1),
                                      np.std(scores_1)))
print('CV accuracy for knn imputed: %.3f +/- %.3f' % (np.mean(scores_2),
                                                      np.std(scores_2)))
print('CV accuracy for iterative imputed: %.3f +/- %.3f' % (np.mean(scores_3),
                                                            np.std(scores_3)))
"""
CV accuracy for drop_na: 0.894 +/- 0.010
CV accuracy for knn imputed: 0.883 +/- 0.018
CV accuracy for iterative imputed: 0.901 +/- 0.018
"""