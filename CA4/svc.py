# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:27:17 2021

@author: Nida

CA4

SVC linear kernel and rbf kernel
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer


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

#Impute with mean
imp_mean = SimpleImputer(strategy='mean') 
mean_imputed=imp_mean.fit_transform(data)
mean_imputed=pd.DataFrame(mean_imputed,columns=column_names)

#Impute with most frequent values
imp_freq = SimpleImputer(strategy='most_frequent')
freq_imputed=imp_freq.fit_transform(data)
freq_imputed=pd.DataFrame(freq_imputed,columns=column_names)

# Make the pipeline
pipe_svc=make_pipeline(StandardScaler(),SVC(random_state=1))

# test for these values for C and gamma
c_val=[10**x for x in range(-3,4)]
gamma_val=[10**x for x in range(-3,3)]

param_grid   = [{'svc__C': c_val, 'svc__kernel': ['linear']},
            {'svc__C': c_val, 'svc__gamma': gamma_val, 'svc__kernel': ['rbf']}]

inner_segments = 5
outer_segments = 5

gs = GridSearchCV(estimator=pipe_svc, 
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
X_mean=mean_imputed[mean_imputed.columns[:6]]
scores_4=cross_val_score(gs,X_mean,y,scoring='accuracy',cv=outer_segments)

X_freq=freq_imputed[freq_imputed.columns[:6]]
scores_5=cross_val_score(gs,X_freq,y,scoring='accuracy',cv=outer_segments)


"""
CV accuracy: 


"""
print('CV accuracy for drop_na: %.3f +/- %.3f' % (np.mean(scores_1),
                                      np.std(scores_1)))
print('CV accuracy for knn imputed: %.3f +/- %.3f' % (np.mean(scores_2),
                                                      np.std(scores_2)))
print('CV accuracy for iterative imputed: %.3f +/- %.3f' % (np.mean(scores_3),
                                                            np.std(scores_3)))
print('CV accuracy for mean imputed: %.3f +/- %.3f' % (np.mean(scores_4),
                                                            np.std(scores_4)))
print('CV accuracy for frequent imputed: %.3f +/- %.3f' % (np.mean(scores_5),
                                                            np.std(scores_5)))

"""
CV accuracy for drop_na: 0.875 +/- 0.014
CV accuracy for knn imputed: 0.870 +/- 0.011
CV accuracy for iterative imputed: 0.885 +/- 0.018


CV accuracy for mean imputed: 0.872 +/- 0.014
CV accuracy for frequent imputed: 0.867 +/- 0.011
"""