# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:13:19 2021

@author: Nida

Best pipeline with kernel
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# -------- importing the data and imputing with IterativeImputer------------
train_data=pd.read_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA4\\train.csv')
X=train_data[train_data.columns[:6]]
y=train_data[train_data.columns[6]]
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_
le.transform(['non_habitable', 'potentially_habitable','very_habitable'])
test=pd.read_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA4\\test.csv')

df = pd.DataFrame(X)
df['target']=y

it=IterativeImputer(random_state=1)
imputed_data=it.fit_transform(df)
column_names=list(df.columns)
imputed_data=pd.DataFrame(imputed_data,columns=column_names)
# overwrite X to be the version where missing values are imputed
X=imputed_data[imputed_data.columns[:6]]

column_names=list(df.columns)
imputed_data=pd.DataFrame(imputed_data,columns=column_names)
test_data=it.fit_transform(test)
test_data=pd.DataFrame(test_data,columns=column_names[:-1])

# --------------------------------------------------------------------------

inner_segments = 5
outer_segments = 5


# pipe_svc_l = make_pipeline(StandardScaler(),SVC(kernel='linear',random_state=1))
# c=[10**x for x in range(-3,4)]
# params_1=[{'svc__C':c}]
# gs_svc = GridSearchCV(estimator=pipe_svc_l,
#                       param_grid=params_1,
#                       scoring='accuracy',
#                       cv=inner_segments,
#                       n_jobs=-1)

# scores_3 = cross_val_score(gs_svc, X, y, scoring='accuracy', cv=outer_segments)


# pipe_svc_r = make_pipeline(StandardScaler(),SVC(kernel='rbf',random_state=1))

# gamma_val=[10**x for x in range(-3,3)]

# params_2=[{'svc__C':c,'svc__gamma':gamma_val}]

# gs_svc_r = GridSearchCV(estimator=pipe_svc_r,
#                       param_grid=params_2,
#                       scoring='accuracy',
#                       cv=inner_segments,
#                       n_jobs=-1)

# scores_4 = cross_val_score(gs_svc_r, X, y, scoring='accuracy', cv=outer_segments)

# print('CV accuracy for svc linear kernel: %.3f +/- %.3f' % (np.mean(scores_3),
#                                       np.std(scores_3)))
# print('CV accuracy for svc rbf kernel: %.3f +/- %.3f' % (np.mean(scores_4),
#                                       np.std(scores_4)))

"""
CV accuracy for svc linear kernel: 0.867 +/- 0.024
CV accuracy for svc rbf kernel: 0.885 +/- 0.018

We got the best estimate for the modle with the rbf kernel. Lets try to include 
PCA.

"""


# pipe = make_pipeline(StandardScaler(), PCA(random_state=1), 
#                       SVC(kernel='rbf',random_state=1))
gamma_val=[10**x for x in range(-3,3)]
c=[10**x for x in range(-3,4)]
# params=[{'pca__n_components':[1,2,3,4,5,6],'svc__gamma':gamma_val,'svc__C':c}]

# gs = GridSearchCV(estimator=pipe,
#                       param_grid=params,
#                       scoring='accuracy',
#                       cv=inner_segments,
#                       n_jobs=-1)
# score_pca=cross_val_score(gs,X,y,scoring='accuracy',cv=outer_segments)
# print('CV accuracy for svc rbf kernel and pca: %.3f +/- %.3f' % (np.mean(score_pca),
#                                       np.std(score_pca)))

"""
CV accuracy for svc rbf kernel and pca: 0.887 +/- 0.014

Now try for kernel PCA.
"""

# pipe_kernel = make_pipeline(StandardScaler(), KernelPCA(random_state=1), 
#                       SVC(kernel='rbf',random_state=1))

# params_kernel=[{'kernelpca__n_components':[1,2,3,4,5,6],
#                 'svc__gamma':gamma_val,'svc__C':c, 
#                 'kernelpca__kernel':['linear','rbf','poly','cosine','sigmoid',
#                            'precomputed']}]

# gs_kernel = GridSearchCV(estimator=pipe_kernel,
#                       param_grid=params_kernel,
#                       scoring='accuracy',
#                       cv=inner_segments,
#                       n_jobs=-1)
# score_kernelpca=cross_val_score(gs_kernel,X,y,scoring='accuracy',cv=outer_segments)
# print('CV accuracy for svc rbf kernel and kernel pca: %.3f +/- %.3f' % (np.mean(score_kernelpca),
#                                       np.std(score_kernelpca)))
"""
CV accuracy for svc rbf kernel and kernel pca: 0.896 +/- 0.016

kernel PCA with Random Forest can be found in file preprcess.py, got
the accuracy estimate:
    CV accuracy: 0.895 +/- 0.016
    
kernel PCA with KNN.
Takes too long with nested, lets do normal gridsearch
"""

knn_pipe = make_pipeline(StandardScaler(),KernelPCA(random_state=1),
                          KNeighborsClassifier(metric='minkowski'))
k_values=range(9,16)
p_values=range(1,11)

param_knn=[{'kernelpca__n_components':[1,2,3,4,5,6],
            'kneighborsclassifier__n_neighbors': k_values,
            'kneighborsclassifier__p': p_values,
            'kernelpca__gamma':[0.3,0.4,0.5,0.6],
            'kernelpca__kernel':['linear','rbf','poly','cosine','sigmoid',
                           'precomputed']}]

# gs_knn = GridSearchCV(estimator=knn_pipe,
#                       param_grid=param_knn,
#                       scoring='accuracy',
#                       cv=inner_segments,
#                       n_jobs=-1)
# score_knn = cross_val_score(gs_knn,X,y,scoring='accuracy',cv=outer_segments)

# print('CV accuracy for knnl and kernel pca: %.3f +/- %.3f' % (np.mean(score_knn),
#                                       np.std(score_knn)))
"""
CV accuracy for knn and kernel pca: 0.889 +/- 0.017
"""








