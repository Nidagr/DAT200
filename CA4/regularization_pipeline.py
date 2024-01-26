# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 13:08:26 2021

@author: Nida

Find and optimize pipeline with regularization

Logistic regression l1 and l2 or svc
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
from sklearn.decomposition import PCA

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
df=X
df['target']=y
# --------------------------------------------------------------------------
"""
We will no do nested cv on the three models, try to get estimate of which
model perform best. Start with Logistic Regression with l1 regularization.
"""

# inner_segments = 5
# outer_segments = 5

# pipe_lr=make_pipeline(StandardScaler(),
#                       LogisticRegression(solver = 'saga', multi_class='auto', 
#                                           random_state=1))
# c_val=[10**x for x in range(-5,5)]
# param_grid={'logisticregression__C':c_val,'logisticregression__penalty':['l1','l2']}
# #inner loop finding best hyperparameter
# gs_lr = GridSearchCV(estimator=pipe_lr, 
#                   param_grid=param_grid, 
#                   scoring='accuracy', 
#                   cv=inner_segments,
#                   n_jobs=-1)

# #outer loop
# scores = cross_val_score(gs_lr, X, y, scoring='accuracy', cv=outer_segments)


# print('CV accuracy for logistic regression: %.3f +/- %.3f' % (np.mean(scores),
#                                       np.std(scores)))
"""
CV accuracy for logistic regression: 0.866 +/- 0.026



Let us now try and include PCA to see if nested CV gives better estimate.
"""
# pipe_pca=make_pipeline(StandardScaler(),PCA(random_state=1),
#                       LogisticRegression(solver = 'saga', multi_class='auto', 
#                                           random_state=1))

# param_grid_pca={'logisticregression__C':c_val,'logisticregression__penalty':['l1','l2'],
#             'pca__n_components':[1,2,3,4,5,6]}
# #inner loop finding best hyperparameter
# gs_pca = GridSearchCV(estimator=pipe_pca, 
#                   param_grid=param_grid_pca, 
#                   scoring='accuracy', 
#                   cv=inner_segments,
#                   n_jobs=-1)

# scores_pca = cross_val_score(gs_pca,X,y,scoring='accuracy',cv=outer_segments)

# print('CV accuracy for logistic regression with PCA:%.3f +/- %.3f' % 
#       (np.mean(scores_pca), np.std(scores_pca)))
"""
CV accuracy for logistic regression with PCA:0.864 +/- 0.025
"""

# # Perceptron also support l1 and l2 probably worse but might try just in case
# from sklearn.linear_model import Perceptron

# pipe_p=make_pipeline(StandardScaler(),Perceptron(random_state=1))
# eta_values=np.arange(0.5,1,0.01)
# n_values =np.arange(10,101,10)
# params_p =[{'perceptron__eta0':eta_values,'perceptron__max_iter':n_values,
#              'perceptron__penalty':['l1','l2']}]

# gs_p = GridSearchCV(estimator=pipe_p, 
#                   param_grid=params_p, 
#                   scoring='accuracy', 
#                   cv=inner_segments,
#                   n_jobs=-1)

# scores_p = cross_val_score(gs_p,X,y,scoring='accuracy',cv=outer_segments)
# # 0.836 +/- 0.020

#---------------------------remove outliers --------------------------------

#remove outliers based on radiation_level feature

rad = np.asarray(df[df.columns[0]])
qa1 = np.quantile(rad,0.10) #10 th percentile
qa3 = np.quantile(rad,0.90) #90 th percentile
indexes_to_remove=[]
for i in range(len(rad)):
    if rad[i]<qa1 or rad[i]>qa3:
        indexes_to_remove.append(i)      
new_df = df.drop(indexes_to_remove)

#remove outliers based on flare_prob feature

fl = np.asarray(new_df[new_df.columns[3]])
fl_indexes=new_df[new_df.columns[3]].index
ql1 = np.quantile(fl,0.10) #10 th percentile
ql3 = np.quantile(fl,0.90) #90 th percentile
indexes_to_remove_fl=[]
for i in range(len(fl)):
    if fl[i]<ql1 or fl[i]>ql3:
        indexes_to_remove_fl.append(fl_indexes[i])
new2_df = new_df.drop(indexes_to_remove_fl)

#remove outliers based on growth_potential feature

gr = np.asarray(new2_df[new2_df.columns[4]])
gr_indexes=new2_df[new2_df.columns[4]].index
s1 = np.quantile(gr,0.10) #10 th percentile
s3 = np.quantile(gr,0.90) #90 th percentile
indexes_to_remove_gr=[]
for i in range(len(gr)):
    if gr[i]<s1 or gr[i]>s3:
        indexes_to_remove_gr.append(gr_indexes[i])
new3_df = new2_df.drop(indexes_to_remove_gr)

#remove outliers based on alien_prob feature

al = np.asarray(new3_df[new3_df.columns[2]])
al_indexes=new3_df[new3_df.columns[2]].index
c1 = np.quantile(al,0.10) #10 th percentile
c3 = np.quantile(al,0.90) #90 th percentile
indexes_to_remove_al=[]
for i in range(len(al)):
    if al[i]<c1 or al[i]>c3:
        indexes_to_remove_al.append(al_indexes[i])
reduced_df = new3_df.drop(indexes_to_remove_al)

X_red=reduced_df[reduced_df.columns[:6]]
y_red=reduced_df[reduced_df.columns[6]]

inner_segments = 5
outer_segments = 5

pipe_lr=make_pipeline(StandardScaler(),
                      LogisticRegression(solver = 'saga', multi_class='auto', 
                                          random_state=1))
c_val=[10**x for x in range(-5,5)]
param_grid={'logisticregression__C':c_val,'logisticregression__penalty':['l1','l2']}
#inner loop finding best hyperparameter
gs_lr = GridSearchCV(estimator=pipe_lr, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=inner_segments,
                  n_jobs=-1)

#outer loop
scores = cross_val_score(gs_lr, X_red, y_red, scoring='accuracy', cv=outer_segments)


print('CV accuracy for logistic regression: %.3f +/- %.3f' % (np.mean(scores),
                                      np.std(scores)))

"""
CV accuracy for logistic regression: 0.929 +/- 0.012
"""