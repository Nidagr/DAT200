# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 17:41:09 2021

@author: Nida

CA4

PCA with randomforest best model til now (feature selection makes it worse).
Now try removing outliers.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# get training data
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

# Impute missing values using IterativeImputer
it=IterativeImputer(random_state=1)
iterative_imputed=it.fit_transform(data)
iterative_imputed=pd.DataFrame(iterative_imputed,columns=column_names)
X=iterative_imputed[iterative_imputed.columns[:6]]

df=pd.DataFrame(X)
df['target']=y

# impute test data
test = pd.read_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA4\\test.csv')
X_p = test[test.columns[:6]]
# missing values in test data, impute
iter_imp=it.fit_transform(X_p)
iter_imp=pd.DataFrame(iter_imp,columns=column_names[:-1])

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

"""

Violin plot to show how almost no outliers are there.
"""

# fig, axs = plt.subplots(2, 3, figsize=(10,10))
# #2 rows 3 columns
# axs[0,0].set_title('Nano-electric radiation level')
# sns.violinplot(data=reduced_df[reduced_df.columns[0]],ax=axs[0,0])

# axs[0,1].set_title('Relative atmospheric pressure')
# sns.violinplot(data=reduced_df[reduced_df.columns[1]],ax=axs[0,1])

# axs[0,2].set_title('Probability of Alien presence')
# sns.violinplot(data=reduced_df[reduced_df.columns[2]],ax=axs[0,2])

# axs[1,0].set_title('Frequenccy of dangerous solar flares')
# sns.violinplot(data=reduced_df[reduced_df.columns[3]],ax=axs[1,0])

# axs[1,1].set_title('Bio-growth potentiality score')
# sns.violinplot(data=reduced_df[reduced_df.columns[4]],ax=axs[1,1])

# axs[1,2].set_title('Identified traces of H2O on the surface')
# sns.violinplot(data=reduced_df[reduced_df.columns[5]],ax=axs[1,2])

"""
Use nested CV to get test accuracy estimate of model built on this data
"""

# pca_pipe=make_pipeline(StandardScaler(),PCA(random_state=1), 
#                         RandomForestClassifier(random_state=1))

# param_grid=[{'randomforestclassifier__criterion':['gini','entropy'],
#     'randomforestclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None],
#     'randomforestclassifier__n_estimators':[50,75,100,125,150,175,200],
#     'pca__n_components':[1,2,3,4,5,6]}]

# inner_segments = 5
# outer_segments = 5

# pca_gs = GridSearchCV(estimator=pca_pipe, 
#                   param_grid=param_grid, 
#                   scoring='accuracy', 
#                   cv=inner_segments,
#                   n_jobs=-1)
# X_red=reduced_df[reduced_df.columns[:6]]
# y_red=reduced_df[reduced_df.columns[6]]
# scores=cross_val_score(pca_gs,X_red,y_red,scoring='accuracy', cv=outer_segments)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
#                                       np.std(scores)))

"""
CV accuracy: 0.926 +/- 0.014

Which is better, however nested cv find test accuracy estimate on data where
outliers are removed. Might not perform as well when fitting on the test data.
However should check it out!
"""

# X_train, X_test, y_train, y_test = train_test_split(X_red,y_red,test_size=0.3,
#                                                     random_state=1,stratify=y_red)

# pca_pipe=make_pipeline(StandardScaler(),PCA(random_state=1), 
#                         RandomForestClassifier(random_state=1))

# param_grid=[{'randomforestclassifier__criterion':['gini','entropy'],
#     'randomforestclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None],
#     'randomforestclassifier__n_estimators':[75,100,125,150,175,200],
#     'pca__n_components':[1,2,3,4,5,6]}]


# gs = GridSearchCV(estimator=pca_pipe, 
#                   param_grid=param_grid, 
#                   scoring='accuracy', 
#                   cv=5,
#                   n_jobs=-1)
# gs = gs.fit(X_train, y_train)
# print(gs.best_score_)
# print(gs.best_params_)
"""
0.9301369863013699
{'pca__n_components': 6, 'randomforestclassifier__criterion': 'entropy', 
 'randomforestclassifier__max_depth': 7, 
 'randomforestclassifier__n_estimators': 100}

I also checked for not including PCA. Got a worse score on kaggle. But 
nested CV estimated better accuracy than when including PCA.
"""

# # fit model on entire training data with best parameters
# clf = gs.best_estimator_
# clf.fit(X_red, y_red)

# predicted=clf.predict(iter_imp)

# # dataframe of predictions
# id_col=[i for i in range(0,iter_imp.shape[0])]
# dict_val={'id': id_col,'Predicted': predicted}
# results=pd.DataFrame(dict_val)
# results=results.set_index('id')
# results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA4\\results_08.csv')

