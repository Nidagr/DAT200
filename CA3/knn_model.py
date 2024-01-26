# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:03:19 2021

@author: Nida

CA3 KNN
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

train_data =pd.read_csv('train.csv',index_col=0)
X=train_data[train_data.columns[:-1]]
y=train_data[train_data.columns[8]]

k_values=range(9,16)
p_values=range(1,11)

test_acc=np.zeros((len(p_values),len(k_values)))
for p_ind,p in enumerate(p_values):
    for k_ind, k in enumerate(k_values):
        acc=[]
        for rs in range(1,101):
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,
                                                    stratify=y,random_state=rs)
            sc = StandardScaler()
            sc.fit(X_train)
            # Transform (standardise) both X_train and X_test with mean and STD from
            # training data
            X_train_sc = sc.transform(X_train)
            X_test_sc = sc.transform(X_test)
            
            knn=KNeighborsClassifier(n_neighbors=k,p=p,metric='minkowski',
                                     n_jobs=-1)
            knn.fit(X_train_sc,y_train)
            test=knn.score(X_test_sc,y_test)
            acc.append(test)
        test_acc[p_ind,k_ind]=np.mean(acc)
            
# Create row names for heat map
rowNames = ['{0}'.format(p) for p in p_values]

# Create column names for heat map
colNames = ['{0}'.format(k) for k in k_values]

# Set up the matplotlib figure for train accuracies
acc_df = pd.DataFrame(test_acc, index=rowNames, columns=colNames)
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn for train accuracies
sns.heatmap(acc_df, vmax=1)
plt.title('Test data accuracies')
plt.xlabel('k')
plt.ylabel('p')
plt.show()


"""
Best accuracy 0.739130 when k=12 and p=1.
"""
#Best model
test_data=pd.read_csv('test.csv',index_col=0)
X_test=test_data[test_data.columns[:]]
# Train model with est parameter on entire train set.
sc = StandardScaler()
sc.fit(X)
X_sc = sc.transform(X)
X_test_sc=sc.transform(X_test)

best_knn=KNeighborsClassifier(n_neighbors=12,p=1,metric='minkowski',n_jobs=-1)
best_knn.fit(X_sc,y)

y_pred=best_knn.predict(X_test_sc)


# dataframe of predictions
#id_col=[i for i in range(0,len(X_test))]
#dict_val={'id': id_col,'Predicted': y_pred}
#results=pd.DataFrame(dict_val)
#results=results.set_index('id')
#results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA3\\knn_results.csv')

"""
Best accuracy 0.7391304347826088 when using k=12 and p=1.
p cant be less than 1 (distance metric) so do not need to check for smaller 
p. Best for k value in the middle of our range do not need to check other 
values.

Find most important features and remove nonimportant ones.
"""
# Split into test and training set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,stratify=y, random_state=0)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#knn=KNeighborsClassifier(n_neighbors=12,p=1,metric='minkowski',n_jobs=-1)
"""
# Initialise Sequential Feature Selector, we use sequential backward selection
#remove one feature at a time until we are left with one feature.
#remove fetures based on accuracy of model.
sfs1 = SFS(knn, 
           k_features=1, 
           forward=False, 
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=5)
# Fit models
sfs1 = sfs1.fit(X_train_std, y_train)

# This dictionary contains results from all compuations, that is, metrics from models with 8 until 1 features
metricDict = sfs1.get_metric_dict()

# Initialise plot
fig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_dev')

#plt.ylim([0.8, 1])
plt.title('Sequential Feature Selection (w. StdDev)')
plt.grid()
plt.show()



# Best 6 features, the best features are 0,2,3,4,5,7
X_sfs=X.iloc[:,[0,2,3,4,5,7]]
test_data=pd.read_csv('test.csv',index_col=0)
X_test_sfs=test_data.iloc[:,[0,2,3,4,5,7]]

sc = StandardScaler()
sc.fit(X_sfs)
X_sc = sc.transform(X_sfs)
X_test_sc=sc.transform(X_test_sfs)

knn_sfs=KNeighborsClassifier(n_neighbors=12,p=1,metric='minkowski',n_jobs=-1)
knn_sfs.fit(X_sc,y)
y_pred=knn_sfs.predict(X_test_sc)

# dataframe of predictions
id_col=[i for i in range(0,len(X_test_sfs))]
dict_val={'id': id_col,'Predicted': y_pred}
results=pd.DataFrame(dict_val)
results=results.set_index('id')
results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA3\\knn_best6_results.csv')


# Initialise Sequential Feature Selector, we use sequential backward selection
#remove one feature at a time until we are left with one feature.
#remove fetures based on accuracy of model.
sfs1 = SFS(knn, 
           k_features=8, 
           forward=True, 
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=5)
# Fit models
sfs1 = sfs1.fit(X_train_std, y_train)

# This dictionary contains results from all compuations, that is, metrics from models with 8 until 1 features
metricDict = sfs1.get_metric_dict()

# Initialise plot
fig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_dev')

#plt.ylim([0.8, 1])
plt.title('Sequential Feature Selection (w. StdDev)')
plt.grid()
plt.show()



# Best 5 features, the best features are 1,2,3,4,7
X_sfs=X.iloc[:,[1,2,3,4,7]]
test_data=pd.read_csv('test.csv',index_col=0)
X_test_sfs=test_data.iloc[:,[1,2,3,4,7]]

sc = StandardScaler()
sc.fit(X_sfs)
X_sc = sc.transform(X_sfs)
X_test_sc=sc.transform(X_test_sfs)

knn_sfs=KNeighborsClassifier(n_neighbors=12,p=1,metric='minkowski',n_jobs=-1)
knn_sfs.fit(X_sc,y)
y_pred=knn_sfs.predict(X_test_sc)

# dataframe of predictions
id_col=[i for i in range(0,len(X_test_sfs))]
dict_val={'id': id_col,'Predicted': y_pred}
results=pd.DataFrame(dict_val)
results=results.set_index('id')
results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA3\\knn_best5_results.csv')

"""