# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 08:34:37 2021

@author: Nida

CA3 decision tree model entropy 
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

train_data =pd.read_csv('train.csv',index_col=0)
X=train_data[train_data.columns[:-1]]
y=train_data[train_data.columns[8]]

tree_depth=range(1,11)
test_acc=[]
for td in tree_depth:
    acc=[]
    for rs in range(1,101):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,
                                                    stratify=y,random_state=rs)
        # Initialise decision tree classifier and fit
        tree = DecisionTreeClassifier(criterion='entropy', 
                                      max_depth=td, 
                                      random_state=1)
        tree.fit(X_train,y_train)
        test=tree.score(X_test,y_test)
        acc.append(test)
    test_acc.append(np.mean(acc))

accuracies = {'test acc':test_acc}
acc_df = pd.DataFrame(data=accuracies)

# Add column holding strings indicating value of tree depth. Needed later for 
# xticks lables
acc_df['Tree depth'] = ['{0}'.format(depth) for depth in tree_depth]

# Plot columns acc train and acc test. Define xticks lables
ax = acc_df.plot(xticks=acc_df.index, rot=45)
ax.set_xticklabels(acc_df['Tree depth'])

# Set axis lables
ax.set_xlabel('Tree depth')
ax.set_ylabel('Classification accuracy')

plt.show()

"""
Best accuracy when using a tree of depth 2. Do not need to check more depths as
the accuracy decreased for deeper trees.
0.7329710144927537

"""
#Best model
# Best model
test_data=pd.read_csv('test.csv',index_col=0)
X_test=test_data[test_data.columns[:]]
# Train model with est parameter on entire train set.
sc = StandardScaler()
sc.fit(X)
X_sc = sc.transform(X)
X_test_sc=sc.transform(X_test)

best_dec_tree_entropy=DecisionTreeClassifier(criterion='entropy', 
                                      max_depth=2,random_state=1)
best_dec_tree_entropy.fit(X_sc,y)
y_pred=best_dec_tree_entropy.predict(X_test_sc)

# dataframe of predictions
id_col=[i for i in range(0,len(X_test))]
dict_val={'id': id_col,'Predicted': y_pred}
results=pd.DataFrame(dict_val)
results=results.set_index('id')
results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA3\\decision_tree_entropy_results.csv')

check=pd.read_csv('svc_linear_results.csv')