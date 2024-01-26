# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 08:47:38 2021

@author: Nida

CA3 decision tree
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

train_data =pd.read_csv('train.csv',index_col=0)
X=train_data[train_data.columns[:-1]]
y=train_data[train_data.columns[8]]

n_trees=range(100,1001,100)
test_acc=[]
for n in n_trees:
    acc=[]
    for rs in range(1,101):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,
                                                    stratify=y,random_state=rs)
        # Initialise decision tree classifier and fit
        trees = RandomForestClassifier(criterion='entropy', 
                                      n_estimators=n, 
                                      random_state=1)
        trees.fit(X_train,y_train)
        test=trees.score(X_test,y_test)
        acc.append(test)
    test_acc.append(np.mean(acc))

accuracies = {'test acc':test_acc}
acc_df = pd.DataFrame(data=accuracies)

# Add column holding strings indicating value of tree depth. Needed later for 
# xticks lables
acc_df['Num Trees'] = ['{0}'.format(n) for n in n_trees]

# Plot columns acc train and acc test. Define xticks lables
ax = acc_df.plot(xticks=acc_df.index, rot=45)
ax.set_xticklabels(acc_df['Num Trees'])

# Set axis lables
ax.set_xlabel('Number of trees')
ax.set_ylabel('Classification accuracy')

plt.show()

"""
Best accuracy when using gini criterion and 70 decision trees. Accuracy 0.74471

Accuracy only improves with increasing number of trees, accuracy is
f.ex. 0.7465942028985509 for 1000 trees. Use this to fit a model 
"""
# Best model
test_data=pd.read_csv('test.csv',index_col=0)
X_test=test_data[test_data.columns[:]]
# Train model with est parameter on entire train set.
sc = StandardScaler()
sc.fit(X)
X_sc = sc.transform(X)
X_test_sc=sc.transform(X_test)

best_rf= RandomForestClassifier(criterion='entropy', n_estimators=1000, 
                                      random_state=1)
best_rf.fit(X_sc, y)
y_pred=best_rf.predict(X_test_sc)


# dataframe of predictions
id_col=[i for i in range(0,len(X_test))]
dict_val={'id': id_col,'Predicted': y_pred}
results=pd.DataFrame(dict_val)
results=results.set_index('id')
results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA3\\random_forest_entropy_results.csv')

#check=pd.read_csv('random_forest_gini_results.csv')