# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:56:09 2021

@author: T30518
"""


'''
from sklearn.datasets import load_iris 
from sklearn import tree 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report 
# K-FOLD CROSS VALIDATIO
from sklearn.model_selection import KFold 

clf= tree.DecisionTreeClassifier() 
kf= KFold(n_splits=2) 
kf.get_n_splits(X) 
print(kf) 

for train_index, test_indexin kf.split(X): 
    print("TRAIN:", train_index) 
    print("TEST:", test_index) 
    X_train, X_test= X[train_index], X[test_index] 
    y_train, y_test= y[train_index], y[test_index] 
    print("TRAIN data:") 
    print(X_train, y_train) 
    print("TEST data:") 
    print(X_test, y_test)
'''



# CROSS VALIDATION

from sklearn.datasets import load_iris 
from sklearn import tree 
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix, classification_report
import pickle

iris = load_iris() 
X = iris.data 
y = iris.target
clf= tree.DecisionTreeClassifier() 
clf= clf.fit(X, y) 
scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy') 
print(scores) 

pkl_filename= 'iris_model.pkl' 
with open(pkl_filename, 'wb') as file:  #寫二進制文件 
    pickle.dump(clf, file)

print("%0.2f accuracy with a standard deviation of %0.2f" % 
      (scores.mean(), scores.std()))
tree_rules= export_text(clf, feature_names=iris['feature_names']) 
print(tree_rules)

y_pred= cross_val_predict(clf, X, y, cv=10) 
conf_mat= confusion_matrix(y, y_pred)
print(confusion_matrix(y, y_pred)) 
print(classification_report(y, y_pred))



# PRUNING TREE
from sklearn.datasets import load_iris 
from sklearnimport tree import graphviz 

X, y = load_iris(return_X_y=True) 
clf= tree.DecisionTreeClassifier(min_samples_leaf=3) 
clf= clf.fit(X, y) 
tree.plot_tree(clf) 
dot_data= tree.export_graphviz(clf,  out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("irislimit3") 






























