# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:03:53 2021

@author: jjryaya
"""


# complte a report for iris


from sklearn.datasets import load_iris 
from sklearn import tree # import graphviz：https://www.graphviz.org/
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_text
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import graphviz 
from sklearn import svm 


iris = load_iris() 
X = iris.data 
y = iris.target 

# decision tree
clf= tree.DecisionTreeClassifier()  #criterion='entropy' 

# TRAIN AND TEST
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.4, random_state=42) 
clf= clf.fit(X_train, y_train) 
print(clf.score(X_test, y_test)) 
y_pred= clf.predict(X_test) 
print(y_pred)


# CROSS VALIDATION
X = X_train
y = y_train
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
tree.plot_tree(clf) 
dot_data= tree.export_graphviz(clf,  out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("irislimit3") 






