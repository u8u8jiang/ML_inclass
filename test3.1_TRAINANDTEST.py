# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:40:25 2021

@author: T30518
"""


'''

# decision tree
from sklearn.datasets import load_iris 
from sklearn import tree # import graphviz：https://www.graphviz.org/
import graphviz
X, y = load_iris(return_X_y=True) 
clf= tree.DecisionTreeClassifier(criterion='entropy') 
clf= clf.fit(X, y) 
tree.plot_tree(clf) 

# OUPUTTREE
dot_data= tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 
# print(graph)

# Refined results of graph
dot_data= tree.export_graphviz(clf, out_file=None, 
                               feature_names=iris.feature_names,  
                               class_names=iris.target_names,  
                               filled=True, rounded=True,  
                               special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("iris") 



from sklearn.datasets import load_iris 
from sklearn import tree 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report 
# k = 5 for KNeighborsClassifier

iris = load_iris() 
X = iris.data 
y = iris.target 

# TRAIN AND TEST
clf= tree.DecisionTreeClassifier() 
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.4, random_state=42) 

clf= clf.fit(X_train, y_train) 
print(clf.score(X_test, y_test)) 
y_pred= clf.predict(X_test) 
print(y_pred)

'''
























