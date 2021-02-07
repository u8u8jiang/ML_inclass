# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 21:17:42 2021

@author: she84
"""


# =============================================================================
# # Grid Search
# =============================================================================

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import multiprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score 
import time


multiprocessing.cpu_count()
startTime = time.time() 

iris = datasets.load_iris()
X = iris["data"] 
y = iris["target"]
# Define a param grid
param_grid = [
    {
        'penalty': ['l1', 'l2'],
        'C': [1e-5, 1e-4, 5e-4, 1e-3, 2.3e-3, 5e-3, 1e-2, 1, 5, 10, 15, 20, 100]  
    }
]

# Create and train a grid search
gs = GridSearchCV(estimator = LogisticRegression(), param_grid = param_grid, scoring ='accuracy', cv = 10)
   
gs.fit(X, y)

# Best estimator
print(gs.best_estimator_)

gs_scores = cross_val_score(gs.best_estimator_, X, y, scoring='accuracy', cv=10)
print('Best estimator CV average score: %.3f' % gs_scores.mean())

endTime = time.time() # 結束
print('%s seconds to calculate.' % (endTime - startTime))


