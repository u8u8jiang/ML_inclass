# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 21:14:07 2021

@author: she84
"""

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score




iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal width (花瓣寬度) and length (長度)
y = iris["target"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state = 46)

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state = 46)
softmax_reg.fit(X_train, Y_train)
print('Logistic regression score: %.3f' % softmax_reg.score(X_test, Y_test))
pd.crosstab(Y_test, softmax_reg.predict(X_test),
            rownames=['label'], colnames=['predict'])


# cross-validation (交叉驗證)
lr = LogisticRegression()
scores = cross_val_score(lr, X, y, scoring='accuracy', cv = 10)
scores














