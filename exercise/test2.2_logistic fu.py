# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 20:15:54 2021

@author: she84
"""

# =============================================================================
# Use one feature (花瓣寬度) to classify
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, zero_one_loss, confusion_matrix, \
    precision_score, recall_score, fbeta_score 
import math
from sklearn.metrics import classification_report



iris = datasets.load_iris()
iris["target"]

X = iris["data"][:, 3:]  # petal width (花瓣寬度)
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0
plt.scatter(X,y)
plt.show()

# split the data set
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20)


# train the model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, Y_train)
print('Logistic regression training score: %.3f' % log_reg.score(X_train, Y_train))
print('Logistic regression testing score: %.3f' % log_reg.score(X_test, Y_test))

# evaluate the model
accuracy_score(Y_test, log_reg.predict(X_test))
log_reg.predict(X_test)
Y_test

compare = Y_test - log_reg.predict(X_test)
compare = np.abs(compare)
compare


(Y_test.shape[0]  - compare.sum()) / Y_test.shape[0] 


correct = 0
for i in range(Y_test.shape[0]):
    if log_reg.predict(X_test)[i] == Y_test[i]:
        correct += 1
correct/Y_test.shape[0] 




# =============================================================================
# # prediction by formula
# =============================================================================

log_reg.coef_[0][0], log_reg.intercept_[0]

log_reg.decision_function(X_test[1].reshape(1,-1))

log_reg.coef_[0][0]*X_test[1] + log_reg.intercept_[0]


# for i in range(Y_test.shape[0]):
#     a = math.exp(log_reg.intercept_[0] + log_reg.coef_[0][0] * X_test[i])
#     print('x: {0}, Decision function: {1}, predicted probability: {2},  real class" {3}'\
#           .format(X_test[i][0], log_reg.coef_[0][0]*X_test[i] + log_reg.intercept_[0], a/(1+a), Y_test[i]))

# for i in range(Y_train.shape[0]):
#     a = math.exp(log_reg.intercept_[0] + log_reg.coef_[0][0] * X_train[i])
#     print('x: ', X_train[i][0], \
#           'Decision function: ', log_reg.coef_[0][0]*X_train[i] + log_reg.intercept_[0], \
#           'pred ', a/(1+a),\
#           'real ', Y_train[i])
        
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                   penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                   verbose=0, warm_start=False)

#confusion matrix: a small example
y_true = [0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
y_pred = [0, 0, 0, 1, 1, 0, 0, 1, 0, 1]
confusion_matrix(y_true, y_pred)
cm = confusion_matrix(y_true=Y_test, y_pred= log_reg.predict(X_test))

print(cm[::-1, ::-1])  
print(classification_report(Y_test, log_reg.predict(X_test)))





