# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 20:48:21 2021

@author: she84
"""

# =============================================================================
# 4.3.3 ROC curve
# =============================================================================

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

iris = datasets.load_iris()
X = iris["data"][:, 3:]  # petal width (花瓣寬度)
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=22)
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train) 
print('Logistic regression score: %.3f' % log_reg.score(X_test, Y_test))
Y_scores = log_reg.decision_function(X_train)


# Compute ROC curve
Y_score = log_reg.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(Y_test, Y_score)
fpr.shape, tpr.shape, thresholds.shape



plt.figure(figsize=(18, 12))
plt.plot(fpr, tpr, color='red', label='Logistic regression (AUC: %.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("04_ROC_plot")
plt.grid()
plt.show()


auc(fpr, tpr)








