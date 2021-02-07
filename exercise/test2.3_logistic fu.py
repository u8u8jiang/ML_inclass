# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 20:24:26 2021

@author: she84
"""

# =============================================================================
# another data set
# =============================================================================

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, zero_one_loss, confusion_matrix, \
    precision_score, recall_score, fbeta_score 
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris["data"][:, 3:]  # petal width (花瓣寬度)
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20)


log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, Y_train)

cm = confusion_matrix(y_true=Y_test, y_pred= log_reg.predict(X_test))
print(cm[::-1, ::-1])
print('Logistic regression score: %.3f' % log_reg.score(X_test, Y_test))


# =============================================================================
# ##### Graphical explanation
# 
# * https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.reshape.html
# * One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
# 
# =============================================================================

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

plt.figure(figsize=(8, 3))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
plt.savefig("logistic_regression_plot")
plt.show()


y_proba = log_reg.predict_proba(X_new)
y_proba
y_proba.shape

decision_boundary  #以 1.61561562 當基準 , > 1.61561562 為 1
log_reg.predict([[1.7], [1.5]])













