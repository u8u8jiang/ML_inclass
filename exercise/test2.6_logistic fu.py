# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 20:53:26 2021

@author: she84
"""

# =============================================================================
# ## 6.3.5 Python with 2 features 
# * 2 features: petal width (花瓣寬度) and length (長度)
# =============================================================================

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, zero_one_loss, confusion_matrix, \
    precision_score, recall_score, fbeta_score 
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal width (花瓣寬度) and length (長度)
y = (iris["target"] == 2).astype(np.int)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20)

log_reg = LogisticRegression(C=10**10, random_state=42)
log_reg.fit(X_train, Y_train)

cm = confusion_matrix(y_true=Y_test, y_pred= log_reg.predict(X_test))
print(cm[::-1, ::-1])
print('Logistic regression score: %.3f' % log_reg.score(X_test, Y_test))
 

# decision boundary

x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 2.7, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = log_reg.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs")
plt.plot(X[y==1, 0], X[y==1, 1], "g^")

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)

left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
#save_fig("logistic_regression_contour_plot")
plt.show()


X.shape









