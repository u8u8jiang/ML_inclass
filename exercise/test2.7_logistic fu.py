# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 20:56:56 2021

@author: she84
"""

# =============================================================================
# 6.4 Softmax Regression
# =============================================================================

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, zero_one_loss, confusion_matrix, \
    precision_score, recall_score, fbeta_score 

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal width (花瓣寬度) and length (長度)
y = iris["target"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state = 42)

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
softmax_reg.fit(X_train, Y_train)

x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)


# decision boundary

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris-Virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris-Versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris-Setosa")


custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap, linewidth=5)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.savefig("data/lec06 softmax_regression_contour_plot")
plt.show()




# confusion matrix and mean accurancy

print('Logistic regression score: %.3f' % softmax_reg.score(X_test, Y_test))
pd.crosstab(Y_test, softmax_reg.predict(X_test),
            rownames=['label'], colnames=['predict'])

softmax_reg.predict([[5, 2]])
softmax_reg.predict_proba([[5, 2]])
np.vstack((softmax_reg.predict(X_test), Y_test))


cm = confusion_matrix(y_true = Y_test, y_pred = softmax_reg.predict(X_test))
print(cm[::-1, ::-1])
print('Logistic regression score: %.3f' % softmax_reg.score(X_test, Y_test))



pd.crosstab(Y_test, softmax_reg.predict(X_test),
            rownames=['label'], colnames=['predict'])






















