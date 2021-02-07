# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 20:45:19 2021

@author: she84
"""

# =============================================================================
# 4.3.2 Precision/Recall Tradeoff (取捨)
# =============================================================================


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt


iris = datasets.load_iris()

X = iris["data"][:, 3:]  # petal width (花瓣寬度)
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=42)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, Y_train) 
Y_scores = log_reg.decision_function(X_train)

precisions, recalls, thresholds = precision_recall_curve(Y_train, Y_scores)
precisions, recalls, thresholds
thresholds.min(), thresholds.max()


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="lower left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-0.5, 2.1])
plt.savefig("data/06_precision_recall_vs_threshold_plot")
plt.show()

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-0.5, 0.7])
plt.savefig("04_precision_recall_vs_threshold_plot")
plt.show()

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-0.5, 1])
plt.savefig("04_precision_recall_vs_threshold_plot")
plt.show()



