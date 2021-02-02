# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 21:51:09 2021

@author: jxunchen
"""
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

#資料集
diabetes = datasets.load_diabetes() #載入資料
# print(diabetes.DESCR)
X = diabetes.data
y = diabetes.target

#獲取一個特徵
diabetes_x_temp = diabetes.data[:, np.newaxis, 2] 

diabetes_x_train = diabetes_x_temp[:-20]   #訓練樣本
diabetes_x_test = diabetes_x_temp[-20:]    #測試樣本 後20行
diabetes_y_train = diabetes.target[:-20]   #訓練標記
diabetes_y_test = diabetes.target[-20:]    #預測對比標記

#迴歸訓練及預測
reg = LinearRegression()
reg.fit(diabetes_x_train, diabetes_y_train)  #注: 訓練資料集

#係數 殘差平法和 方差得分
print ('Coefficients :\n', reg.coef_)
print ("Residual sum of square: %.2f" %np.mean((reg.predict(diabetes_x_test) - diabetes_y_test) ** 2))
print ("variance score: %.2f" % reg.score(diabetes_x_test, diabetes_y_test))

#繪圖
plt.title(u'LinearRegression Diabetes')   #標題
plt.xlabel(u'Attributes')                 #x軸座標
plt.ylabel(u'Measure of disease')         #y軸座標
#點的準確位置
plt.scatter(diabetes_x_test, diabetes_y_test, color = 'black')
#預測結果 直線表示
plt.plot(diabetes_x_test, reg.predict(diabetes_x_test), color='blue', linewidth = 3)
plt.show()