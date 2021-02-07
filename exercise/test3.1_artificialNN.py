# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 09:38:45 2021

@author: T30518
"""

import keras
from keras import models, layers, regularizers
from keras.datasets import mnist
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score


#各項函數
layersDense1= 512 #第一層 hidden nodes
activation1= 'relu' #第一層分類器 Classifier 
layersDense2= 10 #第二層 hidden nodes
activation2= 'relu' #第二層分類器 Classifier
outputnums= 10 #output nodes
activation3= 'softmax' #output分類器 Classifier

compileoptimizer= 'rmsprop'
compileloss= 'categorical_crossentropy'  #可改用 mse
compilemetrics= ['accuracy'] #可改用 ['mae']
epochsnum= 5
batchsize= 128
lablenumber= 1

# 讀入mnist資料集中的訓練資料及測試資料
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 資料預處理及訓練,train資料大小60000,test資料大小10000,像素28*28
train_images1 = train_images.reshape((60000, 28 * 28))
train_images1 = train_images1.astype('float32') / 255
test_images1 = test_images.reshape((10000, 28 * 28))
test_images1 = test_images1.astype('float32') / 255
train_labels_categ = to_categorical(train_labels)
test_labels_categ = to_categorical(test_labels)
kernelregularizer = regularizers.l1(0.01)

# 編譯模型
network = models.Sequential()
network.add(layers.Dense(layersDense1, kernel_regularizer = kernelregularizer,  activation= activation1, input_shape=(28*28,)))
network.add(layers.Dense(layersDense2, activation= activation2))
network.add(layers.Dropout(1)) #dropout layer
network.add(layers.Dense(outputnums, activation= activation3))
network.compile(optimizer= compileoptimizer,
                loss= compileloss,
                metrics= compilemetrics)

#訓練神經網路
history = network.fit(train_images1, train_labels_categ, epochs= epochsnum, batch_size= batchsize, 
                      validation_data=(test_images1, test_labels_categ))

# Performance
prediction = network.predict_classes(test_images1)
crosstablex = pd.crosstab(test_labels, prediction,
            rownames=['label'], colnames=['predict'])
accuracyscore = accuracy_score(test_labels, prediction)

for i in range(len(test_labels)):
    if (test_labels[i] == lablenumber) & (test_labels[i] != prediction[i]):
        print('when ', i, ': test_labels is ', test_labels[i], ', but predict ', prediction[i])
        plt.imshow(test_images[i,:,:], cmap = plt.cm.gray) 
        plt.show()

print('layers1: %2d, '%layersDense1)
        
print(network.predict_classes(test_images1))
print(network.predict(test_images1))
print(prediction)
print(crosstablex)
print(accuracyscore)
