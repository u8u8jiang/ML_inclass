# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:42:44 2021

@author: T30518
"""



from sklearn.preprocessingimport LabelEncoder

# READ DATA
names = ['age','sex','region','income','married','children','car','sa ve_act','current_act','mortgage','pep'] 
dataset = pd.read_csv('bank-data.csv', names=names)
X = dataset.drop('pep', axis=1)  
y = dataset['pep']

# PREPROCESSING DATA
labelencoder= LabelEncoder() 
X_le = X 
X_le['sex'] = labelencoder.fit_transform(X_le['sex']) 
X_le['region'] = labelencoder.fit_transform(X_le['region'])
y_le= labelencoder.fit_transform(y) 
print(X_le) 
print(y_le) 




