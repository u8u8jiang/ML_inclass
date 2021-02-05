# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 23:36:32 2021

@author: she84
"""

import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error




# # data scaling and nomalization
# cat_df = pd.concat([data["sex"], data["smoker"], data["region"]], axis=1)
# pd.get_dummies(cat_df).head()

# # correlation
# all_data = pd.concat([data_df, data_labels], axis=1)
# all_data.corr()


# # training and evaluating on the train set 

# # Create a linear regressor instance
# lr = LinearRegression(normalize=True)
# # Train the model
# lr.fit(data_df, data_labels)
# print( "Score {:.4f}".format(lr.score(data_df, data_labels)) )   #coef of determination


# # prediction and error
# print(np.sqrt(mean_squared_error(data_labels, lr.predict(data_df) )))
# data_df.describe()



