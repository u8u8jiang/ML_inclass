import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting  import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Get the Data
insurance = pd.read_csv("../data/lec03-insurance.csv") 
# insurance.info()
# insurance.describe()   #can't run
data = insurance.dropna()
# data.info()

# data pregressing

# Create a Test Set
test_ratio = 1
# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    # to make this notebook's output identical at every run
    np.random.seed(42)  
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

train_set, test_set = split_train_test(data, 0.2)
print(len(train_set), "train +", len(test_set), "test", ' = total ',  len(train_set) + len(test_set))



# train_set.info()
# test_set.info()

# # Looking for Correlations
# corr_matrix = data.corr()
# # print(corr_matrix)

# # visulation
# attributes = ["age", "bmi", "children" , "charges"]
# scatter_matrix(data[attributes])
# plt.show()

# # data scaling and nomalization
# cat_df = pd.concat([data["sex"], data["smoker"], data["region"]], axis=1)
# cat_df.head()
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



