import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from pandas.plotting  import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# input the Data
data = pd.read_csv("../data/lec03-insurance.csv") 
# data.info()

# =============================================================================
# # 1.data pregressing
# =============================================================================

# drop the missing value
# also can replace the missing value as median, mean or mode, 
# or random replace dep on uniform dist.

insurance = data.dropna()
insurance = insurance.reset_index(drop=True)
#insurance.info()
#print(data.describe())
#print(insurance.shape)


# encoding with OneHotEncoding

insurance_num = insurance.drop(['sex', 'smoker','region'], axis=1)

encoder = LabelBinarizer()
sex_cat = encoder.fit_transform(insurance['sex'])
smoker_cat = encoder.fit_transform(insurance["smoker"])
region_cat = encoder.fit_transform(insurance["region"])

sex_df = pd.DataFrame(sex_cat, columns = ['sex'])  #female=1, male=0
smoker_df = pd.DataFrame(smoker_cat, columns = ['smoker'])
region_df = pd.DataFrame(region_cat, columns = ['rgNE', 'rgNW', 'rgSE','rgSW'])

insurance1 = pd.concat([sex_df, smoker_df, region_df,insurance_num], axis=1)
# insurance1 = pd.merge(insurance_num, sex_df, smoker_df, region_df)


# observe the correlation of variables with charge
# (correlation might imply causation)
corrwcharge = insurance1.corr()['charges'].sort_values(ascending = False)


# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(insurance1.iloc[:, 0:9], insurance1.iloc[:, 9], test_size=0.2, random_state = 42)
trainingset = pd.concat([X_train,Y_train], axis=1)
testingset = pd.concat([X_test,Y_test], axis=1)

# =============================================================================
# # 2.visualization and observation
# =============================================================================


insurance.hist(bins = 50)
plt.savefig("insurance_hist.png")
plt.show()


attributes = ["age", "bmi", "children" , "charges"]
scatter_matrix(insurance[attributes], figsize=(11, 8))
plt.savefig("scatter_matrix_plot.png")
plt.show()


insurance.charges.min(), insurance.charges.max()
insurance.charges.describe()

# observe the correlation of variable!!
corr_matrix = insurance.corr()
corr1_matrix = insurance1.corr()
# how?


# =============================================================================
# # 3.training-and-testing
# =============================================================================


# Train the model
# Create a linear regressor instance
lr = LinearRegression(normalize=True)
lr.fit(X_train, Y_train)


#the score of training data
print( "Score {:.4f}".format(lr.score(X_train, Y_train)) )  
# the regression of intercept ie.beta
print('y = %.3f '% lr.intercept_)
for i, c in enumerate(lr.coef_):
    print('%.3f '% c, X_train.columns.values[i])

# the regression of predicted_y  
predicted_y = lr.intercept_
for i, c in enumerate(lr.coef_):
    predicted_y += c * X_train.iloc[3][i]
print('predicted y = %.3f '% predicted_y)



#print(np.sqrt(mean_squared_error(X_train, lr.predict(Y_train) )))





















