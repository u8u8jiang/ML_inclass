import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# input the Data
data = pd.read_csv("../data/lec03-insurance.csv") 
# data.info()
# print(data.describe())

# =============================================================================
# # 1.data pregressing
# =============================================================================

# drop the missing value
# also can replace the missing value as median, mean or mode, 
# or random replace dep on uniform dist.

insurance = data.dropna()
insurance = insurance.reset_index(drop=True)
# insurance.info()
# print(insurance.shape)


# encoding with OneHotEncoding

insurance_num = insurance.drop(['sex', 'smoker','region'], axis=1)

encoder = LabelBinarizer()
sex_cat = encoder.fit_transform(insurance['sex'])
smoker_cat = encoder.fit_transform(insurance["smoker"])
region_cat = encoder.fit_transform(insurance["region"])

sex_df = pd.DataFrame(sex_cat, columns = ['sex'])  #female=1, male=0
smoker_df = pd.DataFrame(smoker_cat, columns = ['smoker'])
region_df = pd.DataFrame(region_cat, columns = ['rgNE', 'rgNW', 'rgSE','rgSW'])

insurance1 = pd.concat([sex_df,insurance_num, smoker_df, region_df], axis=1)
# insurance1 = pd.merge(insurance_num, sex_df, smoker_df, region_df)






