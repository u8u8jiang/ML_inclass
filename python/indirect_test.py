# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:22:35 2021

@author: hao
"""

import pandas as pd # 輸入
import time, math, os
from gurobipy import *
import numpy as np

from indirect import *
from performance import *

# parameter to record the testing results
parameter = 31

# prediction type: 'lin' or 'const'
# split: HyperplaneSplit or ConstantSplit
if parameter == 1: 
    parameter_vector = [2, 74, 2, 'lin',  .03, .01, 1 / 100 , 1 / 100, 2, 1, parameter, 'ConstantSplit']
elif parameter == 2: 
    parameter_vector = [2, 74, 2, 'lin',  .03, .01, 1 / 100 , 1 / 100, 3, 1, parameter, 'ConstantSplit']
elif parameter == 6: 
    parameter_vector = [2, 74, 4, 'lin',  .03, .01, 1 / 100 , 1 / 100, 2, 1/100, parameter, 'ConstantSplit']
elif parameter == 7: 
    parameter_vector = [2, 74, 3, 'lin',  .03, .01, 1 / 100 , 1 / 100, 3, 1/2, parameter, 'ConstantSplit']
elif parameter == 17: 
    parameter_vector = [2, 74, 3, 'lin',  .03, .01, 1 / 100 , 1 / 100, 3, 2, parameter, 'ConstantSplit']
elif parameter == 3: 
    parameter_vector = [2, 74, 2, 'lin',  .03, .01, 1 / 100 , 1 / 100, 2, .001, parameter, 'HyperplaneSplit']
elif parameter == 4: 
    parameter_vector = [2, 74, 2, 'lin',  .03, .01, 1 / 100 , 1 / 100, 2, .01, parameter, 'HyperplaneSplit']
elif parameter == 19: # step 1, 2 only
    parameter_vector = [2, 74, 2, 'lin',  .03, .01, 1 / 100 , 1 / 100, 3, 1/2, parameter, 'HyperplaneSplit']
elif parameter == 21:  
    parameter_vector = [6, 74, 2, 'lin',  .03, .01, 1 / 100 , 1 / 100, 2, 1/6, parameter, 'HyperplaneSplit']
elif parameter == 5: # the best so far
    parameter_vector = [2, 74, 2, 'lin',  .03, .01, 1 / 100 , 1 / 100, 3, 1/6, parameter, 'HyperplaneSplit']
elif parameter == 16: 
    parameter_vector = [2, 74, 2, 'lin',  .03, .01, 1 / 100 , 1 / 100, 2, 1, parameter, 'HyperplaneSplit']
elif parameter == 22: # step 1, 2 only
    parameter_vector = [6, 74, 2, 'lin',  .03, .01, 1 / 100 , 1 / 100, 3, 1, parameter, 'HyperplaneSplit']
elif parameter == 23: # step 1, 2 0nly
    parameter_vector = [2, 74, 2, 'lin',  .03, .01, 1 / 100 , 1 / 100, 3, 1, parameter, 'HyperplaneSplit']
elif parameter == 29: 
    parameter_vector = [2, 74, 2, 'lin',  3, 1, 1 / 1000 , 1 / 1000, 3, 1, parameter, 'ConstantSplit']
elif parameter == 30: 
    parameter_vector = [2, 74, 2, 'lin',  3, 1, 2 / 1000 , 2 / 1000, 3, 1, parameter, 'ConstantSplit']


    
n_feat = parameter_vector[0] 
allowance =  parameter_vector[1] 
D =  parameter_vector[2] 
prediction_type =  parameter_vector[3] # 'lin' # or 'const':
late_fac =  parameter_vector[4] 
early_fac =  parameter_vector[5]
step1 =  parameter_vector[6]  # so the steps are 0, 0.1, 0.2, 0.3
step2 =  parameter_vector[7] 
step_no = parameter_vector[8]
hours = parameter_vector[9]
time_limit = 3600 * hours # by hour
direc = 'parameter' + str(parameter_vector[10])
if not os.path.exists(direc): 
    os.makedirs(direc)


warm_start = 0 # 0 no
small_no = 0.0001

HR_train= np.zeros(10)
HR_test = np.zeros(10)    
early_hours_train = np.zeros(10) 
late_hours_train = np.zeros(10)   
early_hours_test = np.zeros(10)  
late_hours_test = np.zeros(10) 
timing = np.zeros(10) 


out = []
for file_no in range(10): #  files
    df = pd.read_csv('data-julia/' + str(file_no) + '.csv')
    if n_feat == 2:
        feature = [2, 4]
    else: 
        feature = [i for i in range(1,7)]
        
    X = df.iloc[:,feature]
    y = df.iloc[:,-1]
    
    p = y.shape[0]
    train_r = 0.5
    valid_r = 0.25
    
    y_train = y.iloc[0: int(p*train_r)].to_numpy()
    X_train = X.iloc[0: int(p*train_r), :].to_numpy()
     
    y_valid = y.iloc[int(p*train_r): int(p* (train_r + valid_r))].to_numpy()
    X_valid = X.iloc[int(p*train_r): int(p* (train_r + valid_r)), :].to_numpy()
    
    y_test  = y.iloc[int(p*train_r): int(p* (train_r + valid_r)):].to_numpy() 
    X_test =  X.iloc[int(p*train_r): int(p* (train_r + valid_r)):, :].to_numpy()

    # Validation stage
    # alpha = [step*i for i in range(50)]
    train_scores = np.zeros((step_no, step_no ))
    valid_scores = np.zeros((step_no, step_no ))
    
    t = time.time()
    for alpha in range(step_no):
        for Lambda in range(step_no): 
            if parameter_vector[-1]  == 'ConstantSplit': 
                coef = ConstantSplit(file_no, X_train, y_train,  warm_start, alpha, Lambda, parameter_vector)
            else: # 'HyperplaneSplit'
                coef = HyperplaneSplit(file_no, X_train, y_train, warm_start, alpha, Lambda, parameter_vector)

            train_hr, valid_hr, early_late  = performance2(file_no, coef, X_train, y_train, X_valid, y_valid, parameter_vector)            
            train_scores[alpha, Lambda] = train_hr
            valid_scores[alpha, Lambda] = valid_hr
    
    
    timing[file_no] = (time.time() - t ) / (step_no *step_no)
    out.append(['Jobs done in seconds', time.time()-t])
    out.append(['MAE of training dataset', train_scores])
    out.append(['MAE of validation dataset', valid_scores])
    # testing stage
    position = find_max_index(valid_scores)
    
    out.append(['Maximum position of MAE in validation dataset', position])
    
    coef = pd.read_excel(direc + "/file_" + str(file_no) + "_alpha_" + str(position[0]) + "_Lambda_" + str(position[1]) +  ".xlsx")
    train_hr, test_hr, early_late  = performance2(file_no, coef.iloc[:, 1:], X_train, y_train, X_test, y_test, parameter_vector)
    out.append(['MAE of training dataset', train_hr])
    out.append(['MAE of testing dataset', test_hr]) 
    
    df = pd.DataFrame(out) 
    df.to_excel(direc  + "/best_" + str(file_no) + ".xlsx")
    HR_train[file_no] = train_hr
    HR_test[file_no] = test_hr
    early_hours_train[file_no]  = early_late[0]
    late_hours_train[file_no]  = early_late[1]  
    early_hours_test[file_no]  = early_late[2]
    late_hours_test[file_no]  = early_late[3] 
# In[ ]:

file1 = open(direc  + "/output" + str(parameter) + ".txt","a") 

file1.write("\naccuracy_train\n")
file1.write(str(np.round(HR_train, decimals=4)))
file1.write("\nHR  of training \n")
file1.write(str(np.round(np.min(HR_train), decimals=4)) + ', ')
file1.write(str(np.round(np.max(HR_train), decimals=4))+ ', ')
file1.write( str(np.round(np.mean(HR_train), decimals=4))+ ', ')
file1.write(str(np.round(np.std(HR_train),  decimals=4 ))+ ', ')

file1.write("\naccuracy_test\n")
file1.write(str(np.round(HR_test, decimals=4)))
file1.write("\nHR  of testing \n")
file1.write(str(np.round(np.min(HR_test), decimals=4)) + ', ')
file1.write(str(np.round(np.max(HR_test), decimals=4))+ ', ')
file1.write( str(np.round(np.mean(HR_test), decimals=4))+ ', ')
file1.write(str(np.round(np.std(HR_test),  decimals=4 ))+ ', ')

file1.write("\nTiming \n")
file1.write(str(np.round(np.min(timing), decimals=4)) + ', ')
file1.write(str(np.round(np.max(timing), decimals=4))+ ', ')
file1.write( str(np.round(np.mean(timing), decimals=4))+ ', ')
file1.write(str(np.round(np.std(timing),  decimals=4 ))+ ', ')



file1.write("\nearly_hours_train \n")
file1.write(str(np.round(np.min(early_hours_train), decimals=4)) + ', ')
file1.write(str(np.round(np.max(early_hours_train), decimals=4))+ ', ')
file1.write( str(np.round(np.mean(early_hours_train), decimals=4))+ ', ')
file1.write(str(np.round(np.std(early_hours_train),  decimals=4 ))+ ', ')


file1.write("\nlate_hours_train \n")
file1.write(str(np.round(np.min(late_hours_train), decimals=4)) + ', ')
file1.write(str(np.round(np.max(late_hours_train), decimals=4))+ ', ')
file1.write( str(np.round(np.mean(late_hours_train), decimals=4))+ ', ')
file1.write(str(np.round(np.std(late_hours_train),  decimals=4 ))+ ', ')


file1.write("\nearly_hours_test  \n")
file1.write(str(np.round(np.min(early_hours_test ), decimals=4)) + ', ')
file1.write(str(np.round(np.max(early_hours_test ), decimals=4))+ ', ')
file1.write( str(np.round(np.mean(early_hours_test ), decimals=4))+ ', ')
file1.write(str(np.round(np.std(early_hours_test ),  decimals=4 ))+ ', ')


file1.write("\nlate_hours_test \n")
file1.write(str(np.round(np.min(late_hours_test), decimals=4)) + ', ')
file1.write(str(np.round(np.max(late_hours_test), decimals=4))+ ', ')
file1.write( str(np.round(np.mean(late_hours_test), decimals=4))+ ', ')
file1.write(str(np.round(np.std(late_hours_test),  decimals=4 ))+ ', ')

file1.write("\ninfor we need in the table \n")
file1.write( str(np.round(np.mean(HR_test), decimals=4))+ ', ')
file1.write(str(np.round(np.std(HR_test),  decimals=4 ))+ ', ')
file1.write( str(np.round(np.mean(early_hours_test ), decimals=4))+ ', ')
file1.write( str(np.round(np.std(early_hours_test ), decimals=4))+ ', ')
file1.write( str(np.round(np.mean(late_hours_test), decimals=4))+ ', ')
file1.write( str(np.round(np.std(late_hours_test), decimals=4))+ ', ')
file1.write( str(np.round(np.mean(timing), decimals=4))+ ', ')


file1.close() 
