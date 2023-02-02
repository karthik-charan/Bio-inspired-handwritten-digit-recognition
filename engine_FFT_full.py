# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 02:18:36 2023

@author: Alexandros
"""

# Machine learning model code goes here
#first load the train and test data set
import os
import pandas as pd
import numpy as np
from visualise import plot_digi_fig
import matplotlib.pyplot as plt
from dataloader import  FFT_transform_ST

entries = os.listdir('Data/')
#print(entries)
train_data=[k for k in entries if 'train' in k]
#print(train_data)

X_title_train=[]
X_data_train = []


for entry in train_data:
    name=entry
    
    res = "".join([ele for ele in entry if ele.isdigit()])  # keep only the integer part of the path
    #print(entry)
    with open('Data/'+name) as f:
        #print(f)
        for line in f:
            curr = line.strip()
            mat = np.fromstring(curr, dtype=int, sep='  ')
            mat_r = np.reshape(mat, (-1, 15))
    
            #here we apply the FFT transform
            X_data_train.append(np.array(FFT_transform_ST(mat_r)))

            X_title_train.append(int(res))

X_data_train = np.array(X_data_train)  
  
test_data=[k for k in entries if 'test' in k]
#print(test_data)

#create nested lists for storing reasons
X_title_test=[]
X_data_test = []



for entry in test_data:
    name=entry
    
    res = "".join([ele for ele in entry if ele.isdigit()])  # keep only the integer part of the path
    #print(entry)
    with open('Data/'+name) as f:
        #print(f)
        for line in f:
            curr = line.strip()
            mat = np.fromstring(curr, dtype=int, sep='  ')
            mat_r = np.reshape(mat, (-1, 15))   

            X_data_test.append(np.array(FFT_transform_ST(mat_r)))
 
            X_title_test.append(int(res))
   
X_data_test = np.array( X_data_test)      


X_train_lable = X_title_train
X_test_lable = X_title_test
#print(res_test)
#print(X_title_train)



from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore') # I use that here as the GridSearchCV function generates a number of warning cluttering your console

from sklearn.linear_model import LogisticRegression


############################--------------Linear Regression-----------##############################

x_train = X_data_train
x_train = x_train.reshape(1000,-1)
y_train = X_title_train

x_test= X_data_test
x_test = x_test.reshape(1000,-1)
y_test= X_title_test

logR = LogisticRegression()
logR2= LogisticRegression()
parameters_lr = {
    'penalty' : ['l1','l2'], 
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],}

lrGrid1 = GridSearchCV(logR,
                   param_grid = parameters_lr,
                   scoring='accuracy',
                   cv=2) # here I only selected 2 iteration of the CV as higher values significantly increase the computation time

lr_b= lrGrid1

lr_b.fit(x_train,y_train)




#check accuracy without parameter tuning 
test_np = logR.fit(x_train,y_train)
fft_pred_np= test_np.predict(x_test)
#check predictions with tunned model
fft_pred_p= lr_b.predict(x_test)



print("LR: accuracy WITHOUT parameter tuning:")
print(round(sum(fft_pred_np==y_test)/len(y_test)*100,1))


print("LR: accuracy WITH parameter tuning:")
print(round(sum(fft_pred_p==y_test)/len(y_test)*100,1))





cm1 = confusion_matrix(y_test, fft_pred_np,normalize='true')
labels = ['0','1','2','3','4','5','6','7','8','9']
print(classification_report(y_test, fft_pred_np)) 
fig, ax = plt.subplots(figsize=(8,8))
sns_plot = sns.heatmap(cm1, annot=True, fmt='.2f', cmap='Blues',xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')

plt.title('LR: Confusion Matrix of Untunned Model')
plt.show()
fig.savefig('Conf_Matrix_LR_Untunned.png',bbox_inches='tight',dpi=150)

cm2 = confusion_matrix(y_test, fft_pred_p,normalize='true')
labels = ['0','1','2','3','4','5','6','7','8','9']
print(classification_report(y_test, fft_pred_p)) 
fig, ax = plt.subplots(figsize=(8,8))
sns_plot = sns.heatmap(cm2, annot=True, fmt='.2f', cmap='Blues',xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')

plt.title('LR: Confusion Matrix of Tunned Model')
plt.show()
fig.savefig('Conf_Matrix_LR_Tunned.png',bbox_inches='tight',dpi=150)



############################--------------XGBoost-----------##############################
import xgboost as xgb

xgb_model = xgb.XGBRegressor(objective="multi:softmax", random_state=42,num_class=10,
                             learning_rate=0.02, n_estimators=600,
                             max_depth=5, # re-optimized from v2
                             reg_lambda=1 # default L2 value
                            )
xgb_model.fit(x_train,y_train)

y_pred = xgb_model.predict(x_test)
y_test = np.asarray(y_test)
print("XGBRegressor: FFT accuracy:")
print(round(sum(y_pred==y_test)/len(y_test)*100,1))

cm4 = confusion_matrix(y_test, y_pred,normalize='true')
labels = ['0','1','2','3','4','5','6','7','8','9']
print(classification_report(y_test, y_pred)) 
fig, ax = plt.subplots(figsize=(8,8))
sns_plot = sns.heatmap(cm4, annot=True, fmt='.2f', cmap='Blues',xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')

plt.title('XGBRegressor: Confusion Matrix of FFT')
plt.show()
fig.savefig('Conf_Matrix_XGBRegressor_FFT.png',bbox_inches='tight',dpi=150)



