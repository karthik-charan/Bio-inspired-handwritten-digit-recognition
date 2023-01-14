# Machine learning model code goes here
#please note this code gives some runtimewarnings but runs through. I am unsure whether that is an issue
#first load the train and test data set
import os
import pandas as pd
import numpy as np
from visualise import plot_digi_fig
import matplotlib.pyplot as plt
from dataloader import FFT_transform

entries = os.listdir('Data/')
#print(entries)
train_data=[k for k in entries if 'train' in k]
#print(train_data)
X_title_train=[]
flat_data_train=[]
X_data_train = []
for entry in train_data:
    name=entry
    #print(entry)
    with open('Data/'+name) as f:
        #print(f)
        for line in f:
            curr = line.strip()
            mat = np.fromstring(curr, dtype=int, sep='  ')
            mat_r = np.reshape(mat, (-1, 15))
    
            #here we apply the FFT transform
            mag_val = FFT_transform(mat_r)
            #break
            X_data_train.append(mag_val)
            X_title_train.append(name)
            flat_data_train.append(mag_val.flatten())
    #X_data_train.append(mag_val)
    #X_data.append (image)
test_data=[k for k in entries if 'test' in k]
#print(test_data)
X_title_test=[]
X_data_test = []
flat_data_test=[]
for entry in test_data:
    name=entry
    #print(entry)
    with open('Data/'+name) as f:
        #print(f)
        for line in f:
            curr = line.strip()
            mat = np.fromstring(curr, dtype=int, sep='  ')
            mat_r = np.reshape(mat, (-1, 15))
    

            mag_val = FFT_transform(mat_r)
            #break
            X_data_test.append(mag_val)
            X_title_test.append(name)
            flat_data_test.append(mag_val.flatten())
    #X_data.append (image)
    
    
#calculate the labels of the data
res_train=[]
res_train = [sub.replace('.txt', '') for sub in X_title_train]
res_train = [sub.replace('train_', '') for sub in res_train]
res_train = [eval(i) for i in res_train]
X_train_lable=res_train
res_test=[]
res_test = [sub.replace('.txt', '') for sub in X_title_test]
res_test = [sub.replace('test_', '') for sub in res_test]
res_test = [eval(i) for i in res_test]
X_test_lable=res_test
#print(res_test)
#print(X_title_train)

#train a simple ML algorithm here SVM:
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import math
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
param_grid={'C':[0.1,100],'gamma':[0.0001,0.001],'kernel':['poly']}

svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)

x_train=flat_data_train
#here I just convert nan to numbers but we might want to look deeper into where the nan values are coming from in the first place
x_train=np.nan_to_num(x_train)
#print(len(flat_data_train))
y_train=X_train_lable
#print(len(y_train))
x_test=flat_data_test
x_test=np.nan_to_num(x_test)
y_test=X_test_lable

model.fit(x_train,y_train)

#and run it on the test dataset
y_pred=model.predict(x_test)
print("these are the teacher outputs for the test dataset")
print(y_train)
print("predicted output vector:")
print(y_pred)

#calculate and print the accuracy
print("accuracy:")
print(sum(y_pred==y_test)/len(y_test))

#visualize metrics for understanding in which classes we have bad predictions
cm = confusion_matrix(y_test, y_pred,normalize='true')
labels = ['0','1','2','3','4','5','6','7','8','9']
print(classification_report(y_test, y_pred)) 
fig, ax = plt.subplots(figsize=(8,8))
sns_plot = sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')

plt.title('Confusion Matrix')
plt.show()
fig.savefig('Conf_Matrix.png',bbox_inches='tight',dpi=150)


