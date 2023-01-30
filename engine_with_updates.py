# Machine learning model code goes here
#please note this code gives some runtimewarnings but runs through. I am unsure whether that is an issue
#first load the train and test data set
import os
import pandas as pd
import numpy as np
from visualise import plot_digi_fig
import matplotlib.pyplot as plt
from dataloader import FFT_transform_PM

entries = os.listdir('Data/')
#print(entries)
train_data=[k for k in entries if 'train' in k]
#print(train_data)
X_title_train=[]
flat_data_train=[]
X_data_train = []
X_data_train_phase=[]
flat_data_train_phase=[]
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
            mag_val,phase_val = FFT_transform_PM(mat_r)
            #break
            X_data_train.append(mag_val)
            X_data_train_phase.append(phase_val)
            X_title_train.append(name)
            flat_data_train.append(mag_val.flatten())
            flat_data_train_phase.append(phase_val.flatten())
    #X_data_train.append(mag_val)
    #X_data.append (image)
test_data=[k for k in entries if 'test' in k]
#print(test_data)
X_title_test=[]
X_data_test = []
flat_data_test=[]
X_data_test_phase=[]
flat_data_test_phase=[]
for entry in test_data:
    name=entry
    #print(entry)
    with open('Data/'+name) as f:
        #print(f)
        for line in f:
            curr = line.strip()
            mat = np.fromstring(curr, dtype=int, sep='  ')
            mat_r = np.reshape(mat, (-1, 15))
    

            mag_val,phase_val = FFT_transform_PM(mat_r)
            #break
            X_data_test.append(mag_val)
            X_data_test_phase.append(phase_val)
            X_title_test.append(name)
            flat_data_test.append(mag_val.flatten())
            flat_data_test_phase.append(phase_val.flatten())
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
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import math
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore') # I use that here as the GridSearchCV function generates a number of warning cluttering your console

# param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
param_grid={'C':[0.1,100],'gamma':[0.0001,0.001],'kernel':['poly']}

svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)
model2=GridSearchCV(svc,param_grid)

x_train=flat_data_train
x_train_phase= flat_data_train_phase # added the phase values 
#here I just convert nan to numbers but we might want to look deeper into where the nan values are coming from in the first place
x_train=np.nan_to_num(x_train)
x_train_phase = np.nan_to_num(x_train_phase)
#print(len(flat_data_train))
y_train=X_train_lable
#print(len(y_train))
x_test=flat_data_test
x_test_phase = flat_data_test_phase
x_test=np.nan_to_num(x_test)
x_test_phase= np.nan_to_num(x_test_phase)
y_test=X_test_lable

mag_model=model.fit(x_train,y_train)

phase_model = model2.fit(x_train_phase,y_train)

#and run it on the test dataset
y_pred = mag_model.predict(x_test)
y_phase_pred = phase_model.predict(x_test_phase)
print("these are the teacher outputs for the test dataset")
print(y_train)
print("predicted output vector for Amplitude: ")
print(y_pred)
print("predicted output vector for Phase: ")
print(y_phase_pred)




#calculate and print the accuracy
print("SVM: Amplitude accuracy:")
print(round(sum(y_pred==y_test)/len(y_test)*100,1))
print("SVM: Phase accuracy:")
print(round(sum(y_phase_pred==y_test)/len(y_test)*100,1))

#Added here the part for the linear regression (logistic regression since we dont have a continuous output)
#You will notice that I duplicate the models (for example: logR and logR2) they are identicall, 
#I do so in order to only train each model on amplitude or phase and not contaminate models
# The code can get shorter and optimized, but yeah for now I think its ok! 

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
lrGrid2 = GridSearchCV(logR,
                   param_grid = parameters_lr,
                   scoring='accuracy',
                   cv=2) # See above

lr_amp = lrGrid1
lr_phs = lrGrid2


lr_amp.fit(x_train,y_train)
lr_phs.fit(x_train_phase,y_train)



#check accuracy without parameter tuning for amplitude
test_np = logR.fit(x_train,y_train)
amp_pred_np= test_np.predict(x_test)
#check predictions with tunned model
amp_pred_p= lr_amp.predict(x_test)

#check accuracy without parameter tuning for phase
test_np = logR2.fit(x_train_phase,y_train)
phs_pred_np= test_np.predict(x_test_phase)
#check predictions with tunned model
phs_pred_p= lr_phs.predict(x_test_phase)

print("LR:Amplitude accuracy WITHOUT parameter tuning:")
print(round(sum(amp_pred_np==y_test)/len(y_test)*100,1))

print("LR:Phase accuracy WITHOUT parameter tuning:")
print(round(sum(phs_pred_np==y_test)/len(y_test)*100,1))

print("LR:Amplitude accuracy WITH parameter tuning:")
print(round(sum(amp_pred_p==y_test)/len(y_test)*100,1))

print("LR:Phase accuracy WITH parameter tuning:")
print(round(sum(phs_pred_p==y_test)/len(y_test)*100,1))


#visualize metrics for understanding in which classes we have bad predictions
cm = confusion_matrix(y_test, y_pred,normalize='true')
labels = ['0','1','2','3','4','5','6','7','8','9']
print(classification_report(y_test, y_pred)) 
fig, ax = plt.subplots(figsize=(8,8))
sns_plot = sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')

plt.title('SVM: Confusion Matrix of Amplitudes')
plt.show()
fig.savefig('Conf_Matrix_SVM_AMPL.png',bbox_inches='tight',dpi=150)

cm1 = confusion_matrix(y_test, y_phase_pred,normalize='true')
labels = ['0','1','2','3','4','5','6','7','8','9']
print(classification_report(y_test, y_phase_pred)) 
fig, ax = plt.subplots(figsize=(8,8))
sns_plot = sns.heatmap(cm1, annot=True, fmt='.2f', cmap='Blues',xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')

plt.title('SVM: Confusion Matrix of Phases')
plt.show()
fig.savefig('Conf_Matrix_SVM_PHS.png',bbox_inches='tight',dpi=150)

cm2 = confusion_matrix(y_test, amp_pred_p,normalize='true')
labels = ['0','1','2','3','4','5','6','7','8','9']
print(classification_report(y_test, amp_pred_p)) 
fig, ax = plt.subplots(figsize=(8,8))
sns_plot = sns.heatmap(cm2, annot=True, fmt='.2f', cmap='Blues',xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')

plt.title('LR: Confusion Matrix of Amplitudes')
plt.show()
fig.savefig('Conf_Matrix_LR_AMPL.png',bbox_inches='tight',dpi=150)

cm3 = confusion_matrix(y_test, phs_pred_p,normalize='true')
labels = ['0','1','2','3','4','5','6','7','8','9']
print(classification_report(y_test, phs_pred_p)) 
fig, ax = plt.subplots(figsize=(8,8))
sns_plot = sns.heatmap(cm3, annot=True, fmt='.2f', cmap='Blues',xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')

plt.title('LR: Confusion Matrix of Phases')
plt.show()
fig.savefig('Conf_Matrix_LR_PHS.png',bbox_inches='tight',dpi=150)


