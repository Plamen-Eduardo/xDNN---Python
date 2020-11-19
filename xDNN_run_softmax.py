#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Please cite:
Angelov, P., & Soares, E. (2020). Towards explainable deep neural networks (xDNN). Neural Networks.

"""

###############################################################################
import pandas as pd

from xDNN_class_softmax import *
import numpy as np
from numpy import genfromtxt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time

# Load the files, including features, images and labels. 

X_train_file_path = r'data_df_X_train_dbnet_new.csv'
DBNet_las_1 = r'dbnet_point_cloud_train_1.csv'
DBNet_las_2 = r'dbnet_point_cloud_train_2.csv'
DBNet_las_3 = r'dbnet_point_cloud_train_3.csv'
DBNet_las_4 = r'dbnet_point_cloud_train_4.csv' 
DBNet_las_5 = r'dbnet_point_cloud_train_5.csv' 
y_train_file_path = r'data_df_y_train_dbnet_new.csv'


X_test_file_path = r'data_df_X_test_dbnet_new.csv'
DBNet_las_test = r'dbnet_point_cloud_test.csv'
y_test_file_path = r'data_df_y_test_dbnet_new.csv'

X_train =genfromtxt(X_train_file_path, delimiter=',')
X_train_DB1 = genfromtxt(DBNet_las_1, delimiter=',')
X_train_DB2 = genfromtxt(DBNet_las_2, delimiter=',')
X_train_DB3 = genfromtxt(DBNet_las_3, delimiter=',')
X_train_DB4 = genfromtxt(DBNet_las_4, delimiter=',')
X_train_DB5 = genfromtxt(DBNet_las_5, delimiter=',')
y_train = pd.read_csv(y_train_file_path, delimiter=',',header=None)


X_test = genfromtxt(X_test_file_path, delimiter=',')
X_test_DB = genfromtxt(DBNet_las_test, delimiter=',')
y_test = pd.read_csv(y_test_file_path, delimiter=',',header=None)

Numerical_test = X_test[:,4096:4098]
Numerical_train = X_train[:,4096:4098]

X_test = X_test[:,0:4096]
X_train = X_train[:,0:4096]
 
scaler_test = MinMaxScaler()
scaler_test.fit(X_test)
X_test = scaler_test.transform(X_test)

scaler_test_DB = MinMaxScaler()
scaler_test_DB.fit(X_test_DB)
X_test_DB = scaler_test_DB.transform(X_test_DB)

#X_test = np.hstack((X_test,X_test_DB,Numerical_test))

X_test= X_test_DB

X_train_DB= np.vstack((X_train_DB1, X_train_DB2, X_train_DB3,X_train_DB4,X_train_DB5))


scaler_train = MinMaxScaler()
scaler_train.fit(X_train)
X_train = scaler_train.transform(X_train)

scaler_train_DB = MinMaxScaler()
scaler_train_DB.fit(X_train_DB)
X_train_DB = scaler_train_DB.transform(X_train_DB)

#X_train = np.hstack((X_train,X_train_DB,Numerical_train))

X_train = X_train_DB
y_train = y_train

# Print the shape of the data

print ("###################### Data Loaded ######################")
print("Data Shape:   ")
print("X train: ",X_train.shape)
print("Y train: ",y_train.shape)
print("X test: ",X_test.shape)
print("Y test: ",y_test.shape)


pd_y_train_labels = y_train[1]
pd_y_train_images = y_train[0]

pd_y_test_labels = y_test[1]
pd_y_test_images = y_test[0]



# Convert Pandas to Numpy
y_train_labels = pd_y_train_labels.to_numpy().astype(np.int) 
y_train_images = pd_y_train_images.to_numpy()

y_test_labels = pd_y_test_labels.to_numpy().astype(np.int) 
y_test_images = pd_y_test_images.to_numpy()



# Model Learning
Input1 = {}

Input1['Images'] = y_train_images
Input1['Features'] = X_train
Input1['Labels'] = y_train_labels

Mode1 = 'Learning'

start = time.time()
Output1 = xDNN(Input1,Mode1)

end = time.time()

print ("###################### Model Trained ####################")

print("Time: ",round(end - start,2), "seconds")
###############################################################################

# Load the files, including features, images and labels for the validation mode

Input2 = {}
Input2['xDNNParms'] = Output1['xDNNParms']
Input2['Images'] = y_test_images 
Input2['Features'] = X_test
Input2['Labels'] = y_test_labels 



startValidation = time.time()
Mode2 = 'Validation' 
Output2 = xDNN(Input2,Mode2)
endValidation = time.time()

print ("###################### Results ##########################")


# Elapsed Time
print("Time: ", round(endValidation - startValidation,2), "seconds")
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test_labels , Output2['EstLabs'])
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test_labels , Output2['EstLabs'], average='micro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test_labels , Output2['EstLabs'],average='micro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test_labels , Output2['EstLabs'], average='micro')
print('F1 score: %f' % f1)
# kappa
kappa = cohen_kappa_score(y_test_labels , Output2['EstLabs'])
print('Cohens kappa: %f' % kappa)

# confusion matrix
matrix = confusion_matrix(y_test_labels , Output2['EstLabs'])
print("Confusion Matrix: ",matrix)

