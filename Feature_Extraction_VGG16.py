#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:20:53 2020

@author: eduardosoares
"""
#import libraries
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import os


#Load VGG-16 model
model = VGG16(weights='imagenet', include_top= True )
layer_name = 'fc2'
intermediate_layer_model = keras.Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
intermediate_layer_model.summary()



#Load the data directory  where the images are stored
data_dir = 'iroads_lite/'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]
#Each folder becomes a different class

print(contents)
print(classes)

images = []
batch = []
labels = []

j =0


for each in classes: #Loop for the folders
  print("Starting {} images".format(each))
  class_path = data_dir + each
  files = os.listdir(class_path)
  
  for ii, file in enumerate(files, 1): #Loop for the imgs inside the folders
    # Load the images and resize it to 224X224(VGG-16 size)
    img = image.load_img(os.path.join(class_path, file), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Extract features using the VGG-16 structure
    features = intermediate_layer_model.predict(x)
    # Append features and labels
    batch.append(features[0])
    labels.append(file + ';' + str(j))
    print("finish {}".format(ii))
  j = j + 1

np_batch = np.array(batch)
np_labels = np.array(labels)

#Batch are the names of the files
#Labels
print(np_batch)
print(np_labels)

#Slpit the data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
np_batch, np_labels, test_size=0.4, random_state=0)

print(X_test.shape)
print(y_test.shape)

#Convert data to Pandas in order to save as .csv
import pandas as pd
data_df_X_train = pd.DataFrame(X_train)
data_df_y_train = pd.DataFrame(y_train)
data_df_X_test = pd.DataFrame(X_test)
data_df_y_test = pd.DataFrame(y_test)

print(data_df_X_train)

# Save file as .csv
data_df_X_train.to_csv('data_df_X_train_lite.csv',header=False,index=False)
data_df_y_train.to_csv('data_df_y_train_lite.csv',header=False,index=False)
data_df_X_test.to_csv('data_df_X_test_lite.csv',header=False,index=False)
data_df_y_test.to_csv('data_df_y_test_lite.csv',header=False,index=False)