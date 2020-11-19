#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 22:43:03 2020

@author: eduardosoares
"""

import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
 
 #Class definition for Feature extraction 
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        VGG = models.vgg16(pretrained=True) #Network definition (pretrained)
        self.feature = VGG.features #features are extracted based on VGG weights
        self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3]) # Fully Connected Layer
        pretrained_dict = VGG.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)
 
    def forward(self, x):
        output = self.feature(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output

model = Encoder()
 
def extractor(img_path, net):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),# VGG-16 Takes 224x224 images as input, so we resize all of them
        transforms.ToTensor()]
    )
 
    img = Image.open(img_path)
    img = transform(img)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)

    y = net(x).cpu()
    y = torch.squeeze(y)
    y = y.data.numpy()

    return y


#Load the data directory  where the images are stored
data_dir = '1/imgs/'
contents = os.listdir(data_dir)
data_df = pd.read_csv("Y_dbnet.csv", header = None)
classes = data_df[2]
#Each folder becomes a different class

print(contents)
print(classes)


images = []
batch = []
labels = []

j =0

for each in classes: #Loop for the folders
  print("Starting {} images".format(each))
  class_path = data_dir 
  files = os.listdir(class_path)
  
  for ii, file in enumerate(files, 1):

    img = os.path.join(class_path, file)
    features = extractor(img, model) # Extract features using the VGG-16 structure
    batch.append(features)
    images.append(file)
    labels.append(str(j))
    print("finish {}".format(ii))
  j = j + 1    #Class iterator
np_batch = np.array(batch)
np_labels = np.array(labels)
np_images = np.array(images)

np_labels_T = np_labels.reshape(-1,1)
np_images_T = np_images.reshape(-1,1)
print(np_labels_T)


np_images_labels = np.hstack((np_images_T,np_labels_T))
print(np_images_labels)

#Slpit the data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
np_batch, np_images_labels, test_size=0.3, random_state=0)

#Convert data to Pandas in order to save as .csv
import pandas as pd
data_df_X_train = pd.DataFrame(X_train)
data_df_y_train = pd.DataFrame(y_train)
data_df_X_test = pd.DataFrame(X_test)
data_df_y_test = pd.DataFrame(y_test)

print(data_df_X_train)

# Save file as .csv
data_df_X_train.to_csv('data_df_X_train_covid.csv',header=False,index=False)
data_df_y_train.to_csv('data_df_y_train_covid.csv',header=False,index=False)
data_df_X_test.to_csv('data_df_X_test_covid.csv',header=False,index=False)
data_df_y_test.to_csv('data_df_y_test_covid.csv',header=False,index=False)
