#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from skimage.transform import resize
import pandas as pd


# In[9]:


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataframe, path_to_img, batch_size, width, height, channels=3, shuffle=True):
        self.df = dataframe
        self.width = width
        self.height = height
        self.channels = channels
        self.path = path_to_img
        self.batch_size = batch_size//2
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df.index))

    def __len__(self):
        return int(np.ceil(len(self.indexes)/self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]

        # Initialize data lists
        X, Y = [], []

        for idx in indexes:
            x, y = self. get_sample(idx)
            X.append(x)
            Y.append(y)
            
        return np.array(X), np.array(Y)

    def get_sample(self, idx):
        df_row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.path, df_row['file']))
        image = image.resize((self.height, self.width))
        image = np.asarray(image)

        # Preprocessing steps
        image = self.norm(image)

        # Label
        label = 1
        return image, label

    def norm(self, image):
        image = image/255.0
        return image.astype(np.float32)
    


class DataGenerator2(tf.keras.utils.Sequence):

    def __init__(self, dataframe, path_to_img, batch_size, width, height, centroids, channels=3, shuffle=True):
        self.df = dataframe
        self.width = width
        self.height = height
        self.channels = channels
        self.path = path_to_img
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df.index))
        self.centroids = centroids

    def __len__(self):
        return int(np.ceil(len(self.indexes)/self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]

        # Initialize data lists
        X, Y = [], []

        for idx in indexes:
            x, y = self. get_sample(idx)
            X.append(x)
            Y.append(y)
            
        return np.array(X), np.array(Y)

    def get_sample(self, idx):
        df_row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.path, df_row['file']))
        image = image.resize((self.height, self.width))
        image = np.asarray(image)

        # Preprocessing steps
        image = self.norm(image)

        # Label
        label = df_row['genre']
        return self.quantize(image, label)

    def norm(self, image):
        image = image/255.0
        return image.astype(np.float16)
    
    def squared_euclidean_distance(self, a, b):
        b = tf.transpose(b)        
        a2 = tf.math.reduce_sum(tf.math.square(a), axis=1, keepdims=True)
        b2 = tf.math.reduce_sum(tf.math.square(b), axis=0, keepdims=True)
        ab = tf.linalg.matmul(a, b)
        return a2 - 2 * ab + b2

    def quantize(self, image, label):
        shape = tf.shape(image) # (height, width, color)
        x = tf.reshape(image, (-1, shape[2])) # (height * width, color)
        d = self.squared_euclidean_distance(x, self.centroids) # (height * width, centroids)
        sequence = tf.math.argmin(d, axis=1)  # (height * width)
        return sequence, label


# In[ ]:




