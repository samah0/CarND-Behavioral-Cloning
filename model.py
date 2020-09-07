#!/usr/bin/env python
# coding: utf-8

import csv
import cv2
import pickle
import math
from math import ceil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from scipy import ndimage

from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


def generator(samples, dirname, batch_size=32):
    """ dirname: path to the selected directory where images and driving_logo.csv file are saved """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                path = dirname + '/IMG/'
                try:
                 
                  steering_center = float(batch_sample[3])
                  
                  # Create adjusted steering measurements for the side camera images
                  correction = 0.1
                  steering_left = steering_center + correction
                  steering_right = steering_center - correction
                  
                  # Read in images from center, left and right cameras
                  img_center = ndimage.imread(path + batch_sample[0].split('/')[-1])
                  img_left = ndimage.imread(path + batch_sample[1].split('/')[-1])
                  img_right = ndimage.imread(path + batch_sample[2].split('/')[-1])
                  
                  # Add images and angles to data set
                  images.extend([img_center, img_left, img_right])
                  angles.extend([steering_center, steering_left, steering_right])
                  
                except OSError as e:
                  print('Error|',e)
                  
            # Convert images and measurements into numpy arrays
            X_train = np.array(images)
            y_train = np.array(angles)
            
            #print('X_train shape|',X_train.shape)
            #print('y_train shape|',y_train.shape)
            
            yield sklearn.utils.shuffle(X_train, y_train)

def split_data(dirname):
    """ dirname: path to the selected directory where images and driving_logo.csv file are saved """
    
    # Read the driving_logo.csv file and save it's lines into a list
    samples = []
    with open(dirname + '/driving_log.csv') as csvfile:
       reader = csv.reader(csvfile)
       for line in reader:
         samples.append(line)
         
    # Split data into training and validation sets     
    train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)
   
    return train_samples, validation_samples
    
    
def train_model(dirname):
    """ dirname: path to the selected directory where images and driving_logo.csv file are saved """
    
    # Split data into train and validation sets
    train_samples, validation_samples = split_data(dirname)
    
    # Set our batch size
    batch_size=32
 
    # compile and train the model using the generator function
    train_generator = generator(train_samples, dirname,batch_size=batch_size)
    validation_generator = generator(validation_samples, dirname, batch_size=batch_size)

    model = Sequential()
    model.add(Lambda(lambda x : x /255.0 - 0.5, input_shape=(160,320,3)))
    
    # set up cropping2D layer to trim image to only see section with road
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    
    # set up convolutional layers
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    
    # reshape last convolutional layer
    model.add(Flatten(input_shape=(64,3,3)))
    
    # set up fully connected layers
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    
    # output layer
    model.add(Dense(1))
    """
    # Display shapes of each layer
    print('shapes|')
    for layer in model.layers:
       print(layer.output_shape)
    """   
    # compile the model with loss = MSE and Adam optimizer
    model.compile(loss='mse', optimizer='adam')
    
    # train the netwrok
    history = model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size),epochs=5, verbose=1)
    
    # save the model
    model.save('./model.h5')

    return history
    
def plot_loss(history):

   # print the keys contained in the history object
   print(history.history.keys())

   # plot the training and validation loss for each epoch
   plt.plot(history.history['loss'])
   plt.plot(history.history['val_loss'])
   plt.title('model mean squared error loss')
   plt.ylabel('mean squared error loss')
   plt.xlabel('epoch')
   plt.legend(['training set', 'validation set'], loc='upper right')
   # save training history as png image
   #plt.savefig('./training_history.png')
   plt.show()

if __name__ == '__main__':

    # Path to the selected directory where to save images and driving_logo.csv file
    dirname = '/Users/samah/behaivoral_cloning/data'
    
    # Train the model and return a history object to be used to plot the loss function 
    history = train_model(dirname)
    
    # Plot the training and validation loss for each epoch
    plot_loss(history)
