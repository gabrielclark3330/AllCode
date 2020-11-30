import os
import numpy as np 
import pandas as pd
import scipy as sp
import datetime
import re
import random
import cv2
import glob
from tqdm import tqdm
import pickle
import itertools

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from PIL import Image, ImageChops

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

### GPU CONTROL ###
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# training direct # "/home/alienware/Documents/GabrielClarkAnomalyDetection/CombinedCherryData/good/*.jpg"
# validation direct # "/home/alienware/Documents/GabrielClarkAnomalyDetection/78329394/good/*.jpg" 
# anomalys direct # "/home/alienware/Documents/GabrielClarkAnomalyDetection/78329394/rejects/no-boxes/*.jpg"

### GENERATOR CREATION - DATA CATEGORIZATION ###
batch_size = 85
train_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
train_generator = train_datagen.flow_from_directory(
    '/home/alienware/Documents/GabrielClarkAnomalyDetection/CombinedCherryData/good/',
    target_size=(96, 96),
    batch_size=batch_size,
    class_mode='input'
    )

test_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
validation_generator = test_datagen.flow_from_directory(
    '/home/alienware/Documents/GabrielClarkAnomalyDetection/78329394/good/',
    target_size=(96, 96),
    batch_size=batch_size,
    class_mode='input'
    )

anomaly_generator = test_datagen.flow_from_directory(
    '/home/alienware/Documents/GabrielClarkAnomalyDetection/78329394/rejects/',
    target_size=(96, 96),
    batch_size=batch_size,
    class_mode='input'
    )


model_filepath = 'Anomalymodels/FirstrunAutoE.keras' ### This is the path of the model you would like to run

model = keras.models.load_model(model_filepath) 


### MODEL TEST BY IMAGE RECONSTRUCTION ###
data_list = []
batch_index = 0
while batch_index <= train_generator.batch_index:
    data = train_generator.next()
    data_list.append(data[0])
    batch_index = batch_index + 1

predicted = model.predict(data_list[0])
no_of_samples = 4
_, axs = plt.subplots(no_of_samples, 2, figsize=(5, 8))
axs = axs.flatten()
imgs = []
for i in range(no_of_samples):
    imgs.append(data_list[i][i])
    imgs.append(predicted[i])
for img, ax in zip(imgs, axs):
    ax.imshow(img)
plt.show()

### ERROR CALCULATION ###  #remember you want to maximize the difference between these two
print(f"Error on validation set:{(model.evaluate_generator(validation_generator))*100}%, error on anomaly set:{(model.evaluate_generator(anomaly_generator))*100}%")


### EXTRACTING ENCODED IMAGE ###
# This network takes a an image which is 96x96x3 and compresses it down to a 3x3x3 tensor (copy of first encoder )
encoder_replica = Sequential()
encoder_replica.add(Conv2D(16, (3, 3), padding='same',activation='relu', input_shape=(96, 96, 3), weights=model.layers[0].get_weights()) )
encoder_replica.add(MaxPooling2D(pool_size=(4,4), padding='same'))
encoder_replica.add(Conv2D(8,(3, 3),activation='relu',  padding='same', weights=model.layers[2].get_weights()))
encoder_replica.add(MaxPooling2D(pool_size=(4,4), padding='same'))
encoder_replica.add(Conv2D(3,(3, 3),activation='relu',  padding='same', weights=model.layers[4].get_weights()))
encoder_replica.add(MaxPooling2D(pool_size=(2,2), padding='same'))
encoder_replica.summary()

# The SKLearn kernel density function only works with 1D arrays so we need to flatten the tensors created by the encoder
encoded_images = encoder_replica.predict_generator(train_generator)
encoded_images_flat = [np.reshape(img, (27)) for img in encoded_images]

validation_encoded = encoder_replica.predict_generator(validation_generator)
val_enc_flat = [np.reshape(img, (27)) for img in validation_encoded]

anom_encoded = encoder_replica.predict_generator(anomaly_generator)
anom_enc_flat = [np.reshape(img, (27)) for img in anom_encoded]

# Kernel Density Estimation of the encoded vectors
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(encoded_images_flat)
training_density_scores = kde.score_samples(encoded_images_flat) 
validation_density_scores = kde.score_samples(val_enc_flat)
anomaly_density_scores = kde.score_samples(anom_enc_flat)

# Plotting the density distributions of the training (normal), validation (normal) and anomalous images
# Ideally we want to see high separation between the normal and anomalous classes
plt.figure(figsize = (10,7))
plt.title('Distribution of Density Scores')
plt.hist(training_density_scores, 12, alpha=0.5, label='Training Normal')
plt.hist(validation_density_scores, 12, alpha=0.5, label='Validation Normal')
plt.hist(anomaly_density_scores, 12, alpha=0.5, label='Anomalies')
plt.legend(loc='upper right')
plt.xlabel('Density Score')
plt.show()


### CHECKING THE MODEL AGAINST KNOWNS ###

### ANOMALY FUNCTION ###
def check_anomaly(img_path):
    density_threshold = 18.375 # This threshold was chosen based on looking at the distribution of the density scores of the normal class (validation set)
    reconstruction_error_threshold = 0.04 # This threshold was chosen based on looking at the distribution of reconstruction errors of the good class
    img  = Image.open(img_path)
    img = np.array(img.resize((96,96), Image.ANTIALIAS))
    img = img / 255
    encoded_img = encoder_replica.predict([[img]]) # Create a compressed version of the image using the encoder
    encoded_img = [np.reshape(img, (27)) for img in encoded_img] # Flatten the compressed image
    density = kde.score_samples(encoded_img)[0] # get a density score for the new image
#     print(f'density: {density}')
    reconstruction = model.predict([[img]])
    reconstruction_error = model.evaluate([reconstruction],[[img]], batch_size = 1)
#     print(f'reconstruction_error: {reconstruction_error}')
    if density < density_threshold or reconstruction_error > reconstruction_error_threshold:
        return True
    else:
        return False


# validation direct # "/home/alienware/Documents/GabrielClarkAnomalyDetection/78329394/good/*.jpg" 
# anomalys direct # "/home/alienware/Documents/GabrielClarkAnomalyDetection/78329394/rejects/no-boxes/*.jpg"
# Check what proportion of good are classified as anomalous (want close to 0)
good_test = []
for (dirpath, dirnames, filenames) in os.walk('/home/alienware/Documents/GabrielClarkAnomalyDetection/78329394/good/'):
    for x in filenames:
        if x.endswith(".jpg"):
            good_test.append(os.path.join(dirpath, x))

good_predictions = []
for file in good_test:
    good_predictions.append(check_anomaly(file))
print(sum(good_predictions)/len(good_predictions))

# Check what proportion of bad get classed as anomalous.
bad_test = []
for (dirpath, dirnames, filenames) in os.walk('/home/alienware/Documents/GabrielClarkAnomalyDetection/78329394/rejects/no-boxes/'):
    for x in filenames:
        if x.endswith(".jpg"):
            bad_test.append(os.path.join(dirpath, x))

bad_predictions = []
for file in bad_test:
    bad_predictions.append(check_anomaly(file))
print(sum(bad_predictions)/len(bad_predictions))