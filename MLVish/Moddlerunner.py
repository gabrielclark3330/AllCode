import numpy as np 
import pandas as pd
import scipy as sp
import datetime
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras.utils import *

from PIL import Image
from io import BytesIO
import os
import random
import pickle
import itertools

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import cv2
import glob


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

### IMAGE COLLECTION ###
imageStatus = []
bImages = []
gImages = []
fname = []
def image_format(jpg_location):
    files = glob.glob (jpg_location)
    I = 0
    for myFile in tqdm(files):
        head, tail = os.path.split(myFile)
        fname.append(tail)
        image = cv2.imread(myFile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = array_to_img(image).crop((430, 375, 965, 910))
        image = image.resize((224,224))
        numpy_img = img_to_array(image)
        if jpg_location.find("rejects") != -1:
            bImages.append(numpy_img.astype('float16'))
            imageStatus.append("Bad")
        if jpg_location.find("good") != -1:
            gImages.append(numpy_img.astype('float16')) 
            imageStatus.append("Good")
        I = I + 1
        if I == 2000:
            break
    return bImages, gImages, imageStatus, fname

# 78329378 original set 83 bad
# 78329394 200sum bad
# CombinedCherryData
### BAD IMAGES ###
holder = []
bImages, gImages, imageStatus, holder = image_format("/home/alienware/Documents/GabrielClarkAnomalyDetection/78329394/rejects/no-boxes/*.jpg")
print('bad images shape:', np.array(bImages).shape)

### GOOD IMAGES ###
bImages, gImages, imageStatus, fname = image_format("/home/alienware/Documents/GabrielClarkAnomalyDetection/78329394/good/*.jpg")
# shape checks
print('good images shape:', np.array(gImages).shape)
imageStatus = np.vstack(imageStatus)
print('imageStatus shape:', np.array(imageStatus).shape)
images = []
images = np.concatenate((bImages, gImages), axis=0)
print('images shape:', np.array(images).shape)
fname = np.vstack(fname)
print('fname shape:', np.array(fname).shape)

### DATAFRAME ###
df = pd.DataFrame(data= imageStatus, columns = ['labels'])
df['fnames'] = fname

### IMAGE SORT ###
test_images = images
test_labels = df['labels']

### LOADING THE NEW MODEL ###
new_model = tf.keras.models.load_model('saved_model/shunt_heatm_nonautoe/AmazingHeatMaps-badtraing')

### ARCHITECTURE CHECK ###
new_model.summary()



### model Check ###
Y = to_categorical((df.labels.values == 'Bad') +0, num_classes=2)
X_train, X_test, y_train, y_test = train_test_split(images, Y, random_state = 42, test_size=.95)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), 
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)

model = new_model
model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
print(classification_report(np.argmax(y_test,axis=1), np.argmax(model.predict(X_test/255),axis=1)))


### CONFUSION MATRIX ###

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize = 14)

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

cnfClasses = ["Good","Bad"]
cnf_matrix = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(model.predict(X_test/255),axis=1))
plt.figure(figsize=(7,7))
plot_confusion_matrix(cnf_matrix, classes=cnfClasses, title="Confusion matrix")
plt.show()


### HEATMAP ###

def plot_activation(img):
  
    pred = model.predict(img[np.newaxis,:,:,:])
    pred_class = np.argmax(pred)

    weights = model.layers[-1].get_weights()[0] #weights last classification layer
    class_weights = weights[:, pred_class]

    intermediate = Model(model.input, model.get_layer("block5_conv3").output)
    conv_output = intermediate.predict(img[np.newaxis,:,:,:])
    conv_output = np.squeeze(conv_output)

    h = int(img.shape[0]/conv_output.shape[0])
    w = int(img.shape[1]/conv_output.shape[1])

    activation_maps = sp.ndimage.zoom(conv_output, (h, w, 1), order=1)
    out = np.dot(activation_maps.reshape((img.shape[0]*img.shape[1], 512)), class_weights).reshape(img.shape[0],img.shape[1])

    
    plt.imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
    plt.imshow(out, cmap='jet', alpha=0.35)
    plt.title("Predicted Label: "+'Bad' if pred_class == 1 else "Predicted Label: "+ 'Good')
    plt.colorbar().ax.set_ylabel("Error Concentration", rotation= -90, va="bottom")
    return out, pred_class


### CHECKING HEAT MAP ###
plot_activation(X_test[100]/255)
plt.show(block=True)
plot_activation(X_test[99]/255)
plt.show(block=True)
plot_activation(X_test[98]/255)
plt.show(block=True)
plot_activation(X_test[101]/255)
plt.show(block=True)
plot_activation(X_test[102]/255)
plt.show(block=True)
plot_activation(X_test[103]/255)
plt.show(block=True)