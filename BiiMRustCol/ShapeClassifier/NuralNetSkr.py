import numpy as np
import cv2
import PIL
from PIL import Image
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from random import shuffle
from random import seed
from random import uniform
from random import random
from random import randint
import math


# The clases that compose the neural network
class Neuron:
    def __init__(self, preSize):
        self.weights = np.random.uniform(low=-2.0, high=2.0, size=(1, preSize))
        self.bias = uniform(-3,3)
    def inputN(self, input): # Input is a previous layer of neurons
        output = (np.dot(self.weights, input)) + self.bias
        output = sigmoid(output)
        return output

class NetLayer:
    def __init__(self, selfSize, preSize):
        self.selfSize = selfSize
        self.preSize = preSize
    def inputF(self):
        layer = []
        if self.preSize >= 1:
            for i in range(0, self.selfSize):
                layer.append(Neuron(self.preSize))
            output = layer
        else:
            return
        return output

class NuNet:
    def __init__(self, strucArr):
        self.strucArr = strucArr
        self.nn = []
        for i in range(0, len(self.strucArr)):
            if i == 0:
                preSize = 0
            else:
                preSize = strucArr[i-1]
            selfSize = strucArr[i]
            netLayer = NetLayer(selfSize, preSize)
            self.nn.append(netLayer.inputF())
    def getRowActivation(self, img, index):
        layerCounter = 2
        lastLayer = img
        currentLayer = self.nn[1]
        store = []
        for layer in self.strucArr:
            if layerCounter <= len(self.strucArr):
                for neuron in currentLayer:
                    neuronCounter = 0
                    store.append(float(neuron.inputN(lastLayer)))
                    neuronCounter += 1
                np.array(store)
                np.reshape(store, (len(store), 1))
                lastLayer = store
                store = []
                if layerCounter == index + 1:
                    return lastLayer
                currentLayer = self.nn[layerCounter]
                layerCounter +=1
    def run(self, inputImg):
        layerCounter = 2
        lastLayer = inputImg
        currentLayer = self.nn[1]
        store = []
        for layer in self.strucArr:
            if layerCounter <= len(self.strucArr):
                for neuron in currentLayer:
                    neuronCounter = 0
                    store.append(float(neuron.inputN(lastLayer)))
                    neuronCounter += 1
                np.array(store)
                np.reshape(store, (len(store), 1))
                lastLayer = store
                store = []
                if layerCounter == len(self.nn):
                    return lastLayer
                currentLayer = self.nn[layerCounter]
                layerCounter +=1

def backProp(nn, img):
    def transferDerivative(input):
        return input * (1.0 - input)
    resultMAT = nn.run(img[0])
    cResultID = img[1]
    cost = costFunction(resultMAT, img[1])
    nnLength = len(nn.nn)
    cResultMAT = np.array(resultMAT) # the next cost result matrix will just contain the error values for the previous layer of neurons
    for i in range(0, len(resultMAT)):
        if i == cResultID-1:
            cResultMAT[i] = 1
        else:
            cResultMAT[i] = 0
    def inBackProp(lastLayerIndex, layerIndex, cResultMAT):
        for neuron in nn.nn[layerIndex]:
            for counter in range(0, len(cResultMAT)):
                error = (cResultMAT[counter] - resultMAT[counter]) * transferDerivative(resultMAT[counter])
                # Maybe implement step size by changing error with scalor value!!!!!!!!!
                neuronActivations = testNN.getRowActivation(img[0], layerIndex - 1)
                editedActivations = neuronActivations
                editedActivations[:] = [nurAc - .5 for nurAc in neuronActivations]
                for index in range(0, len(neuron.weights[0])):
                    neuron.weights[0][index] = neuron.weights[0][index] + error 
    inBackProp(nnLength-1, nnLength-1, cResultMAT)

# The sigmoid function with input x and output y
def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

# Converts an image into a tensor
def imgToTensor(img):
    rows = len(img)
    cols = len(img[0])
    size = rows*cols
    img = np.reshape(img, (size, 1))
    return img

# Converts a color img (numpy array with pixles as tupple of red green blue) to a normalized black and white img numpy array
def blackAndWhiteNorm(img):
    rows = len(img)
    cols = len(img[0])
    bAWImg = np.zeros((rows,cols))
    for row in range(0,rows):
        for col in range(0,cols):
            pix = img[row][col]
            newPix = (int(pix[0]) + int(pix[1]) + int(pix[2]))/(3*255)
            bAWImg[row][col] = newPix
    return bAWImg

# Inverts a matrix (image) of 0-1s
def invert(img):
    rows = len(img)
    cols = len(img[0])
    iNVImg = np.zeros((rows,cols))
    for row in range(0,rows):
        for col in range(0,cols):
            pix = img[row][col]
            newPix = 1 - pix
            iNVImg[row][col] = newPix
    return iNVImg

# Grabs all images from a directory, converts them into a np.array, and adds them to a imgSet array.
def grabImages(path):
    imgSet = []
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            img = Image.open(path + "\\" + filename)
            numpyData = np.asarray(img)
            numpyData = blackAndWhiteNorm(numpyData)
            numpyData = invert(numpyData)
            imgSet.append(numpyData)
    imgSet = np.array(imgSet)
    return imgSet

def costFunction(resultNN, resultNum):
    resultID = np.array(resultNN)
    for i in range(0, len(resultNN)):
        if i == resultNum-1:
            resultID[i] = 1
        else:
            resultID[i] = 0
    cost = 0.0
    for i in range(0, len(resultNN)):
        cost += (resultNN[i] - resultID[i])**2
    return cost

# This version of bubble sort is built for the pair array in backprop
def bubbleSort(arr):
    trigger = True
    counter = 0
    while trigger == True:
        trigger = False
        counter += 1
        for i in range(0, len(arr)-counter):
            if arr[i][0] > arr[i+1][0]:
                holder = arr[i]
                arr[i] = arr[i+1]
                arr[i+1] = holder
                trigger = True
    return arr

# Lables for each of the images
key = {1:"circle", 2:"square", 3:"triangle"}

# Bring in an image as a 28*28(=784) array of values and add each image brought in to the imgSet array.
circles = grabImages("handDrawnShapes\circles")
squares = grabImages("handDrawnShapes\squares")
triangles = grabImages("handDrawnShapes\\triangles")

# Create an array of all images with a label attached.
dataSet = []
for img in circles:
    dataSet.append([img, 1])
for img in squares:
    dataSet.append([img, 2])
for img in triangles:
    dataSet.append([img, 3])
shuffle(dataSet)
shuffle(dataSet)
dataSet = np.asarray(dataSet)

# Viewing a random picture and lable
plotIndex = randint(0, 300)
plt.imshow(dataSet[plotIndex][0])
plt.title(key[dataSet[plotIndex][1]])
plt.colorbar()
plt.show()
print(key[1])

# Transforming each image in our dataSet into a tensor for network
counter = 0
for img in dataSet:
    dataSet[counter][0] = imgToTensor(img[0])
    counter += 1

# Split shuffled data into a test (25%) and train (75%) set
train = .75
test = .25
cut = math.floor((len(dataSet) * train))
trainData = dataSet[:cut]
testData = dataSet[cut:]

# Neural net set up
strucArr = [784, 16, 16, 3]
testNN = NuNet(strucArr)

# Cost function
averageCost = 0
for img in trainData:
    result = testNN.run(img[0])
    cost = costFunction(result, img[1])
    averageCost += cost
averageCost = averageCost/len(trainData)
print(averageCost)
 
# Back prop testing last layer only
for epocs in range(0, 50):
    averageCost = 0
    for img in trainData:
        backProp(testNN, img)
        result = testNN.run(img[0])
        cost = costFunction(result, img[1])
        averageCost += cost
    averageCost = averageCost/len(trainData)
    print(averageCost)