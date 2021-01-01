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
        layerW = np.random.uniform(low=-2.0, high=2.0, size=(self.selfSize, self.preSize))
        layerB = np.random.uniform(low=-2.0, high=2.0, size=(self.selfSize, 1))
        layer = [layerW, layerB]
        if self.preSize >= 1:
            output=layer
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
        layerCounter = 1
        lastLayer = img
        for layer in self.strucArr:
            if layerCounter < len(self.strucArr):
                layerWeights = self.nn[layerCounter][0]*lastLayer
                layerWeightsBias = layerWeights + self.nn[layerCounter][1]
                activatedLayer = np.zeros((layerWeightsBias,1))
                for i in range(0, len(activatedLayer)):
                    activatedLayer[i] = sigmoid(layerWeightsBias[1])
                lastLayer = activatedLayer
                if layerCounter == index:
                    return lastLayer
                layerCounter +=1
    def run(self, inputImg):
        layerCounter = 1
        lastLayer = inputImg
        for layer in self.strucArr:
            if layerCounter < len(self.strucArr):
                layerWeights = np.matmul(self.nn[layerCounter][0], lastLayer)
                layerWeightsBias = np.add(layerWeights, self.nn[layerCounter][1])
                activatedLayer = np.zeros((len(layerWeightsBias),1))
                for i in range(0, len(activatedLayer)):
                    activatedLayer[i] = sigmoid(layerWeightsBias[1])
                lastLayer = activatedLayer
                if layerCounter == len(self.nn)-1:
                    return lastLayer
                layerCounter +=1

def backProp(nn, img):
    def transferDerivative(input):
        return input*(1.0-input)
    def errorInLayer(forwardError, weights):
        errorInLayer = np.zeros((1,1))
        return errorInLayer
        
    def inBackProp(oporationIndex, neuronError):
        for neuron in nn.nn[oporationIndex]:
            for counter in range(0, len(neuronError)):
                neuron.bias = neuron.bias + np.sum(neuronError)
                for index in range(0, len(neuron.weights[0])):
                    neuron.weights[0][index] = neuron.weights[0][index]+neuronError[counter]

    # Calculation of the error of final layer
    outputMAT = nn.run(img[0])
    targetMAT = np.array(outputMAT)
    for i in range(0, len(outputMAT)):
        if i == img[1]-1:
            targetMAT[i] = 1
        else:
            targetMAT[i] = 0
    neuronError = np.zeros((len(targetMAT), 1))
    for counter in range(0, len(targetMAT)):
        neuronError[counter] = (targetMAT[counter]-outputMAT[counter])*transferDerivative(outputMAT[counter])
    # Starting backprop calculation
    inBackProp(len(nn.nn)-1, neuronError)
    #errorInLayer(neuronError, weights)

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