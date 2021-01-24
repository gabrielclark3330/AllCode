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
class NetLayer:
    def __init__(self, selfSize, preSize):
        self.selfSize = selfSize
        self.preSize = preSize
    def inputF(self):
        layerW = np.random.uniform(low=-1.0, high=1.0, size=(self.selfSize, self.preSize))
        layerB = np.zeros((self.selfSize, 1))
        layerW = np.asmatrix(layerW)
        layerB = np.asmatrix(layerB)
        layer = [layerW, layerB]
        if self.preSize >= 1:
            output=layer
        else:
            return
        return output

class NuNet:
    def __init__(self, strucArr, lr):
        self.strucArr = strucArr
        self.nn = []
        self.lr = lr
        for i in range(0, len(self.strucArr)):
            if i == 0:
                preSize = 0
            else:
                preSize = strucArr[i-1]
            selfSize = strucArr[i]
            netLayer = NetLayer(selfSize, preSize)
            self.nn.append(netLayer.inputF())
    # running getRowActivation with the index of len(self.nn)-1 will do the same thing as running the network.
    def getRowActivation(self, img, index):
        img[0] = np.asmatrix(img[0])
        if index==0:
            return img
        layerCounter = 1
        lastLayer = img
        for layer in self.strucArr:
            if layerCounter < len(self.strucArr):
                layerWeights = np.matmul(self.nn[layerCounter][0], lastLayer)
                layerWeightsBias = np.add(layerWeights, self.nn[layerCounter][1])
                # Piece that activates the layers.
                if layerCounter != len(self.nn)-1:
                    activatedLayer = np.zeros((len(layerWeightsBias),1))
                    for i in range(0, len(activatedLayer)):
                        activatedLayer[i] = sigmoid(layerWeightsBias[i][0])
                    lastLayer = activatedLayer
                else:
                    lastLayer = layerWeightsBias
                if layerCounter == index:
                    return np.asarray(lastLayer)
                layerCounter +=1
    def run(self, img):
        img[0] = np.asmatrix(img[0])
        layerCounter = 1
        lastLayer = img
        for layer in self.strucArr:
            if layerCounter < len(self.strucArr):
                layerWeights = np.matmul(self.nn[layerCounter][0], lastLayer)
                layerWeightsBias = np.add(layerWeights, self.nn[layerCounter][1])
                # Piece that activates the layers.
                if layerCounter != len(self.nn)-1:
                    activatedLayer = np.zeros((len(layerWeightsBias),1))
                    for i in range(0, len(activatedLayer)):
                        activatedLayer[i] = sigmoid(layerWeightsBias.item(i,0))
                    lastLayer = activatedLayer
                else:
                    lastLayer = layerWeightsBias
                if layerCounter == len(self.nn)-1:
                    return np.asarray(lastLayer)
                layerCounter +=1

# Function takes the image and label and adjusts network to get closer to the right answer.
def backProp(nn, img):
    img[0] = np.asmatrix(img[0])
    # Pass this function the error in the forward layer and the weights behind it. It will give you the error in the previous layer.
    def errorInLayer(forwardError, weights):
        # print(f"Weights:{weights}")
        # print(f"forwardError:{forwardError}")
        # print(f"weights.shape{weights.shape}")
        errorInLayer = np.zeros((weights.shape[1], 1))
        for i in range(0, weights.shape[0]):
            for j in range(0, weights.shape[1]):
            # If statement that if the columns are == to 1 then dont use i on weights of forward error???
                errorInLayer[j][0] += weights.item((i,j))*forwardError.item((i,0))
        errorInLayer = np.asmatrix(errorInLayer)
        return errorInLayer
    # Function to adjust the weights and biasses.
    def inBackProp(oporationIndex, neuronError, endOutput):
        # newBias = originalBias+error*lr
        nn.nn[oporationIndex][1] = np.add(nn.nn[oporationIndex][1], nn.lr*neuronError)
        # newWeight = originalWeight+error*lr*originalWeight
        # print(f"OPIND:{oporationIndex}")
        weights = nn.nn[oporationIndex][0]
        # print(f"weights:{weights}")
        previousOuput = nn.getRowActivation(img[0], oporationIndex-1)
        # print(f"transposePRE:{np.transpose(previousOuput)}")
        # print(f"neuronError:{neuronError}")
        lrGradient = nn.lr*np.multiply(neuronError, np.multiply(endOutput, (1-endOutput)))
        # print(f"lrGradient:{lrGradient}")
        # print(f"matmul:{np.matmul(lrGradient, np.transpose(previousOuput))}")
        # print("---------------------------------------------------")
        nn.nn[oporationIndex][0] = np.add(weights, np.matmul(lrGradient, np.transpose(previousOuput)))
        #print(np.add(weights, np.matmul(lrGradient, np.transpose(previousOuput))))

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
        neuronError[counter] = (targetMAT[counter]-outputMAT[counter])
    # Starting backprop calculation for last layer
    for i in range(1,len(nn.nn)):
        inBackProp(len(nn.nn)-i, neuronError, nn.getRowActivation(img[0], len(nn.nn)-i))
        neuronError = errorInLayer(neuronError, nn.nn[len(nn.nn)-i][0])

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

################################################ XOR START #######################################

# XOR train data
trainData = [
    [np.array([[1],[1]]),0],
    [np.array([[1],[0]]),1],
    [np.array([[0],[1]]),1],
    [np.array([[0],[0]]),0]
]

# Test for xor problem
strucArr = [2, 2, 1]
testNN = NuNet(strucArr, .08)

for epocs in range(0, 10000):
    averageCost = 0
    shuffle(trainData)
    for img in trainData:
        backProp(testNN, img)
        result = testNN.run(img[0])
        cost = costFunction(result, img[1])
        averageCost += cost
    averageCost = averageCost/len(trainData)
    print(averageCost)
print(testNN.run([[1],[1]]),testNN.run([[1],[0]]),testNN.run([[0],[1]]),testNN.run([[0],[0]]))
######################## XOR END #####################################

# Lables for each of the images
key = {1:"circle", 2:"square", 3:"triangle"}

# Bring in an image as a 28*28(=784) array of values and add each image brought in to the imgSet array.
circles = grabImages("handDrawnShapes\circles")
squares = grabImages("handDrawnShapes\squares")
triangles = grabImages("handDrawnShapes\\triangles")

# Create an array of all images with a label attached.
dataSet = []
for img in circles:
    img = np.asarray(img)
    dataSet.append([img, 1])
for img in squares:
    img = np.asarray(img)
    dataSet.append([img, 2])
for img in triangles:
    img = np.asarray(img)
    dataSet.append([img, 3])
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
strucArr = [784, 4, 4, 3]
testNN = NuNet(strucArr, .1)
 
# Back prop testing last layer only
for epocs in range(0, 50):
    averageCostTrain = 0
    averageCostTest = 0
    for img in trainData:
        backProp(testNN, img)
        result = testNN.run(img[0])
        cost = costFunction(result, img[1])
        averageCostTrain += cost
    averageCostTrain = averageCostTrain/len(trainData)
    for img in testData:
        result = testNN.run(img[0])
        cost = costFunction(result, img[1])
        averageCostTest += cost
    averageCostTest = averageCostTest/len(trainData)
    print(f"Train:{averageCostTrain} Test:{averageCostTest}")

# Cost function
averageCost = 0
for img in testData:
    result = testNN.run(img[0])
    cost = costFunction(result, img[1])
    averageCost += cost
averageCost = averageCost/len(trainData)
print(averageCost)

