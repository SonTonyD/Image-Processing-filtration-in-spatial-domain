import math
import matplotlib.pyplot as plt
import numpy as np
import re

def rgbShape(image_matrix):
    reds, greens, blues = image_matrix[:,:,0], image_matrix[:,:,1], image_matrix[:,:,2]
    return [reds, greens, blues]

def displayHistogram(image_matrix, name, color):

    plt.plot(countPixel(image_matrix))

    parsedName = nameParsing(name)
    parsedName += color
    
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.title("Histogram of "+ parsedName)
    plt.savefig('./histogram/hist_'+ parsedName +'.png')
    plt.show()

def nameParsing(name):
    regex = re.compile(r'(?!images\b)\b(?!bmp\b)\b\w+')
    newName = regex.search(name).group()

    return newName

def invokeFunction(function_name, currentFunction, image_matrix):
    isColoredImage = False

    if len(image_matrix.shape) == 3:
        rgbMatrix = rgbShape(image_matrix)
        colors = ["red", "green", "blue"]
        isColoredImage = True
    
    if isColoredImage:
        for i in range(3):
            value = currentFunction(rgbMatrix[i])
            print(function_name, colors[i] ,"=",value)
    else:
        value = currentFunction(image_matrix)
        print(function_name ,"=",value)

def invokeHistogram(image_matrix, name):
    isColoredImage = False

    if len(image_matrix.shape) == 3:
        rgbMatrix = rgbShape(image_matrix)
        colors = ["red", "green", "blue"]
        isColoredImage = True

    if isColoredImage:
        for i in range(3):
            displayHistogram(rgbMatrix[i], name, colors[i])
    else:
        displayHistogram(image_matrix, name, "grey")

def hhyper_function(image_matrix, hhyper):
    width, height = image_matrix.shape[0], image_matrix.shape[1]
    gmin, gmax = hhyper
    countTable = countPixel(image_matrix)
    #normalizedHist = normalizedCountPixel(countTable, image_matrix)

    result_hist = np.zeros(256)
    exponent = 0

    for i in range(256):
        exponent = 0
        for j in range(i):
            exponent += countTable[j]
        exponent = exponent/(width*height)
        result_hist[i] = math.pow(gmin*(gmax/gmin), exponent)
    
    return result_hist

def countPixel(image_matrix):
    width, height = image_matrix.shape[0], image_matrix.shape[1]
    countTable = np.zeros(256)
    for i in range(width):
        for j in range(height):
            countTable[image_matrix[i,j]] += 1
    return countTable

def replacePixel(image_matrix, lookup_table):
    width, height = image_matrix.shape[0], image_matrix.shape[1]
    for i in range(width):
        for j in range(height):
            image_matrix[i,j] = round(lookup_table[image_matrix[i,j]])
    return image_matrix