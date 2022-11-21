import math
import numpy as np

def cmean_function(image_matrix):
    width, height = image_matrix.shape[0], image_matrix.shape[1]
    sum = 0 

    for i in range(width):
        for j in range(height):
            sum += image_matrix[i,j]
    return sum/(width*height)

def cvariance_function(image_matrix):
    width, height = image_matrix.shape[0], image_matrix.shape[1]
    mean = cmean_function(image_matrix)

    sum = 0
    for i in range(width):
        for j in range(height):
            sum += math.pow(image_matrix[i,j]-mean, 2)
    return sum/(width*height)

def cstdev_function(image_matrix):
    return math.sqrt(cvariance_function(image_matrix))

def cvarcoi_function(image_matrix):
    return cstdev_function(image_matrix)/cmean_function(image_matrix)

#histogram values (not normalized)
def countPixel(image_matrix):
    width, height = image_matrix.shape[0], image_matrix.shape[1]
    countTable = np.zeros(256)
    for i in range(width):
        for j in range(height):
            countTable[image_matrix[i,j]] += 1
    return countTable

#k-th central moment
def standardizedMoment(image_matrix, k):
    width, height = image_matrix.shape[0], image_matrix.shape[1]
    sum = 0
    countTable = countPixel(image_matrix)
    mean = cmean_function(image_matrix)
    for i in range(width):
        for j in range(height):
            sum += math.pow(image_matrix[i,j] - mean, k) * ( countTable[image_matrix[i,j]]/(width*height) )
    return sum

def casyco_function(image_matrix):
    return standardizedMoment(image_matrix,3) / math.pow(cstdev_function(image_matrix),3)

def cflaco_function(image_matrix):
    return standardizedMoment(image_matrix,4) / math.pow(cstdev_function(image_matrix),4)

def squareMatrix(image_matrix):
    width, height = image_matrix.shape[0], image_matrix.shape[1]

    for i in range(width):
        for j in range(height):
            image_matrix[i,j] = math.pow(image_matrix[i,j], 2)
    return image_matrix

def cvarcoii_function(image_matrix):
    tmp_image_matrix = image_matrix
    width, height = image_matrix.shape[0], image_matrix.shape[1]
    squared_matrix = squareMatrix(tmp_image_matrix)
    return cmean_function(squared_matrix)/math.pow(width*height , 2)

def normalizedCountPixel(countTable, image_matrix):
    width, height = image_matrix.shape[0], image_matrix.shape[1]
    for i in range(len(countTable)):
        countTable[i] = countTable[i]/(width*height)
    return countTable

def centropy_function(image_matrix):
    countTable = countPixel(image_matrix)
    normalizedHist = normalizedCountPixel(countTable, image_matrix)

    sum = 0
    for i in range(len(normalizedHist)):
        if normalizedHist[i] != 0:
            sum += normalizedHist[i]*math.log2(normalizedHist[i])
    
    entropy_value = -1*sum
    return entropy_value