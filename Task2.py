from PIL import Image
import numpy as np
import click
import math
import matplotlib.pyplot as plt


def rgbShape(image_matrix):
    reds, greens, blues = image_matrix[:,:,0], image_matrix[:,:,1], image_matrix[:,:,2]
    return [reds, greens, blues]


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
            sum += math.pow(image_matrix[i,j]-mean,2)
    return sum/(width*height)

def cstdev_function(image_matrix):
    return math.sqrt(cvariance_function(image_matrix))

def cvarcoi_function(image_matrix):
    return cstdev_function(image_matrix)/cmean_function(image_matrix)

#casyco
def countPixel(image_matrix):
    width, height = image_matrix.shape[0], image_matrix.shape[1]
    countTable = np.zeros(256)
    for i in range(width):
        for j in range(height):
            countTable[image_matrix[i,j]] += 1
    return countTable

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

def displayHistogram(image_matrix, name, color):
    image_matrix = image_matrix.flatten()
    plt.hist(image_matrix, color = color, bins=256)

    parsedName = nameParsing(name)
    parsedName += color
    
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.title("Histogram of "+ parsedName)
    plt.savefig('A_hist_'+ parsedName +'.png')
    plt.show()

def nameParsing(name):
    newName = name.split("\\")[1]
    newName = newName.split(".")[0]
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


    


@click.command()
@click.option('--name', default="lena.bmp", help='path of the image. Example:--name .\lenac.bmp  ')
@click.option('--cmean', is_flag=True, help='Compute the mean of all pixels')
@click.option('--cvariance', is_flag=True, help='Compute the variance')
@click.option('--histogram', is_flag=True, help='Save the histogram of the given image')
@click.option('--cstdev', is_flag=True, help='Compute the standard deviation')
@click.option('--cvarcoi', is_flag=True, help='Compute the variance coefficient')
@click.option('--casyco', is_flag=True, help='Compute the asymetry coefficient')
def operation(name, cmean, cvariance, histogram, cstdev, cvarcoi, casyco) :
    img = Image.open(name)
    image_matrix = np.array(img)

    function_names = ["cmean", "cvariance", "cstdev", "cvarcoi", "casyco"]
    functions = [cmean_function, cvariance_function, cstdev_function, cvarcoi_function, casyco_function]
    function_flags = [cmean, cvariance, cstdev, cvarcoi, casyco]

    for i in range(len(functions)):
        if function_flags[i]:
            invokeFunction(function_names[i], functions[i], image_matrix)

    if histogram:
        invokeHistogram(image_matrix, name)

if __name__ == '__main__':
    operation()
    

