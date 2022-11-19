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

def replacePixel(image_matrix, lookup_table):
    width, height = image_matrix.shape[0], image_matrix.shape[1]
    result_matrix = np.copy(image_matrix)
    for i in range(width):
        for j in range(height):
            for k in range(256):
                if image_matrix[i,j] == k :
                    result_matrix[i,j] = round(lookup_table[k])
    return result_matrix
            

def convolutionOperation(image_matrix, kernel):
    width, height = image_matrix.shape[0], image_matrix.shape[1]
    copy_image_matrix = np.copy(image_matrix)

    for i in range(1, width-1):
        for j in range(1, height-1):
            #For a 3x3 kernel
            sum_kernel, weighted_sum = 0, 0
            for k in range(-1,2):
                for l in range(-1,2):
                    sum_kernel += abs(kernel[k+1,l+1])
                    weighted_sum += image_matrix[i+k,j+l]*kernel[k+1,l+1]
            copy_image_matrix[i,j] = weighted_sum/sum_kernel
    return copy_image_matrix

def robertsOperator(image_matrix):
    width, height = image_matrix.shape[0], image_matrix.shape[1]
    copy_image_matrix = np.copy(image_matrix)
    for i in range(1,width-1):
        for j in range(1,height-1):
            copy_image_matrix[i,j] = abs(int(image_matrix[i,j]) - int(image_matrix[i+1,j+1])) + abs(int(image_matrix[i,j+1]) - int(image_matrix[i+1,j]))
    return copy_image_matrix





@click.command()
@click.option('--name', default="lena.bmp", help='path of the image. Example:--name .\lenac.bmp  ')
@click.option('--cmean', is_flag=True, help='Compute the mean of all pixels')
@click.option('--cvariance', is_flag=True, help='Compute the variance')
@click.option('--histogram', is_flag=True, help='Save the histogram of the given image')
@click.option('--cstdev', is_flag=True, help='Compute the standard deviation')
@click.option('--cvarcoi', is_flag=True, help='Compute the variance coefficient')
@click.option('--cvarcoii', is_flag=True, help='Compute the variance coefficient II')
@click.option('--casyco', is_flag=True, help='Compute the asymetry coefficient (Skewness)')
@click.option('--cflaco', is_flag=True, help='Compute the flattening coefficient (Kurtosis)')
@click.option('--centropy', is_flag=True, help='Compute the entropy of the image')
@click.option('--hhyper', nargs=2, type=int, help='Histogram modification with hyperbolic final probability density function; 2 parameters: minBrightness maxBrightness')
@click.option('--slineid', default=-1, help='perform a line identification operation: 0 -> horizontal, 1 -> vertical, 2 -> diagUp, 3 -> diagDown')
@click.option('--orobertsii', is_flag=True, help='Roberts Operator (edge detection)')
def operation(name, cmean, cvariance, histogram, cstdev, cvarcoi, casyco, cflaco, cvarcoii, centropy, hhyper, slineid, orobertsii) :
    img = Image.open(name)
    image_matrix = np.array(img)

    function_names = ["cmean", "cvariance", "cstdev", "cvarcoi", "casyco", "cflaco", "cvarcoii", "centropy"]
    functions = [cmean_function, cvariance_function, cstdev_function, cvarcoi_function, casyco_function, cflaco_function, cvarcoii_function, centropy_function]
    function_flags = [cmean, cvariance, cstdev, cvarcoi, casyco, cflaco, cvarcoii, centropy]

    if histogram:
        invokeHistogram(image_matrix, name)

    for i in range(len(functions)):
        if function_flags[i]: 
            invokeFunction(function_names[i], functions[i], image_matrix)

    if len(hhyper) != 0:
        lookup_table = hhyper_function(image_matrix, hhyper)
        new_matrix = replacePixel(image_matrix, lookup_table)
        Image.fromarray(new_matrix).save("./images/"+name+"hhyper"+".bmp")
        Image.fromarray(new_matrix).show("New Image")
        invokeHistogram(new_matrix, name)

        

    if slineid != -1:
        horizontal_kernel = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])
        vertical_kernel = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
        diagonalUp_kernel = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])
        diagonalDown_kernel = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])

        line_detection_kernels = [horizontal_kernel, vertical_kernel, diagonalUp_kernel, diagonalDown_kernel]

        new_image_matrix = convolutionOperation(image_matrix, line_detection_kernels[slineid])
        Image.fromarray(new_image_matrix).save("./images/"+name+"slineid"+str(slineid)+".bmp")
        Image.fromarray(new_image_matrix).show("New Image")
    
    if orobertsii:
        new_matrix = robertsOperator(image_matrix)
        Image.fromarray(new_matrix).save("./images/"+name+"orobertsii"+".bmp")
        Image.fromarray(new_matrix).show("New Image")



    

if __name__ == '__main__':
    operation()
    

