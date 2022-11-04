from PIL import Image
import numpy as np
import click
import math
import matplotlib.pyplot as plt



def cmean(image_matrix):
    width, height = image_matrix.shape[0], image_matrix.shape[1]
    sum = 0 

    for i in range(width):
        for j in range(height):
            sum += image_matrix[i,j]
    return sum/(width*height)

def cvariance(image_matrix):
    width, height = image_matrix.shape[0], image_matrix.shape[1]
    mean = cmean(image_matrix)

    sum = 0
    for i in range(width):
        for j in range(height):
            sum += math.pow(image_matrix[i,j]-mean,2)
    return sum/(width*height)

def cstdev(image_matrix):
    return math.sqrt(cvariance(image_matrix))

def cvarcoi(image_matrix):
    return cstdev(image_matrix)/cmean(image_matrix)

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
    mean = cmean(image_matrix)
    for i in range(width):
        for j in range(height):
            sum += math.pow(image_matrix[i,j] - mean, k) * ( countTable[image_matrix[i,j]]/(width*height) )
    return sum

def casyco(image_matrix):
    return standardizedMoment(image_matrix,3) / math.pow(cstdev(image_matrix),3)

def displayHistogram(image_matrix, name):
    image_matrix = image_matrix.flatten()
    plt.hist(image_matrix, color = "grey", bins=256)

    parsedName = nameParsing(name)
    
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.title("Histogram of "+ parsedName)
    plt.savefig('A_hist_'+ parsedName +'.png')
    plt.show()

def nameParsing(name):
    newName = name.split("\\")[1]
    newName = newName.split(".")[0]
    return newName

    


@click.command()
@click.option('--name', default="lena.bmp", help='path of the image. Example:--name="./Images/lenac.bmp"  ')
@click.option('--cmean', default=False, help='Can be true or false. Compute the mean value of all the pixels in the image')
@click.option('--cvariance', default=False, help='Can be true or false. Compute the variance value of all the pixels in the image')
@click.option('--histogram', is_flag=True, help='Save the histogram of the given image')
def operation(name, cmean, cvariance, histogram) :
    img = Image.open(name)
    image_matrix = np.array(img)

    if histogram:
        displayHistogram(image_matrix, name)





    

if __name__ == '__main__':
    operation()
    

