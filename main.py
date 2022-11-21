from PIL import Image
import numpy as np
import click

import module.characteristic as ch
import module.imageFiltration as ifi
import module.histogramOperation as hop

def saveResult(new_matrix, save_path):
    Image.fromarray(new_matrix).save(save_path)
    Image.fromarray(new_matrix).show("New Image")


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
    functions = [ch.cmean_function, ch.cvariance_function, ch.cstdev_function, ch.cvarcoi_function, ch.casyco_function, ch.cflaco_function, ch.cvarcoii_function, ch.centropy_function]
    function_flags = [cmean, cvariance, cstdev, cvarcoi, casyco, cflaco, cvarcoii, centropy]

    # H0
    if histogram:
        hop.invokeHistogram(image_matrix, name)

    # C1 C2 C3 C4 C5 C6
    for i in range(len(functions)):
        if function_flags[i]: 
            hop.invokeFunction(function_names[i], functions[i], image_matrix)

    # H5
    if len(hhyper) != 0:
        lookup_table = hop.hhyper_function(image_matrix, hhyper)
        new_matrix = hop.replacePixel(image_matrix, lookup_table)
        saveResult(new_matrix,"./result/"+hop.nameParsing(name)+"_hhyper"+".bmp")
        hop.invokeHistogram(new_matrix, hop.nameParsing(name)+"_hhyper")

    # S6
    if slineid != -1:
        line_detection_kernels = [ifi.horizontal_kernel, ifi.vertical_kernel, ifi.diagonalUp_kernel, ifi.diagonalDown_kernel]
        new_image_matrix = ifi.convolutionOperation(image_matrix, line_detection_kernels[slineid])
        saveResult(new_image_matrix, "./result/"+hop.nameParsing(name)+"_slineid_"+str(slineid)+".bmp")
    
    # O2
    if orobertsii:
        new_matrix = ifi.robertsOperator(image_matrix)
        saveResult(new_matrix, "./result/"+hop.nameParsing(name)+"_orobertsii"+".bmp")


if __name__ == '__main__':
    operation()
    

