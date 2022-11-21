import numpy as np

horizontal_kernel = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])
vertical_kernel = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
diagonalUp_kernel = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])
diagonalDown_kernel = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])

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