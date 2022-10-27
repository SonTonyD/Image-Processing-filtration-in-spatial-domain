from turtle import width
import unittest
import Task2
import numpy as np

class TestCharacteristics(unittest.TestCase):

    def test_cmean(self):
        image_matrix = np.array([[5,2],[1,4]])
        result = Task2.cmean(image_matrix)

        self.assertEqual(result, 3)

    def test_cvariance(self):
        image_matrix = np.array([[5,2],[1,4]])
        result = Task2.cvariance(image_matrix)

        self.assertEqual(result, 2.5)

    def test_cstdev(self):
        image_matrix = np.array([[5,2],[1,4]])
        result = Task2.cstdev(image_matrix)

        self.assertEqual(result, 1.5811388300841898)

    def test_cvarcoi(self):
        image_matrix = np.array([[5,2],[1,4]])
        result = Task2.cvarcoi(image_matrix)

        self.assertEqual(result, 0.5270462766947299)
    
    def test_countPixel(self):
        image_matrix = np.array([[5,4],[1,4]])

        testCountTable = np.zeros(256)
        testCountTable[1] = 1
        testCountTable[4] = 2
        testCountTable[5] = 1

        isEqual = np.array_equal(testCountTable, Task2.countPixel(image_matrix))

        self.assertTrue(isEqual)

    def test_standardizedMoment(self):
        image_matrix = np.array([[5,4],[1,4]])
        result = Task2.cvarcoi(image_matrix)

        self.assertEqual(result, 0.5270462766947299)

if __name__ == '__main__':
    unittest.main()