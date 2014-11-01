__author__ = 'msingh'
from regression.least_squares import *
from pyspark import SparkContext
from utils.matrix import *



if __name__ == "__main__":

    sc = SparkContext("local[4]", "test") # initiate an Spark object
    Ab = np.loadtxt('/Users/msingh/Downloads/semicoh_65536_100.txt')
    Ab_rdd = sc.parallelize(Ab.tolist()) # Create a RDD for the rows of A.

    A = Ab[:,:-1]
    b = Ab[:,-1]

    matrix = Matrix(Ab_rdd)
    for technique in ['projection', 'sampling']:
        for method in ['cw', 'gaussian']:
            ls = RandLeastSquares(matrix, k=3, c=5000, s=1000, method=method, technique=technique)
            ls.fit(A,b)