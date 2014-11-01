__author__ = 'msingh'
from regression.least_squares import *
from pyspark import SparkContext
from utils.matrix import *



if __name__ == "__main__":

    sc = SparkContext("local[4]", "test") # initiate an Spark object
    Ab = np.loadtxt('/Users/msingh/Downloads/semicoh_65536_100.txt')
    Ab_rdd = sc.parallelize(Ab.tolist()) # Create a RDD for the rows of A.

    matrix = Matrix(Ab_rdd)
    method='cw'
    technique='sampling'
    ls = RandLeastSquares( k=3, c=5000, s=1000, method=method, technique=technique)
    ls.fit(matrix,)