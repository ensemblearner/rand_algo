__author__ = 'msingh'
#This code performs randomized algorithms on large-scale matrix.
#Objectives include approximating leverage scores, solving least squares problems and CUR decomposition.

from rand_solvers.projections import *
from utils.matrix import *

from pyspark import SparkContext
import sys

if __name__ == "__main__":

    sc = SparkContext("local[4]", "test") # initiate an Spark object
    Ab = np.loadtxt('/Users/msingh/Downloads/semicoh_65536_100.txt')
    print Ab.shape
    Ab_rdd = sc.parallelize(Ab.tolist()) # Create a RDD for the rows of A.

    matrix = Matrix(Ab_rdd)
    #matrix = Ab_rdd
    #for technique in ['sampling']:
    for technique in ['projection', 'sampling']:
        for method in ['cw', 'gaussian']:
            projection = Projections(matrix, k=3, c=5000, s=1000, method=method, technique=technique)
            result = projection.execute()
            print method, " ", technique, " => ", result, " ", len(result[0])
            sys.exit()