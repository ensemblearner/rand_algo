__author__ = 'msingh'
#This code performs randomized algorithms on large-scale matrix.
#Objectives include approximating leverage scores, solving least squares problems and CUR decomposition.

import sys

import numpy as np
from rand_solvers.projections import *
#from rand_mat_alg_core import *
#from utils import *

from pyspark import SparkContext


if __name__ == "__main__":

    sc = SparkContext("local[4]", "test") # initiate an Spark object
    Ab = np.loadtxt('/Users/msingh/Downloads/semicoh_65536_100.txt')
    Ab_rdd = sc.parallelize(Ab.tolist()) # Create a RDD for the rows of A.

    matrix = Matrix(Ab_rdd)
    #matrix = Ab_rdd
    for technique in ['projection', 'sampling']:
        for method in ['cw', 'gaussian']:
            projection = Projections(matrix, k=3, c=5000, s=1000, method=method, technique=technique)
            print method," " ,technique," => ", projection.solution