__author__ = 'msingh'
from regression.least_squares import *
from pyspark import SparkContext
from utils.matrix import *

if __name__ == "__main__":

    sc = SparkContext(appName="test") # initiate an Spark object
    Ab = np.loadtxt('/Users/msingh/Downloads/semicoh_65536_100.txt')

    A = Ab[:, :-1]
    b = Ab[:, -1]

    A_rdd = sc.parallelize(A.tolist()) # Create a RDD for the rows of A.
    b_rdd = sc.parallelize(b)

    matrix_A = Matrix(A_rdd)
    matrix_b = Matrix(b_rdd)

    for technique in ['projection', 'sampling']:
        for method in ['srdht','cw','gaussian','rademacher']:
            ls = RandLeastSquares(matrix_A, matrix_b, sc=sc, k=2, c=3000, s=2000, method=method, technique=technique)
            x, costs = ls.fit(A, b)

