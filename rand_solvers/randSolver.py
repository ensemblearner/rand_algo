__author__ = 'msingh'
#This code performs randomized algorithms on large-scale matrix.
#Objectives include approximating leverage scores, solving least squares problems and CUR decomposition.

import sys, os

import numpy as np

from projections import *
from utils.basic import *
os.environ['SPARK_HOME'] = "/Users/msingh/Desktop/spark"
sys.path.append("/Users/msingh/Desktop/spark/python")
from pyspark import SparkContext

class randSolverSpark:
    def __init__(self, rowsA, n, d, colsA=[]):
        self.rowsA = rowsA    # RDD containing the rows of A
        self.colsA = colsA   # RDD containing the columns of A
        self.n = n  # The first dimension of A
        self.d = d  # The second dimension of A

    def apprLev(self, proj, m, c=1, secondProj='', r=1, getSumLev=0):
        # 'proj' denotes the embedding method to use. Here you can use either 'SRHT' or 'Sparse JLT'.
        # m denotes the embedding dimension.
        # c will only be used in "Sparse JLT'. Igonore it for now.
        # 'secondProj' is used when the second projection matrix is needed.
        # r denotes the second embedding dimension.
        # getSumLev is 1 if you want to return the sum of the leverage scores.

        if proj == 'SRHT':
            PiA = SRHT(self.colsA, m) # columns of A can be processed in parallel
        elif proj == 'SpaJLT':
            PiA = SpaFJLT(self.rowsA, m, c) # rows of A can be processed in parallel

        #print PiA
        R = np.linalg.qr(PiA, 'r')
        R = np.linalg.inv(R)

        if secondProj == 'Gaussian':
            G = np.random.normal(0,1/float(r),(self.d,r))
            R = np.dot(R, G)  # should change this line

        lev = self.rowsA.map(lambda row: np.linalg.norm(np.dot(row, R))**2)

        if getSumLev:
            return self.d/lev.reduce(add), R
        else:
            return self.d*np.array(lev.collect())/lev.reduce(add) # normalized the output to sum to d. This is slightly different from the theory

    def leastSquaresSampling(self, s, proj='', m=1, c=1, secondProj='', r=1, unif=False):
        if unif:
            n = self.n
            SAb = self.rowsAb.map(lambda line: unifSampling(line, n, s)).filter(lambda item: item != None).collect() #uniform sampling
            SAb = np.array(SAb)
        else:
            sumLev, R = self.apprLev(proj, m, c, secondProj, r, True)

            SAb = self.rowsAb.map(lambda line: sampling(line, R, sumLev, s)).filter(lambda item: item != None).collect()
            SAb = np.array(SAb)

        return np.dot(np.linalg.pinv(SAb[:,:-1]), SAb[:,-1])

def jiyans(A,sc,n,d,m):
    rowsA = sc.parallelize(A) # Create a RDD for the rows of A.
    colsA = sc.parallelize(A.T) # Create one of columns of A.

    obj = randSolverSpark(rowsA, n, d, colsA)

    #lev_appr = obj.apprLev('SpaJLT', m, secondProj='Gaussian', r=300) # Compute the approximate leverage scores.
    lev_appr = obj.apprLev('SpaJLT', m) # Not using the second projection matrix

    #lev_appr = obj.apprLev('SRHT', m) # If you use this, make sure n is some power of 2 as required by the Hadamard transform.
    lev_appr = np.array(lev_appr)

    U, D, V = np.linalg.svd(A, full_matrices=False)
    lev_exact = np.sum(U**2,axis = 1) # Compute the exact leverage scores.

    # Computing the estimation error
    print "------------------------------"
    print "------------------------------"
    print "Approximation error:"
    error =  np.linalg.norm( lev_exact - lev_appr) / np.linalg.norm(lev_exact)
    print error
    return error
