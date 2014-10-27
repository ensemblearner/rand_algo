import numpy as np
from utils.matrix import *
from utils.basic import *
from .projection_utils import *

class Projections(object):
    def __init__(self, matrix, method='fjlt'):
        self.matrix = matrix
        self.method = method

    def SpaFJLT(self, m, c):
        #A is an RDD
        PiA = self.matrix.matrix.flatMap(lambda line: mapSFJLT(line, m, c)).reduceByKey(add).collect()
        #print PiA
        return np.array([item[1] for item in PiA])


