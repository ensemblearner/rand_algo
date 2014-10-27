from utils.matrix import *
from .projection_utils import *

class Projections(object):
    def __init__(self, matrix, sc, method='fjlt'):
        self.matrix = matrix
        self.method = method
        self.sc = sc

    def SpaFJLT(self, m, c):
        PiA = self.matrix.matrix.flatMap(lambda line: mapSFJLT(line, m, c)).reduceByKey(add).collect()
        #print PiA
        projection = np.array([item[1] for item in PiA])
        projection_rdd = self.sc.parallelize(projection)
        return Matrix(projection_rdd)
