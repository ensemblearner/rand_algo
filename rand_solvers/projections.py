from utils.matrix import *
from .projection_utils import *
import numpy.linalg as npl
class Projections(object):
    def __init__(self, matrix, **kwargs):
        self.matrix = matrix
        self.method = kwargs.pop('method', 'cw')
        self.k = kwargs.pop('k')
        self.c = kwargs.pop('c')
        self.technique = kwargs.pop('technique')
        self.solution = self.__execute()
        self.__validate()

    def __validate(self):
        if self.method not in Projections.METHODS:
            raise NotImplementedError("method not yet implemented")
        if not self.c:
            raise ValueError("'c' param is missing")
        if not self.k:
            raise ValueError("'k' param is missing")
        if self.technique not in ['projection', 'sampling']:
            raise NotImplementedError('%s technique not implemented yet' % self.technique)

    def __execute(self):
        PA = self.__project()
        return self.__solve(PA)


    def __project(self):
        PA_inter = self.__cw(self.k, self.c) if self.method == 'cw' else self.__gaussian(self.k, self.c)
        PA = map((lambda (x, y): (x, list(y))), sorted(PA_inter.collect()))
        return PA

    def __solve(self, PA):
        data = self.random_projection(PA) if self.technique == 'projection' else self.random_sample(PA)
        return data

    def __cw(self, k, c):
        PA_inter = self.matrix.matrix.flatMap(lambda line: cw_map(line, c, k)).reduceByKey(add).map(lambda x: (x[0][0],x[1].tolist())).groupByKey()
        return PA_inter

    def __gaussian(self, k, c):
         PA_inter = self.matrix.matrix.mapPartitions(lambda line: gaussian_map(line, c, k)).reduceByKey(add).map(lambda x: (x[0][0], x[1].tolist())).groupByKey()
         return PA_inter

    def random_projection(self, PA):

        x = []
        for i in range(self.k):
            PAb = np.array(PA[i][1])
            A = PAb[:, :-1]
            b = PAb[:, -1]
            x.append(np.linalg.lstsq(A, b)[0].tolist())
        return x

    def random_sample(self, PA):
        N = []
        for i in range(self.k):
            PAb = np.array(PA[i][1])
            [U, s, V] = npl.svd(PAb, 0)
            N.append(V.transpose()/s)
        return N


    METHODS = ['cw', 'gaussian']