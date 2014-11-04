from rand_solvers.rma_utils import *
import numpy.linalg as npl

class Projections(object):
    def __init__(self, **kwargs):
        #self.matrix = matrix
        self.method = kwargs.pop('method', 'cw')
        self.k = kwargs.pop('k')
        self.c = kwargs.pop('c')
        self.s = kwargs.pop('s', None)
        self.sc = kwargs.pop('sc', None)
        self.__validate()

    def __validate(self):
        if self.method not in Projections.METHODS:
            raise NotImplementedError('%s method not yet implemented' % self.method)
        if not self.c:
            raise ValueError('"c" param is missing')
        if not self.k:
            raise ValueError('"k" param is missing')
        if not self.sc:
            raise ValueError('"sc" param is missing')

    def execute(self, matrix, output_type):
        #self.matrix = matrix
        PA = self.__project(matrix)
        if output_type == 'solve':
            return self.__solve(PA)
        elif output_type == 'svd':
            return self.__svd(PA)
        else:
            return PA

    def __project(self, matrix):
        c = self.c
        k = self.k
        if self.method == 'cw':
            PA = matrix.matrix.flatMap(lambda line: cw_map(line, c, k)).reduceByKey(add).map(lambda x: (x[0][0], x[1].tolist())).groupByKey().collect()
        elif self.method == 'gaussian':
            PA = matrix.matrix.mapPartitions(lambda line: gaussian_map(line, c, k)).reduceByKey(add).map(lambda x: (x[0][0], x[1].tolist())).groupByKey().collect()
        elif self.method == 'rademacher':
            PA = matrix.matrix.mapPartitions(lambda line: rademacher_map(line, c, k)).reduceByKey(add).map(lambda x: (x[0][0], x[1].tolist())).groupByKey().collect()
        elif self.method == 'srdht':
            m = matrix.m
            idx = self.sc.parallelize(range(m))
            seed_s = np.random.randint(10000, size=k)
            #Note one can use zipWithIndex to substitute .zip(idx)
            PA = matrix.matrix.zip(idx).mapPartitions(lambda line: srdht_map(line, c, k, m, seed_s)).reduceByKey(add).map(lambda x: (x[0][0], x[1].tolist())).groupByKey().collect()

        return PA

    def __solve(self, PA):

        x = []
        for i in range(self.k):
            pab = [row for row in PA[i][1]]
            pab = np.array(pab)
            A = pab[:, :-1]
            b = pab[:, -1]
            x.append(np.linalg.lstsq(A, b)[0].tolist())
        return x

    def __svd(self, PA):

        N = []
        for i in range(self.k):
            pa = [row for row in PA[i][1]]
            pa = np.array(pa)
            [U, s, V] = npl.svd(pa, 0)
            N.append(V.transpose()/s)
        return N

    METHODS = ['cw', 'gaussian', 'rademacher', 'srdht']
