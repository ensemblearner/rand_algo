from pyspark import SparkContext
from utils.utils import *
from operator import add

class TSQR(object):
    """
        Based on http://simons.berkeley.edu/sites/default/files/docs/782/gleichslides.pdf
    """
    def __init__(self, matrix, block_size, sc):
        self.matrix = matrix
        self.block_size = block_size
        self.sc = sc


    def tsqr(self):

        partitioned_rdd = partition_rdd(self.matrix.matrix, self.matrix.m, self.block_size, self.sc).map(lambda x: np.array(x))
        blocked_matrices = partitioned_rdd.map(lambda x: np.array(x))
        qr_blocks_mappers = blocked_matrices.map(lambda x: np.linalg.qr(x, 'r'))
        each_r_dim = qr_blocks_mappers.map(lambda x: np.shape(x)).take(1)[0]

        num_rows = qr_blocks_mappers.count()
        flat_list = qr_blocks_mappers.flatMap(lambda x: -1*x).flatMap(lambda x: x).collect()

        reshaped_matrix = np.reshape(np.array(flat_list), (each_r_dim[0]*num_rows, each_r_dim[1]))
        r = np.linalg.qr(reshaped_matrix, 'r')
        return r
