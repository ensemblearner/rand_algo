from pyspark import SparkContext

from factorization.tsqr import TSQR
from rand_solvers.projections import *
from utils.matrix import Matrix
from utils.utils import *

if __name__ == "__main__":
    sc = SparkContext()
    n = 32768
    d = 10
    num_projections = 200
    matrix_list, matrix = gen_synth_data(n, d)
    rdd = sc.parallelize(matrix_list)

    spark_matrix = Matrix(rdd)
    num_experiments = 50
    c = 1
    projection = Projections(spark_matrix, sc)
    projection_matrix = projection.SpaFJLT(num_projections, c)

    tsqr = TSQR(projection_matrix, 3, sc)
    R = tsqr.tsqr()
    R = np.linalg.inv(R)
    lev = spark_matrix.get_rdd().map(lambda row: np.linalg.norm(np.dot(row, R))**2)
    sum_vel = d*np.array(lev.collect())/lev.reduce(add)
    app_lev = np.array(sum_vel)
    U, D, V = np.linalg.svd(matrix, full_matrices=False)
    lev_exact = np.sum(U**2, axis=1)
    error = np.linalg.norm(lev_exact - app_lev) / np.linalg.norm(lev_exact)
    print "error ", error

