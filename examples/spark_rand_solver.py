from pyspark import SparkContext

from factorization.tsqr import TSQR
from utils.matrix import Matrix
from utils.utils import *
from rand_solvers.randSolver import *
from rand_solvers.projections import *
#import matplotlib.pyplot   as plt

if __name__ == "__main__":
  sc = SparkContext()
  n = 32768
  d = 10
  m = 200
  matrix_list, matrix = gen_synth_data(n, d)
  #matrix = gen_data_2(n, d)
  rdd = sc.parallelize(matrix_list)
  #print rdd.take(2)
  #sys.exit()
  spark_matrix = Matrix(rdd)

  num_experiments = 50
  c = 1
  spark_error = []
  jiyans_error = []
  x_s = [i for i in range(num_experiments)]
  f = open('out.txt', 'w')
  for i in range(num_experiments):
    sfjlt = SpaFJLT(spark_matrix.matrix, m, c)
    sfjlt_rdd = sc.parallelize(sfjlt)
    print sfjlt_rdd
    tsqr = TSQR(Matrix(sfjlt_rdd), 3)
    R = tsqr.tsqr(sc)
    R = np.linalg.inv(R)
    lev = spark_matrix.get_rdd().map(lambda row: np.linalg.norm(np.dot(row, R))**2)
    sum_vel = d*np.array(lev.collect())/lev.reduce(add)
    app_lev = np.array(sum_vel)
    U, D, V = np.linalg.svd(matrix, full_matrices=False)
    lev_exact = np.sum(U**2,axis = 1)

    error = np.linalg.norm( lev_exact - app_lev) / np.linalg.norm(lev_exact)
    spark_error.append(error)
    je = jiyans(np.array(matrix),sc,n,d,m)
    f.write(str(error)+" " +str(je)+"\n")
    jiyans_error.append(je)
  """
  fig, ax = plt.subplots()
  ax.plot(x_s,spark_error,'b--',label="alpha matrix library")
  ax.plot(x_s, spark_error,'go',label="alpha matrix library")
  ax.plot(x_s,jiyans_error,'r--',label ="jiyan's code")
  ax.plot(x_s,jiyans_error,'mo',label ="jiyan's code")
  ax.set_xlabel("number of experiments")
  ax.set_ylabel("approx error")
  legend = ax.legend(loc='upper right', shadow=True)
  """