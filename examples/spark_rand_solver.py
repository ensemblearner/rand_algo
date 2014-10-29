from pyspark import SparkContext
from regression.least_squares import *
from scipy import stats
from sklearn import linear_model

def gen_random_data(num_samples, num_features):
    np.random.seed(10)
    X = np.random.randn(num_samples, num_features)
    lambda_ = 4.
    w = np.zeros(num_features)
    relevant_features = np.random.randint(0, num_features, 10)
    for i in relevant_features:
        w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
    alpha_ = 50.
    noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=num_samples)
    y = np.dot(X, w) + noise
    return X.tolist(), X, y

def original_least_squares(X,y):
    clf = linear_model.LinearRegression()
    clf.fit(X, y)
    return clf.coef_

if __name__ == "__main__":
    sc = SparkContext()
    n = 500000
    d = 10
    c = 1
    num_projections = 200
    matrix_list, matrix, y = gen_random_data(n, d)
    rdd = sc.parallelize(matrix_list)
    y_rdd = sc.parallelize(y)
    spark_matrix = Matrix(rdd)
    ls = RandLeastSquares(1000, sc)
    ls.fit(spark_matrix, y_rdd, num_projections, c)
    print ls.clf.coef_
    print original_least_squares(matrix_list, y)


    #U, D, V = np.linalg.svd(matrix, full_matrices=False)
    #lev_exact = np.sum(U**2, axis=1)
    #error = np.linalg.norm(lev_exact - app_lev) / np.linalg.norm(lev_exact)
    #print "error ", error

