from factorization.tsqr import *
from rand_solvers.projections import *

class RandLeastSquares:
    def __init__(self, sc):
        self.sc = sc

    def fit(self, X, y, num_projections=50, c=1):
        leverage_scores = self.__compute_leverage_scores(X, num_projections, c)

        return leverage_scores

    def __compute_leverage_scores(self, X, num_projections, c):
        projection_matrix = self.__project(X, num_projections, c)
        R = self.__factorize(projection_matrix)
        lev = X.get_rdd().map(lambda row: np.linalg.norm(np.dot(row, R))**2)
        num_cols = X.get_dimensions()[1]
        sum_vel = num_cols*np.array(lev.collect())/lev.reduce(add)
        app_lev = np.array(sum_vel)
        return app_lev

    def __project(self, X, num_projections, c):
        projection = Projections(X, self.sc)
        projection_matrix = projection.SpaFJLT(num_projections, c)
        return projection_matrix

    def __factorize(self, projection_matrix):
        tsqr = TSQR(projection_matrix, 3, self.sc)
        R = tsqr.tsqr()
        R = np.linalg.inv(R)
        return R
