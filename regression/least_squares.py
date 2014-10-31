from factorization.tsqr import *
from rand_solvers.projections import *
from utils.utils import *
from sklearn import linear_model
import numpy as np

class RandLeastSquares:
    def __init__(self, num_samples, sc):
        self.num_samples = num_samples
        self.sc = sc
        self.clf = linear_model.LinearRegression()

    def fit(self, X, y, num_projections=50, c=1):
        leverage_scores = self.__compute_leverage_scores(X, num_projections, c)
        X = X.matrix.collect()
        y = y.collect()
        sampled_X, sampled_y = self.__sample_and_dataset(X, y, leverage_scores)

        self.clf.fit(sampled_X, sampled_y)



        """
        zipped_X = X.matrix.zipWithIndex().map(lambda x: (x[1], x[0]))
        zipped_y = y.zipWithIndex().map(lambda x: (x[1], x[0]))
        zipped_prob = self.sc.parallelize(leverage_scores).zipWithIndex().map(lambda x: (x[1], x[0]))
        joined_by_key = zipped_X.join(zipped_y).join(zipped_prob)
        sorted_prob = joined_by_key.sortBy(lambda x: x[1][1])
        #print sorted_prob.take(1)
        sorted_prob = sorted_prob.map(lambda x: include(x)).filter(lambda x:x[2] == True).take(self.k)
        print sorted_prob
        #sorted_prob.map(lambda x:x[1][1]).map(lambda x: )
        #sort_by_probs = joined_by_key.sortBy(lambda x:x[2])
        #print joined_by_key.take(2)
        """
        return leverage_scores


    def __sample_and_scale_dataset(self, X, y, probs, scale=True):

        samples = sample(zip(X,y), self.num_samples, prob=probs)
        sampled_X = samples['samples'][0]
        sampled_Y = samples['samples'][1]
        selected_probs = samples['probs']
        if scale:
            sampled_X = rescale(sampled_X, selected_probs)

        return sampled_X, sampled_Y

    def __compute_leverage_scores(self, X, num_projections, c):
        projection_matrix = self.__project(X, num_projections, c)
        R = self.__factorize(projection_matrix)
        lev = X.get_rdd().map(lambda row: np.linalg.norm(np.dot(row, R))**2)
        num_cols = X.get_dimensions()[1]
        #sum_vel = lev.map( lambda vec: num_cols*np.array(vec))/lev.reduce(add)
        sum_vel = num_cols*np.array(lev.collect())/lev.reduce(add)
        app_lev = np.array(sum_vel)
        return list(app_lev)

    def __project(self, X, num_projections, c):
        projection = Projections(X, self.sc)
        projection_matrix = projection.SpaFJLT(num_projections, c)
        return projection_matrix

    def __factorize(self, projection_matrix):
        tsqr = TSQR(projection_matrix, 3, self.sc)
        R = tsqr.tsqr()
        R = np.linalg.inv(R)
        return R

