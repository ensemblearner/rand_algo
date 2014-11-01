from rand_solvers.projections import *

class RandLeastSquares:
    def __init__(self, **kwargs):
        self.projection = Projections(**kwargs)


    def fit(self, X, y=None, num_projections=50, c=1):
        print self.projection.execute(X)
        #return leverage_scores


