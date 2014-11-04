from rand_solvers.projections import *
from rand_solvers.rma_utils import *
from utils.matrix import *
import logging

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)
class RandLeastSquares:
    def __init__(self, matrix_A, matrix_b, **kwargs):
        self.k = kwargs.get('k')
        self.s = kwargs.get('s', None)
        self.technique = kwargs.get('technique')
        self.matrix_A = matrix_A
        self.matrix_b = matrix_b
        self.matrix_Ab = Matrix(self.matrix_A.matrix.zip(self.matrix_b.matrix).map(lambda x: x[0]+[x[1]]))
        self.projection = Projections(**kwargs)
        self.cost = []

    def fit(self, A, b, debug=True):

        if self.technique == 'projection':
            x = self.projection.execute(self.matrix_Ab, 'solve')
        elif self.technique == 'sampling':
            if not self.s:
                raise ValueError("s param cannot be none but is %s" % self.s)

            N = self.projection.execute(self.matrix_Ab, 'svd')
            sumLev = comp_lev(self.matrix_Ab.matrix, N)
            x = sample_solve(self.matrix_Ab.matrix, N, sumLev, self.s)
        else:
            raise ValueError("invalid technique")

        for i in xrange(self.k):
            cost = npl.norm(np.dot(A, x[i])-b)
            if debug:
                x_opt, f_opt = self.__ideal_cost(A, b)
                logger.info(npl.norm(np.array(x[i])-x_opt)/npl.norm(x_opt))
                print npl.norm(np.array(x[i])-x_opt)/npl.norm(x_opt)
                logger.info(np.abs(f_opt-cost)/f_opt)
                print np.abs(f_opt-cost)/f_opt

            self.cost.append(cost)

        return x, cost

    def __ideal_cost(self, A, b):
        x_opt = npl.lstsq(A, b)[0]
        f_opt = npl.norm(np.dot(A, x_opt)-b)
        return x_opt, f_opt


