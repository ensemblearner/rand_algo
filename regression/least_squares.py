from rand_solvers.projections import *

class RandLeastSquares:
    def __init__(self,matrix, **kwargs):
        self.k = kwargs.get('k')
        self.projection = Projections(matrix, **kwargs)
        self.cost = []

    def fit(self, X, y, debug=True):

        l2_proj = self.projection.execute()
        if debug:
            x_opt, f_opt = self.__ideal_cost(X,y)

        for i in range(self.k):
            cost =  npl.norm(np.dot(X, l2_proj[i])-y)
            if debug:
                print npl.norm(np.array(l2_proj[i])-x_opt)/npl.norm(x_opt)
                print np.abs(f_opt-cost)/f_opt

            self.cost.append(cost)




    def __projection(self):
        return self.projection.execute()

    def __ideal_cost(self,A,b):
        x_opt = npl.lstsq(A,b)[0]
        f_opt = npl.norm(np.dot(A,x_opt)-b)
        return x_opt, f_opt


