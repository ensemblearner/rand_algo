import numpy.linalg as npl

from utils.utils import *

def cw_map(row, c, k):
    row = np.array(row)
    rt = np.random.randint(c, size=k).tolist()
    coin = (np.random.rand(k)<0.5).astype(int)*2-1
    pairs = [((i, rt[i]), coin[i]*row) for i in range(k)]
    return pairs


def gaussian_map(rows, c, k):
    data = np.array([row for row in rows])
    m = data.shape[0]

    for i in range(k):
        for j in range(c):
            yield ((i, j), np.dot(np.random.randn(m), data))


def comp_lev(rows, N):
    return rows.map(lambda row: xN(row, N)).reduce(add).tolist()


def xN(x, N):
    return np.array([npl.norm(np.dot(np.array(x), N1))**2 for N1 in N])


def sample_solve(rows, N, sumLev, s):
    SA_inter = rows.flatMap(lambda row: _sample(row, N, sumLev, s)).groupByKey()
    SA = map((lambda (x, y): (x, list(y))), sorted(SA_inter.collect()))
    x = []
    for sa in SA:
        SAb = np.array(sa[1])

        A = SAb[:, :-1]
        b = SAb[:, -1]
        x.append(np.linalg.lstsq(A, b)[0].tolist())

    return x


def _sample(x, N, sumLev, s):
    x = np.array(x)
    k = len(N)
    for i in range(k):
        q = npl.norm(np.dot(x, N[i]))**2
        p = min(q*s/sumLev[i], 1.0)
        if np.random.rand() < p:
            yield (i,(x/p).tolist())