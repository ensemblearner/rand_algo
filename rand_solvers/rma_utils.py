import numpy.linalg as npl

from utils.utils import *

def srdht_map(rows, c, k, n, seed_s):
    data = []
    row_idx = []
    for row in rows:
        data.append(row[0])
        row_idx.append(row[1])
    data = np.array(data)
    row_idx = np.array(row_idx)
    m = data.shape[0]

    for i in range(k):
        S = np.arange(n)
        np.random.seed(seed_s[i])
        np.random.shuffle(S)
        S = S[:c]
        r = (np.random.rand(m)>0.5)*2-1
        rand_data = np.dot(np.diag(r), data)
        for j in range(c):
            row_h = np.sqrt(2)*np.cos(2*np.pi*row_idx*S[j]/n-np.pi/4)
            yield ((i, j), np.dot(row_h, rand_data)/np.sqrt(n))

def cw_map(row, c, k):
    row = np.array(row)
    rt = np.random.randint(c, size=k).tolist()
    coin = (np.random.rand(k)<0.5).astype(int)*2-1
    pairs = [((i, rt[i]), coin[i]*row) for i in range(k)]
    return pairs

def gaussian_map(rows,c,k):
    data = np.array([row for row in rows]) # m-by-d
    m = data.shape[0]

    for i in range(k):
        for j in range(c):
            yield ((i, j), np.dot(np.random.randn(m), data)/np.sqrt(c))

    #pairs = [((i,j),np.dot(np.random.randn(m),data)) for j in range(c) for i in range(k)] 

def rademacher_map(rows,c,k):
    data = np.array([row for row in rows]) # m-by-d
    m = data.shape[0]

    for i in range(k):
        for j in range(c):
            yield ((i, j), np.dot((np.random.rand(m)>0.5).astype(int), data))

def comp_lev(rows, N):
    return rows.map(lambda row: xN(row, N)).reduce(add).tolist()

def xN(x, N):
    return np.array([npl.norm(np.dot(np.array(x), N1))**2 for N1 in N])

def sample_solve(rows, N, sumLev, s):
    SA = rows.flatMap(lambda row: sample(row, N, sumLev, s)).groupByKey().collect()
    x = []
    for sa in SA:
        SAb = [row for row in sa[1]]
        SAb = np.array(SAb)
        A = SAb[:, :-1]
        b = SAb[:, -1]
        x.append(np.linalg.lstsq(A, b)[0].tolist())

    return x

def sample(x, N, sumLev, s):
    x = np.array(x)
    k = len(N)
    for i in range(k):
        q = npl.norm(np.dot(x, N[i]))**2
        p = min(q*s/sumLev[i], 1.0)
        if np.random.rand() < p:
            yield (i, (x/p).tolist())

