import numpy as np


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
            yield ((i,j),np.dot(np.random.randn(m),data))