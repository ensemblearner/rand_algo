import numpy as np


def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])


def add(x, y):
    x += y
    return x


def sampling(row, R, sumLev, s):
    #row = row.getA1()
    lev = np.linalg.norm(np.dot(row[:-1], R))**2
    p = s*lev/sumLev
    coin = np.random.rand()
    if coin < p:
        return row/p


def unifSampling(row, n, s):
    #row = row.getA1()
    p = s/n
    coin = np.random.rand()
    if coin < p:
        return row/p


def flip(row):
    return [(e[1], e[0], e[2]) for e in row]


def extract(row):
    return [ele[2] for ele in row]

def other_iterator(itr):
    yield sum(np.outer(x, y) for x, y in itr)


def num_rows_each_partition(iterator):
    yield sum(1 for _ in iterator)


def indexing(splitIndex, iterator, count_each_partition):
    # count = 0
    offset = sum(count_each_partition[:splitIndex]) if splitIndex else 0
    indexed = []
    for i, e in enumerate(iterator):
        index = offset + i
        for j, ele in enumerate(e):
            indexed.append((index, j, ele))
    yield indexed
