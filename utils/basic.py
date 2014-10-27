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