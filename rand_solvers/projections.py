__author__ = 'msingh'
import numpy as np
from utils.matrix import *
from utils.basic import *


def SpaFJLT(A, m, c):
    #A is an RDD
    PiA = A.flatMap(lambda line: mapSFJLT(line, m, c)).reduceByKey(add).collect()
    #print PiA
    return np.array([item[1] for item in PiA])


def mapSFJLT(line, m, c):
    vec = np.array(line)

    rt = np.random.randint(m, size=c)

    #rt = [1]
    tmp = []
    for i in range(c):
        coin = np.random.rand()
        if coin < 0.5:
            a = -1
        else:
            a = 1

        tmp.append((rt[i], a*vec/np.sqrt(c)))
    #print " temp is ", tmp
    return tmp
