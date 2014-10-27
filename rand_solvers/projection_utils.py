import numpy as np

def mapSFJLT(line, m, c):
    vec = np.array(line)
    rt = np.random.randint(m, size=c)
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
