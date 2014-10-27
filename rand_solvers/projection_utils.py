import numpy as np

def mapSFJLT(line, m, c):
    vec = np.array(line)
    rt = np.random.randint(m, size=c)
    tmp = []
    for i in range(c):
        coin = np.random.rand()
        a = -1 if coin < 0.5 else 1

        tmp.append((rt[i], a*vec/np.sqrt(c)))
    return tmp
