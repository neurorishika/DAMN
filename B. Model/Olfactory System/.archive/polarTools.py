import numpy as np

def polar2cartesian(r, arr):
    """r is Radius; arr is [ n-2 elements from [0,pi] , 1 element from [0,2*pi] ] """
    a = np.concatenate((np.array([2*np.pi]), arr))
    si = np.sin(a)
    si[0] = 1
    si = np.cumprod(si)
    co = np.cos(a)
    co = np.roll(co, -1)
    return si*co*r

def generateUniform(radius,dimension=2):
    phis = np.arccos(np.random.uniform(-1,1,size=dimension-2))
    theta = np.random.uniform(0,2*np.pi,size=1)
    arr = np.concatenate((phis,theta),axis=None)
    return polar2cartesian(radius, arr)