import numpy as np

def add(m1,m2):
    return m1+m2

def substract(m1,m2):
    return m1-m2

def dot(m1,m2):
    return np.dot(m1,m2)
    
def transpose(m):
    return np.transpose(m)
    
def det(m):
    return np.linalg.det(m)

def inv(m):
    return np.linalg.inv(m)

def norm(m):
    return np.linalg.norm(m)