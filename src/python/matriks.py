import numpy as np

double_epsilon = 1e-140

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

def normalize(m):
    return m/np.linalg.norm(m)

def swap_row(m,r1,r2):
    m[[r1,r2]] = m[[r2,r1]]
    return m

def multiply(m,k,r1):
    l = len(m[0])
    for i in range(l):
        m[r1][i] *= k
    return m

def add_multiply(m,k,r,rm):
    l = len(m[0])
    for i in range(l):
        m[r][i] += k*m[rm][i]
    return m

def eq(a,b):
    return abs(a-b) < double_epsilon

def firstNonZeroOccurence(m,row,start_idx):
    i = start_idx
    found = False
    while(i<len(m[row]) and not found):
        found = not eq(m[row][i],0)
        i+=1
    
    return (i-1) if found else None

def gauss(m):
    i = 0
    k = 0
    while(i<len(m) and k<len(m[0])):
        i_max = firstNonZeroOccurence(m,k,i)
        if(eq(m[i_max][k], 0.0)):
            k+=1
        else:
            if(i_max!=i):
                m = swap_row(m,i_max, i)
            tmp = m[i][k]
            for j in range(len(m[0])):
                m[i][j] /= tmp
            for j in range(i+1, len(m)):
                for p in range(k+1, len(m[0])):
                    m[j][p] = m[j][p] - m[i][p]*m[j][k];
                m[j][k] = 0.0;
            i+=1
            k+=1
    
    return m

def jordan(m):
    i = 0
    k = 0
    while(i<len(m) and k<len(m[0])):
        i_max = firstNonZeroOccurence(m, k, i);
        if(eq(m[i_max][k], 0.0)):
            k+=1
        else:
            if(i_max!=i):
                swap_row(m, i_max, i)
            tmp = m[i][k]
            for j in range(len(m[0])):
                m[i][j] /= tmp
            for j in range(len(m)):
                if(j==i):
                    continue
                for p in range(k+1, len(m[0])):
                    m[j][p] = m[j][p] - m[i][p]*m[j][k] 
                m[j][k] = 0.0
            i+=1
            k+=1
    return m