import numpy as np
from numpy import array,identity,diagonal
from math import sqrt

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

import numpy as np
from numpy import array,identity,diagonal
from math import sqrt

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

def eigen(A,tol = 1.0e-9): 
    def maxElem(A): 
        n = len(A)
        Amax = 0.0
        for i in range(n-1):
            for j in range(i+1,n):
                if abs(A[i,j]) >= Amax:
                    Amax = abs(A[i,j])
                    k = i; l = j
        return Amax,k,l
 
    def rotate(A,p,k,l): 
        n = len(A)
        Adiff = A[l,l] - A[k,k]
        if abs(A[k,l]) < abs(Adiff)*1.0e-36: 
            t = A[k,l]/Adiff
        else:
            phi = Adiff/(2.0*A[k,l])
            t = 1.0/(abs(phi) + sqrt(phi**2 + 1.0))
            if phi < 0.0: 
                t = -t
        c = 1.0/sqrt(t**2 + 1.0); s = t*c
        tau = s/(1.0 + c)
        temp = A[k,l]
        A[k,l] = 0.0
        A[k,k] = A[k,k] - t*temp
        A[l,l] = A[l,l] + t*temp
        for i in range(k): 
            temp = A[i,k]
            A[i,k] = temp - s*(A[i,l] + tau*temp)
            A[i,l] = A[i,l] + s*(temp - tau*A[i,l])
        for i in range(k+1,l): 
            temp = A[k,i]
            A[k,i] = temp - s*(A[i,l] + tau*A[k,i])
            A[i,l] = A[i,l] + s*(temp - tau*A[i,l])
        for i in range(l+1,n):
            temp = A[k,i]
            A[k,i] = temp - s*(A[l,i] + tau*temp)
            A[l,i] = A[l,i] + s*(temp - tau*A[l,i])
        for i in range(n):
            temp = p[i,k]
            p[i,k] = temp - s*(p[i,l] + tau*p[i,k])
            p[i,l] = p[i,l] + s*(temp - tau*p[i,l])
    
    n = len(A)
    maxRot = 5*(n**2)
    p = identity(n)*1.0
    for i in range(maxRot): 
        Amax,k,l = maxElem(A)
        if Amax < tol: 
            return diagonal(A),p
        rotate(A,p,k,l)
    print('No eigen value')
        
A = eval(input('Enter the matrix A:'))
print('Eigenvalues and Eigenvectors of matrix:\n', A)
print('is\n', eigen(A,tol = 1.0e-9))
