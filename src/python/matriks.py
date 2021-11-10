import numpy as np
from numpy import array, identity, diagonal
from numpy.lib.shape_base import split
from numpy.linalg import norm
from random import normalvariate
from math import sqrt

from numpy.ma.core import count

double_epsilon = 1e-140


def add(m1, m2):
    return m1+m2


def substract(m1, m2):
    return m1-m2


def dot(m1, m2):
    return np.dot(m1, m2)


def transpose(m):
    return np.transpose(m)


def det(m):
    return np.linalg.det(m)


def inv(m):
    return np.linalg.inv(m)


def normalize(m):
    return m/np.linalg.norm(m)


def swap_row(m, r1, r2):
    m[[r1, r2]] = m[[r2, r1]]
    return m


def multiply(m, k, r1):
    l = len(m[0])
    for i in range(l):
        m[r1][i] *= k
    return m


def add_multiply(m, k, r, rm):
    l = len(m[0])
    for i in range(l):
        m[r][i] += k*m[rm][i]
    return m


def eq(a, b):
    return abs(a-b) < double_epsilon


def firstNonZeroOccurence(m, row, start_idx):
    i = start_idx
    found = False
    while(i < len(m[row]) and not found):
        found = not eq(m[row][i], 0)
        i += 1

    return (i-1) if found else None


def gauss(m):
    i = 0
    k = 0
    while(i < len(m) and k < len(m[0])):
        i_max = firstNonZeroOccurence(m, k, i)
        if(eq(m[i_max][k], 0.0)):
            k += 1
        else:
            if(i_max != i):
                m = swap_row(m, i_max, i)
            tmp = m[i][k]
            for j in range(len(m[0])):
                m[i][j] /= tmp
            for j in range(i+1, len(m)):
                for p in range(k+1, len(m[0])):
                    m[j][p] = m[j][p] - m[i][p]*m[j][k]
                m[j][k] = 0.0
            i += 1
            k += 1

    return m


def jordan(m):
    i = 0
    k = 0
    while(i < len(m) and k < len(m[0])):
        i_max = firstNonZeroOccurence(m, k, i)
        if(eq(m[i_max][k], 0.0)):
            k += 1
        else:
            if(i_max != i):
                swap_row(m, i_max, i)
            tmp = m[i][k]
            for j in range(len(m[0])):
                m[i][j] /= tmp
            for j in range(len(m)):
                if(j == i):
                    continue
                for p in range(k+1, len(m[0])):
                    m[j][p] = m[j][p] - m[i][p]*m[j][k]
                m[j][k] = 0.0
            i += 1
            k += 1
    return m


double_epsilon = 1e-140


def add(m1, m2):
    return m1+m2


def substract(m1, m2):
    return m1-m2


def dot(m1, m2):
    return np.dot(m1, m2)


def transpose(m):
    return np.transpose(m)


def det(m):
    return np.linalg.det(m)


def inv(m):
    return np.linalg.inv(m)


def normalize(m):
    return m/np.linalg.norm(m)


def swap_row(m, r1, r2):
    m[[r1, r2]] = m[[r2, r1]]
    return m


def multiply(m, k, r1):
    l = len(m[0])
    for i in range(l):
        m[r1][i] *= k
    return m


def add_multiply(m, k, r, rm):
    l = len(m[0])
    for i in range(l):
        m[r][i] += k*m[rm][i]
    return m


def eq(a, b):
    return abs(a-b) < double_epsilon


def firstNonZeroOccurence(m, row, start_idx):
    i = start_idx
    found = False
    while(i < len(m[row]) and not found):
        found = not eq(m[row][i], 0)
        i += 1

    return (i-1) if found else None


def gauss(m):
    i = 0
    k = 0
    while(i < len(m) and k < len(m[0])):
        i_max = firstNonZeroOccurence(m, k, i)
        if(eq(m[i_max][k], 0.0)):
            k += 1
        else:
            if(i_max != i):
                m = swap_row(m, i_max, i)
            tmp = m[i][k]
            for j in range(len(m[0])):
                m[i][j] /= tmp
            for j in range(i+1, len(m)):
                for p in range(k+1, len(m[0])):
                    m[j][p] = m[j][p] - m[i][p]*m[j][k]
                m[j][k] = 0.0
            i += 1
            k += 1

    return m


def jordan(m):
    i = 0
    k = 0
    while(i < len(m) and k < len(m[0])):
        i_max = firstNonZeroOccurence(m, k, i)
        if(eq(m[i_max][k], 0.0)):
            k += 1
        else:
            if(i_max != i):
                swap_row(m, i_max, i)
            tmp = m[i][k]
            for j in range(len(m[0])):
                m[i][j] /= tmp
            for j in range(len(m)):
                if(j == i):
                    continue
                for p in range(k+1, len(m[0])):
                    m[j][p] = m[j][p] - m[i][p]*m[j][k]
                m[j][k] = 0.0
            i += 1
            k += 1
    return m


def minimum(A, B):
    if A >= B:
        return B
    else:
        return A


def ukuranMatriks(A):
    n, m = A.shape
    return n, m


def singularKiri(A):
    return dot(A, transpose(A))


def singularKanan(A):
    return dot(transpose(A), A)


def ArrayOfFloat(A):
    return np.array(A, dtype=float)


def CopyMatriks(A):
    return A.copy()


def Keluaran(A, B):
    return np.outer(A, B)


def SplitMatriks(A):
    X, Y, Z = [array(i) for i in zip(*A)]
    return X, Y, Z


def VektorAwal(ukuran):  # ambil sembarang vektor bisa dari random, untuk sebagai awal dari iterasi atau bisa disebut X0
    TidakNormalisasi = [normalvariate(0, 1) for i in range(ukuran)]
    sum = 0
    for i in TidakNormalisasi:
        sum += i**2
    Normalisasi = sqrt(sum)
    VektorAwal = [i / Normalisasi for i in TidakNormalisasi]
    return VektorAwal


def SVD_Awal(A, epsilon=1e-11):
    '''
    Menghitung iteration awal dari SVD dengan menggunakan Power Method

    '''
    N, M = ukuranMatriks(A)
    x = VektorAwal(minimum(N, M))  # set x sebagai vektor awal yang sembarang
    count = 0
    V_Awal = x
    if N <= M:
        B = singularKiri(A)
    else:
        B = singularKanan(A)
    V_Akhir = V_Awal
    V_Awal = dot(B, V_Akhir)
    V_Awal = V_Awal / norm(V_Awal)
    count += 1
    while (abs(dot(V_Awal, V_Akhir)) + epsilon <= 1):
        V_Akhir = V_Awal
        V_Awal = dot(B, V_Akhir)
        V_Awal = V_Awal / norm(V_Awal)
        count += 1
    return V_Awal


def SVD(A, epsilon=1e-11):
    '''
    Menghitung Next iteration dari SVD dengan menggunakan Power Method

    '''
    A = ArrayOfFloat(A)
    N, M = ukuranMatriks(A)
    Tampungan_SVD = []
    Min_Ukuran = min(N, M)
    increment = 0
    while (increment < Min_Ukuran):
        MatrixClone = CopyMatriks(A)
        for singularValue, U, V in Tampungan_SVD[:increment]:
            MatrixClone -= singularValue * Keluaran(U, V)
        if N > M:
            V = SVD_Awal(MatrixClone, epsilon=epsilon)
            U_TidakNormalized = dot(A, V)
            sigma = norm(U_TidakNormalized)
            U = U_TidakNormalized / sigma
        else:  # jika M >= N
            U = SVD_Awal(MatrixClone, epsilon=epsilon)
            V_TidakNormalized = dot(transpose(A), U)
            sigma = norm(V_TidakNormalized)
            V = V_TidakNormalized / sigma
        Tampungan_SVD.append((sigma, U, V))
        increment += 1
    singularValues, U_Singular, V_Singular = SplitMatriks(Tampungan_SVD)
    return singularValues, transpose(U_Singular), transpose(V_Singular)


if __name__ == "__main__":
    CEK = np.array([
        [4, -30, 60, -35], [-30, 300, -675, 420],
        [60, -675, 1620, -1050], [918, 123, 124, 543]
    ], dtype='float64')
    print(VektorAwal(len(CEK)))

    A, B, C = SVD(CEK)
    print(A)
    print(B)
    print(C)
