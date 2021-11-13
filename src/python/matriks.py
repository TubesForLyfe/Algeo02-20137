import numpy
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from numpy.linalg import norm
from random import normalvariate
from math import sqrt
from scipy.linalg import null_space
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


def minimum(A, B):
    if A >= B:
        return B
    else:
        return A


def multiply_matriks(A, B):
    return np.matmul(A, B)


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


def nullspace(A):
    D = []
    NullSpace = null_space(A)
    for i in range(len(NullSpace)):
        D.extend(NullSpace[i])
    return D


def VektorAwal(ukuran):
    '''
    Initialize Vektor awal yang diambil secara random

    '''
    TidakNormalisasi = [0 for i in range(ukuran)]
    VektorAwal = [0 for i in range(ukuran)]
    sum = 0
    for i in range(ukuran):
        TidakNormalisasi[i] = normalvariate(0, 1)
        sum += TidakNormalisasi[i] ** 2
    Normalisasi = sqrt(sum)
    for i in range(ukuran):
        VektorAwal[i] = TidakNormalisasi[i]/Normalisasi
    return VektorAwal


def SVD_Awal(A, epsilon=1e-10):
    '''
    Menghitung iteration awal dari SVD dengan menggunakan Power Method

    '''
    A = ArrayOfFloat(A)
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
        print("-------------------------------------------------------------")
        print("LOADING")
        print("-------------------------------------------------------------")
        print("Mohon Tunggu")
        V_Akhir = V_Awal
        V_Awal = dot(B, V_Akhir)
        V_Awal = V_Awal / norm(V_Awal)
        count += 1
    return V_Awal


def SVD(A, epsilon=1e-10):
    '''
    Menghitung Next iteration dari SVD dengan menggunakan Power Method

    '''
    N, M = ukuranMatriks(A)
    Tampungan_SVD = []
    Min_Ukuran = minimum(N, M)
    increment = 0
    while (increment < Min_Ukuran):
        MatrixClone = CopyMatriks(A)
        for singularValue, U, V in Tampungan_SVD[0:increment]:
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

    C = []
    C.extend(nullspace(multiply_matriks(transpose(A), A)))
    V_SingularFix = [[] for i in range(len(transpose(V_Singular)))]
    for i in range(len(transpose(V_Singular))):
        for j in range(len(transpose(V_Singular)[i])):
            V_SingularFix[i].append(transpose(V_Singular)[i][j])
    for i in range(len(V_SingularFix)):
        V_SingularFix[i].append(C[i])
    V_SingularFix1 = np.array(V_SingularFix)

    return U_Singular, singularValues, V_SingularFix1


NAME = input('MASUKKAN NAMA FILE YANG INGIN DICOMPRESS : ')
images = {
    "GWF KOMPRESS": np.asarray(Image.open(NAME))
}


def show_images(img_name):
    'It will show image in widgets'
    print("Loading...")
    plt.title("Close this plot to open compressed image...")
    plt.imshow(images[img_name])
    plt.axis('off')
    plt.show()


show_images('GWF KOMPRESS')
compressed_image = None


def Kompress_image(Nama_image, k):
    print("processing...")
    global compressed_image
    img = images[Nama_image]
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    print("compressing...")
    ur, sr, vr = SVD(r)
    ug, sg, vg = SVD(g)
    ub, sb, vb = SVD(b)
    rr = np.dot(ur[:, :k], np.dot(np.diag(sr[:k]), vr[:k, :]))
    rg = np.dot(ug[:, :k], np.dot(np.diag(sg[:k]), vg[:k, :]))
    rb = np.dot(ub[:, :k], np.dot(np.diag(sb[:k]), vb[:k, :]))

    print("arranging...")
    rimg = np.zeros(img.shape)
    rimg[:, :, 0] = rr
    rimg[:, :, 1] = rg
    rimg[:, :, 2] = rb

    for ind1, row in enumerate(rimg):
        for ind2, col in enumerate(row):
            for ind3, value in enumerate(col):
                if value < 0:
                    rimg[ind1, ind2, ind3] = abs(value)
                if value > 255:
                    rimg[ind1, ind2, ind3] = 255

    compressed_image = rimg.astype(np.uint8)
    plt.title(Nama_image+"\n")
    plt.imshow(compressed_image)
    plt.axis('off')
    plt.show()
    compressed_image = Image.fromarray(compressed_image)


imOrig = Image.open(NAME)
im = numpy.array(imOrig)
A, B, C = im.shape
print("Batas terbesar nilai K pada Kompresi : ", A)
print("Catatan : semakin besar nilai k yang dimasukkan semakin tidak terkompresi gambarnya")
print("-------------------------------------------------------------------------------------")
K = int(input("MASUKKAN NILAI K : "))
NAME_SAVE = input("MASUKKAN NAMA FILE YANG INGIN DISAVE : ")
print('DONE - Compressed the image! Over and out!')
Kompress_image("GWF KOMPRESS", K)
