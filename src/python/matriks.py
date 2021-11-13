import numpy
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from math import sqrt
from math import copysign

# Operasi Yang Diperlukan


def DiagonalMatriks(A):
    return np.diagonal(A)


def transpose(A):
    return np.transpose(A)


def array(A):
    return np.array(A)


def Dot(A, B):
    return np.dot(A, B)


def get_u(A, v, sigma):
    u = []
    for i in range(len(sigma)):
        ui = (1/sigma[i])*Dot(A, v.T[i])
        u.append(ui)

    return transpose(array(u))


def Outer(A, B):
    return np.outer(A, B)


def Identity(A):
    return np.identity(A)


def Norm(A):
    return np.linalg.norm(A)


def CopyMatriks(A):
    return np.copy(A)


def UkuranMatriks(A):
    return np.shape(A)


def Zero_Like(A):
    return np.zeros_like(A)

# Code Untuk Mencari SVD


def Refleksi_HouseHolder(A):
    (Baris, Kolom) = UkuranMatriks(A)
    Q = Identity(Baris)
    R = CopyMatriks(A)

    for i in range(Baris - 1):
        x = R[i:, i]

        e = Zero_Like(x)
        e[0] = copysign(Norm(A), -A[i, i])
        u = x + e

        v = u / Norm(u)

        Q_i = Identity(Baris)
        Q_i[i:, i:] -= 2.0 * Outer(v, v)

        R = Dot(Q_i, R)
        Q = Dot(Q, Q_i.T)

    return (Q, R)


def qr_eigen(M, maxIter):
    """
    Menghitung nilai Sigma dan V

    """
    # Initialize empty array to store eigenvalues
    A = []

    V = np.eye(M.shape[0])

    # Append input matrix M to A
    A.append(None)
    A.append(M)

    # lakukan iterasi sebanyak maxIter untuk menghitung eigenvalue and eigenvector
    # Gunakan QR houseHolder Reflection
    for k in range(maxIter):
        A[0] = A[1]
        q, R = Refleksi_HouseHolder(A[0])
        A[1] = Dot(R, q)
        V = Dot(V, q)
    SingularValues = []
    for i in DiagonalMatriks(A[1]):
        if i > 0:
            SingularValues.append(sqrt(i))
    Sigma = array(SingularValues)
    return Sigma, V


def SVD(A, K):
    Sigma, V = qr_eigen(Dot(A.T, A), K)
    U = get_u(A, V, Sigma)
    V_singular = np.negative(V)
    return np.negative(U), Sigma, V_singular.T


NAME = input('MASUKKAN NAMA FILE YANG INGIN DICOMPRESS : ')
images = {
    "GWF KOMPRESS": np.asarray(Image.open(NAME))
}


def show_images(img_name):
    """
    Untuk Menampilkan Gambar

    """
    print("Loading...")
    plt.title("Bila plot ini ditutup maka gambar akan segera dicompress...")
    plt.imshow(images[img_name])
    plt.axis('off')
    plt.show()


show_images("GWF KOMPRESS")
compressed_image = None


def Kompress_image(Nama_image, k):
    print("Please Wait")
    print("---------------------------------------------------")
    print("PROCESSING...")
    global compressed_image
    img = images[Nama_image]
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    print("Please Wait")
    print("------------------------------------------------------")
    print("COMPRESSING...")
    ur, sr, vr = SVD(r, k)
    ug, sg, vg = SVD(g, k)
    ub, sb, vb = SVD(b, k)
    rr = Dot(ur[:, :k], Dot(np.diag(sr[:k]), vr[:k, :]))
    rg = Dot(ug[:, :k], Dot(np.diag(sg[:k]), vg[:k, :]))
    rb = Dot(ub[:, :k], Dot(np.diag(sb[:k]), vb[:k, :]))
    print("Please Wait")
    print("-----------------------------------------------")
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
