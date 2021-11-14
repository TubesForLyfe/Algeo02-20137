from flask import Flask, render_template, url_for, request, redirect
import os
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from math import sqrt
from math import copysign
    
app = Flask(__name__)

app.config["ALLOW"] =["PNG", "JPG", "JPEG", "GIF"]

def allowed(filename):
   if not "." in filename:
      return False

   format = filename.rsplit(".", 1)[1]
   if format.upper() in app.config["ALLOW"]:
      return True
   else:
      return False

@app.route('/', methods=["GET","POST"])
def main():
   # Operasi Yang Diperlukan

   def DiagonalMatriks(A):
      return np.diagonal(A)


   def transpose(A):
      return np.transpose(A)


   def array(A):
      return np.array(A)


   def Dot(A, B):
      return np.dot(A, B)


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

   # Code Untuk Mencari SVD dan Kompresi Gambar

   def Refleksi_Householder(A):
      """Perform QR decomposition of matrix A using Householder reflection."""
      (Baris, Kolom) = UkuranMatriks(A)

      Q = Identity(Baris)
      R = CopyMatriks(A)

      for i in range(Baris - 1):
         x = R[i:, i]
         e = Zero_Like(x)
         e[0] = copysign(Norm(x), -A[i, i])
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

      Q = np.eye(M.shape[0])

      # Append input matrix M to A
      A.append(None)
      A.append(M)

      # lakukan iterasi sebanyak maxIter untuk menghitung eigenvalue and eigenvector
      # Gunakan QR houseHolder Reflection
      for k in range(maxIter):
         A[0] = A[1]
         q, R = Refleksi_Householder(A[0])
         A[1] = Dot(R, q)
         Q = Dot(Q, q)
      SingularValues = []
      for i in DiagonalMatriks(A[1]):
         if i > 0:
            SingularValues.append(sqrt(i))
      Sigma = array(SingularValues)
      return Sigma, Q


   def SVD(A, K):
      Sigma, U = qr_eigen(Dot(A, transpose(A)), K)
      Sigma, V = qr_eigen(Dot(transpose(A), A), K)
      return np.negative(U), Sigma, V.T


   def show_images(img_name):
      """
      Untuk Menampilkan Gambar

      """
      print("Loading...")
      plt.title("Bila plot ini ditutup maka gambar akan segera dicompress...")
      plt.imshow(images[img_name])
      plt.axis('off')
      plt.show()


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

   # Code Utama

   if request.method == "POST":
      if request.files and request.form:
         image = request.files["image"]
         size = float(request.form["number"])
         if not(allowed(image.filename)):
            print("Image extension is not allowed")
         elif (size < 0 or size > 100):
            print("Compress size is not allowed")
         else:
            filename = secure_filename(image.filename)
            image.save(os.path.join("",filename))
            images = { "GWF KOMPRESS": np.asarray(Image.open(filename)) }
            show_images("GWF KOMPRESS")
            imOrig = Image.open(filename)
            im = np.array(imOrig)
            A, B, C = im.shape
            K = int(A * size / 100)
            Kompress_image("GWF KOMPRESS", K)
            print('DONE - Compressed the image! Over and out!')
         return redirect(request.url)
   return render_template('gwf.html')

if __name__ == '__main__':
   app.run(debug=True)

