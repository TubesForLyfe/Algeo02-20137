import java.util.Scanner;

public class MatriksBalikan extends Matrix {
    // ---------------------INVERS MATRIKS ----------------------------------

    public static Matrix transpose(Matrix M) {
        // Menghasilkan transpose matriks dari M
        Matrix Hsl = new Matrix(M.rowNum, M.colNum);
        int i, j;
        for (i = 0; i < M.rowNum; i++) {
            for (j = 0; j < M.colNum; j++) {
                Hsl.data[i][j] = M.data[j][i];
            }
        }
        return Hsl;
    }

    public static Matrix copy(Matrix M) {
        // Menghasilkan Matriks yang merupakan salinan dari Matriks M
        Matrix Hsl = new Matrix(M.rowNum, M.colNum);
        int i, j;
        for (i = 0; i < M.rowNum; i++) {
            for (j = 0; j < M.colNum; j++) {
                Hsl.data[i][j] = M.data[i][j];
            }
        }
        return Hsl;
    }

    public static boolean IsIdentitas(Matrix M) {
        // Menghasilkan True jika Matriks M merupakan Matriks Identitas
        boolean cek = true;
        int i, j;
        for (i = 0; i < M.rowNum; i++) {
            if (M.data[i][i] != 1) {
                return false;
            }
            for (j = 0; j < M.colNum; j++) {
                if (i != j) {
                    if (M.data[i][j] != 0) {
                        return false;
                    }
                }
            }
        }
        return cek;
    }

    public static void multiplycons(Matrix M, Double k) {
        // Mengalikan setiap elemen matriks dengan konstanta k
        int i, j;
        for (i = 0; i < M.rowNum; i++) {
            for (j = 0; j < M.colNum; j++) {
                M.data[i][j] = k * M.data[i][j];
            }
        }
    }

    public static Matrix MergeMatrix(Matrix M, Matrix N) {

        Matrix Hsl = new Matrix(M.rowNum, M.colNum + N.colNum);
        for (int i = 0; i < M.rowNum; i++) {
            for (int j = 0; j < Hsl.colNum; j++) {
                if (j < M.colNum) {
                    Hsl.data[i][j] = M.data[i][j];
                } else {
                    Hsl.data[i][j] = N.data[i][j - M.colNum];
                }
            }
        }
        return Hsl;
    }

    public static Matrix MergeIdentitas(Matrix M) {
        // Menghasilkan matriks dengan besar kolom 2 kali lipat yang merupakan gabungan
        // matriks M dan matriks Identitas
        Matrix Hsl = new Matrix(M.rowNum, M.colNum + M.colNum);
        for (int i = 0; i < M.rowNum; i++) {
            for (int j = 0; j < Hsl.colNum; j++) {
                if (j < M.colNum) {
                    Hsl.data[i][j] = M.data[i][j];
                } else {
                    if (i + M.colNum == j) {
                        Hsl.data[i][j] = 1.0;
                    } else {
                        Hsl.data[i][j] = 0.0;
                    }
                }

            }
        }
        return Hsl;
    }

    public static Matrix SplitColIdentitas(Matrix M) {
        // membagi matriks menjadi 2 berdasarkan kolom
        // Menghasilkan Matriks dengan Memisahkan setengah ukuran kolom sebelah kiri 
        // yang merupakan matriks Identitas sehingga didapatkan matriks hasil OBE
        int colNew = M.colNum / 2;
        Matrix Hsl = new Matrix(M.rowNum, colNew);
        for (int i = 0; i < M.rowNum; i++) {
            for (int j = 0; j < Hsl.colNum; j++) {
                Hsl.data[i][j] = M.data[i][j + colNew];
            }
        }
        return Hsl;
    }

    public static boolean isHaveInversKof(Matrix M) {
        // Menghasilkan true jika determinan Matriks tidaklah nol
        Double det = Determinan.detEksKof(M);
        return (det != 0.0);
    }

    public static Matrix inversAdjoin(Matrix M) {
        // Menghasilkan Matrix Balikan dengan menggunakan metode Adjoin
        Matrix Hsl;
        int i, j;
        Hsl = copy(M);
        Double det = Determinan.detEksKof(M);
        Double k = 1 / det;
        Hsl = (M.cofactor()).transpose();
        multiplycons(Hsl, k);
        for (i = 0; i < Hsl.rowNum; i++) {
            for (j = 0; j < Hsl.colNum; j++) {
                if (Hsl.data[i][j] == 0.00) {
                    Hsl.data[i][j] = Math.abs(Hsl.data[i][j]);
                }
            }
        }
        return Hsl;
    }

    // ----------INVERS GAUSS JORDAN
    // -------------------------------------------------------
    public static boolean isHaveInversGaussJordan(Matrix M) {
        // Menghasilkan true jika setelah dilakukan gauss jordan jika setengah 
        // ukuran kolom sebelah kiri matriks M merupakan matriks Identitas
        boolean cek = false;
        Matrix cekMatrix;
        cekMatrix = copy(M);
        GaussJordan(cekMatrix);
        if (IsIdentitas(cekMatrix)) {
            cek = true;
        }
        return cek;
    }

    public static int indeks0(int i, Matrix M) {
        // Menghasilkan indeks dengan nilai 0 paling ujung di sebuah baris i Matriks M
        int j, idx;
        idx = -1;
        for (j = 0; j < M.colNum; j++) {
            if (M.data[i][j] == 0) {
                idx++;
            } else {
                break;
            }
        }
        return idx;
    }

    public static void swap(int i, int j, Matrix M) {
        Matrix copy;
        int k;
        copy = copy(M);
        for (k = 0; k < M.colNum; k++) {
            M.data[i][k] = copy.data[j][k];
            M.data[j][k] = copy.data[i][k];
        }

    }

    public static void cek0(Matrix M) {
        int i, j;
        int idxmin;
        idxmin = indeks0(0, M);
        for (i = 0; i < M.rowNum; i++) {
            idxmin = indeks0(i, M);
            for (j = i + 1; j < M.rowNum; j++) {
                if (indeks0(j, M) < idxmin) {
                    swap(i, j, M);
                    idxmin = indeks0(j, M);
                }
            }
        }
    }

    public static void satuan(int i, Matrix M) {
        // Membuat elemen tidak 0 pada baris i di Matrix M menjadi satu utama
        int j;
        int k = indeks0(i, M) + 1;
        Double kons;
        if (k < M.colNum) {
            kons = M.data[i][k];
            for (j = k; j < M.colNum; j++) {
                M.data[i][j] = M.data[i][j] / kons;
            }
        }
    }

    public static void reduksi(int i, int i1, Matrix M) {
        // baris i baris i1
        int j, k;
        Double kons;
        k = indeks0(i, M) + 1;
        if (k < M.colNum) {
            kons = M.data[i1][k];
            for (j = k; j < M.colNum; j++) {
                M.data[i1][j] = M.data[i1][j] - (M.data[i][j] * kons);
            }
        }
    }

    public static void Gauss(Matrix M) {
        // Melakukan Metode Gauss pada Matrix M 
        int i, j;
        for (i = 0; i < M.rowNum; i++) {
            satuan(i, M);
            for (j = i + 1; j < M.rowNum; j++) {
                reduksi(i, j, M);
            }
        }

        satuan(M.rowNum - 1, M);
    }

    public static int indeks1(int i, Matrix M) {
        int j, idx;
        idx = 0;
        for (j = 0; j < M.colNum; j++) {
            if (M.data[i][j] == 1) {
                idx = j;
                break;
            }
        }
        return idx;
    }

    public static void reduksiJordan(int i, int i1, Matrix M) {
        // baris i baris i1
        int j, k;
        Double kons;
        k = indeks1(i, M);
        kons = M.data[i1][k];
        for (j = indeks1(i, M); j < M.colNum; j++) {
            M.data[i1][j] = M.data[i1][j] - (M.data[i][j] * kons);
        }
    }

    public static void GaussJordan(Matrix M) {
        // dilakukan Metode GaussJordan agar dihasilkan nilai 0 diatas satu utama
        int i, j;
        Gauss(M);
        for (i = 0; i < M.rowNum; i++) {
            for (j = i + 1; j < M.rowNum; j++) {
                reduksiJordan(j, i, M);
            }
        }
    }

    public static Matrix InversGaussJordan(Matrix M) {
        // dilakukan Metode GaussJordan agar dihasilkan nilai 0 diatas satu utama 
        Matrix Hsl, Invers;
        Hsl = MergeIdentitas(M);
        GaussJordan(Hsl);
        Invers = SplitColIdentitas(Hsl);
        return Invers;
    }
    // --------------------------------------------------------------------------------------------

    public static hasilOutput Run(int submenu, int inputType) {
        // hasil Output yang akan ditampilkan di menu saat dipanggil
        hasilOutput res = new hasilOutput();
        Scanner sc = new Scanner(System.in);
        Matrix M = new Matrix();
        switch (inputType) {
            case 1:
                System.out.printf("Dimensi Matrix : ");
                int dim = sc.nextInt();
                M = new Matrix(dim, dim);
                System.out.println("Matrix :");
                M.readMatrix();
                break;

            case 2:
                M.parseFile();
                break;
        }
        switch (submenu) {
            case 1:
                res = P_InverseGaussJordan(M);
                break;
            case 2:
                res = P_InverseAdjoin(M);
                break;

        }
        return res;
    }

    public static hasilOutput P_InverseAdjoin(Matrix M) {
        // hasil Output yang akan ditampilkan di menu saat dipanggil
        hasilOutput res = new hasilOutput();
        // Konstruksi matrix kosong
        Matrix Hsl;
        // cek apakah Matriks mempunyai invers
        if (isHaveInversKof(M)) {
            Hsl = inversAdjoin(M);
            res.addMatrix(Hsl);
        } else {
            res.add("Matriks M tidak punya balikan");
        }
        return res;

    }

    public static hasilOutput P_InverseGaussJordan(Matrix M) {
        // hasil Output yang akan ditampilkan di menu saat dipanggil
        hasilOutput res = new hasilOutput();
        // Konstruksi matrix kosong
        Matrix Hsl;
        // Cek apakah Matriks Mempunya invers
        if (isHaveInversGaussJordan(M)) {
            Hsl = InversGaussJordan(M);
            res.addMatrix(Hsl);
        } else {
            res.add("Matriks M tidak punya balikan");
        }
        return res;

    }

}
