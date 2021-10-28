import java.util.Scanner;

public class Determinan extends Matrix {
    public static Double epsilon = 0.000001;

    public static boolean isZero(Double a) {
        // Mengeluarkan apakah sebuah Double
        // nol dengan membandingkannya dengan
        // angka yang kecil
        return (Math.abs(a) < epsilon);
    }

    // Fungsi Remove Column
    public static Matrix removeCol(Matrix M, int idxCol) {
        // Membuat matrix baru dengan kolom idxCol
        // dari Matrix M dihilangkan
        int i, j;
        Matrix newMat = new Matrix(M.rowNum, M.colNum - 1);
        for (i = 0; i < newMat.rowNum; i++) {
            for (j = 0; j < newMat.colNum; j++) {
                if (j < idxCol) {
                    newMat.data[i][j] = M.data[i][j];
                } else {
                    newMat.data[i][j] = M.data[i][j + 1];
                }
            }
        }
        return newMat;
    }

    // Fungsi Remove Row
    public static Matrix removeRow(Matrix M, int idxRow) {
        // Membuat matrix baru dengan baris idxRow
        // dari Matrix M dihilangkan
        int i, j;
        Matrix newMat = new Matrix(M.rowNum - 1, M.colNum);
        for (i = 0; i < newMat.rowNum; i++) {
            for (j = 0; j < newMat.colNum; j++) {
                if (i < idxRow) {
                    newMat.data[i][j] = M.data[i][j];
                } else {
                    newMat.data[i][j] = M.data[i + 1][j];
                }
            }
        }
        return newMat;
    }

    // Fungsi Remove Row and Column
    public static Matrix removeRowCol(Matrix M, int idxRow, int idxCol) {
        // Membuat matrix baru dengan kolom idxCol
        // dan baris idxRow dari Matrix M dihilangkan
        return removeRow(removeCol(M, idxCol), idxRow);
    }

    // Fungsi Determinan Ekspansi Kofaktor
    public static Double detEksKof(Matrix M) {
        // Prekondisi : Matrix M persegi
        // Menghitung determinan Matrix M
        // menggunakan Ekspansi Kofaktor

        int i;
        int koef = 1;
        Double res;
        if (M.rowNum == 1) {
            res = M.data[0][0];
        } else {
            res = 0.0;
            for (i = 0; i < M.colNum; i++) {
                res += M.data[0][i] * koef * detEksKof(removeRowCol(M, 0, i));
                koef *= -1;
            }
        }
        return res;
    }

    public static void rowSwap(Matrix M, int idx1, int idx2) {
        // I.S. Matrix M terisi
        // F.S. Matrix M dengan baris idx1 dan idx2 ditukar
        Double[] temp = new Double[M.colNum];
        temp = M.data[idx1];
        M.data[idx1] = M.data[idx2];
        M.data[idx2] = temp;
    }

    public static void addMultiOfRow(Matrix M, int idxMain, int idxSecond, Double k) {
        // I.S. Matrix M terisi
        // F.S. Baris idxMain ditambahkan k kali baris idxSecond
        int i;
        for (i = 0; i < M.rowNum; i++) {
            M.data[idxMain][i] = M.data[idxMain][i] + k * M.data[idxSecond][i];
        }
    }

    public static class HasilSeg {
        // Kelas untuk menyimpan hasil transformasi ke segitiga atas
        // beserta berapa banyak pertukaran yang terjadi
        Matrix matrix;
        int swap;

        public HasilSeg(Matrix mat, int swp) {
            matrix = mat;
            swap = swp;
        }

    }

    // Prosedur to Segitiga Atas
    public static HasilSeg toSegBawah(Matrix MAwal) {
        // Prekondisi : Matrix MAwal persegi
        // Mengeluarkan matrix baru, yaitu matrix MAwal
        // yang diubah menjadi matrix segitiga bawah
        Matrix M = new Matrix();
        M.deepclone(MAwal);// Deep copy
        int rowItr; // Iterasi Row
        int rowCheck; // Iterasi row, mengecek 0
        int rowZ; // Iterasi pengenolan baris
        int swap = 0;
        Double k;

        for (rowItr = M.rowNum - 1; rowItr >= 0; rowItr--) {
            // System.out.println(rowItr);
            if (isZero(M.data[rowItr][rowItr])) {
                rowCheck = rowItr;
                while (rowCheck < M.rowNum && isZero(M.data[rowItr][rowItr])) {
                    if (!isZero(M.data[rowCheck][rowItr])) {
                        rowSwap(M, rowItr, rowCheck);
                        swap++;
                    }
                    rowCheck++;
                }
            }
            /**
             * Pakai dua if supaya karena if yang atas memastikan diagonal tidak nol
             */
            if (!isZero(M.data[rowItr][rowItr])) {
                for (rowZ = rowItr - 1; rowZ >= 0; rowZ--) {
                    k = M.data[rowZ][rowItr] / M.data[rowItr][rowItr];
                    M.display();
                    // System.out.printf("k : %f\nrowZ : %d\nrowItr : %d\n", k, rowZ, rowItr);

                    addMultiOfRow(M, rowZ, rowItr, -k);
                    //M.display();
                    System.out.printf("\n");
                }
            }

        }
        return new HasilSeg(M, swap);

    }

    public static HasilSeg toSegAtas(Matrix MAwal) {
        // Prekondisi : Matrix MAwal persegi
        // Mengeluarkan matrix baru, yaitu matrix MAwal
        // yang diubah menjadi matrix segitiga bawah
        Matrix M = new Matrix();
        M.deepclone(MAwal);// Deep copy
        int rowItr; // Iterasi Row
        int rowCheck; // Iterasi row, mengecek 0
        int rowZ; // Iterasi pengenolan baris
        int swap = 0;
        Double k;

        for (rowItr = 0; rowItr < M.rowNum; rowItr++) {
            if (isZero(M.data[rowItr][rowItr])) {
                rowCheck = rowItr;
                while (rowCheck < M.rowNum && isZero(M.data[rowItr][rowItr])) {
                    if (!isZero(M.data[rowCheck][rowItr])) {
                        rowSwap(M, rowItr, rowCheck);
                        swap++;
                    }
                    rowCheck++;
                }
            }
            /**
             * Pakai dua if supaya karena if yang atas memastikan diagonal tidak nol
             */
            if (!isZero(M.data[rowItr][rowItr])) {
                for (rowZ = rowItr + 1; rowZ < M.rowNum; rowZ++) {
                    k = M.data[rowZ][rowItr] / M.data[rowItr][rowItr];
                    // System.out.printf("k : %f\nrowZ : %d\nrowItr : %d\n", k, rowZ, rowItr);
                    //M.display();
                    addMultiOfRow(M, rowZ, rowItr, -k);
                }
            }

        }
        //M.display();
        return new HasilSeg(M, swap);

    }

    public static Double detOBE(Matrix M) {
        // Prekondisi : Matrix M persegi
        // Menghitung determinan M dengan
        // metode Operasi Baris Elementer
        HasilSeg SegAtasM = toSegAtas(M);
        Matrix MSegAtas = SegAtasM.matrix;
        int SwapSegAtas = SegAtasM.swap;
        int i = 0;
        Double det = MSegAtas.data[i][i];
        for (i = 1; i < MSegAtas.rowNum; i++) {
            det *= MSegAtas.data[i][i];
        }
        return det * Math.pow(-1, SwapSegAtas);
    }

    /*
     * Udah pindah ke Martix.java public static void parseFile(Matrix M, String
     * path) { // Prekondisi : Matrix Persegi // Mem-parse file dari path lalu
     * menyimpanya sebagai // Double di Matrix M File file = new File(path); try {
     * Scanner scan = new Scanner(file); String[] s = scan.nextLine().split(" ");
     * Integer dim = Integer.parseInt(s[0]); M.rowNum = dim; M.colNum = dim; M.data
     * = new Double[dim][dim]; int i = 0, j; for (j = 0; j < M.colNum; j++) {
     * M.data[i][j] = Double.parseDouble(s[j]); } for (i = 1; i < M.rowNum; i++) {
     * for (j = 0; j < M.colNum; j++) { M.data[i][j] = Double.parseDouble(s[j]); }
     * if (scan.hasNextLine()) { s = scan.nextLine().split(" "); } }
     * 
     * scan.close(); ; } catch (FileNotFoundException e) {
     * System.out.println("File not found!"); }
     * 
     * }
     */
    public static hasilOutput Run(int submenu, int inputType) {
        hasilOutput Result = new hasilOutput();
        Scanner sc = new Scanner(System.in);
        Matrix M = new Matrix();
        switch (inputType) {
            case 1:
                System.out.printf("Dimensi Matrix : ");
                int dim = sc.nextInt();
                M = new Matrix(dim, dim);
                M.readMatrix();
                break;

            case 2:
                M.parseFile();
                break;
        }
        switch (submenu) {
            case 1:
                Result.add(String.format("Determinan Matriks dengan metode reduksi baris : %f\n", detOBE(M)));
                break;

            case 2:
                Result.add(String.format("Determinan Matriks dengan metode ekspansi kofaktor : %f\n", detEksKof(M)));
                break;
        }
        return Result;

    }
    /*
     * public static void main(String[] args) { //Matrix M = new Matrix(new
     * Double[][] { { 0.0, 1.0, 6.0, 7.0, 8.0 }, //{ 0.0, 0.0, 0.0, 1.0, 7.0 } });
     * Matrix M = new Matrix(); parseFile(M, args[0]); //M.display();
     * //toSegAtas(M).matrix.display(); M = toSegBawah(M).matrix;
     * toSegAtas(M).matrix.display(); }
     */

}