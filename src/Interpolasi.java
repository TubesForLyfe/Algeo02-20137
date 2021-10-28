import java.util.Scanner;

public class Interpolasi extends Matrix {

    public static void readTitik(Matrix M) {
        // Membaca titik-titik yang akan dicari persamaan interpolasinya sebanyak n.
        // dengan n merupakan jumlah titik yang akan diinput dari pengguna
        // KAMUS LOKAL
        Scanner sc = new Scanner(System.in);
        int i, j;
        // ALGORITMA
        if (M.rowNum == null || M.colNum == null) {
            System.out.printf("Masukkan Jumlah Titik: ");
            M.rowNum = sc.nextInt();
            M.colNum = 2; // karena input titik x,y
            M.data = new Double[M.rowNum][M.colNum];
        }
        System.out.println("Matrix :");
        for (i = 0; i < M.rowNum; i++) {
            for (j = 0; j < M.colNum; j++) {
                M.data[i][j] = sc.nextDouble();
            }
        }
    }

    public static Double pangkat(int i, Double x) {
        // menghasilkan pangkat i dari x
        int j;
        Double Hsl = x;
        for (j = 1; j < i; j++) {
            Hsl = Hsl * x;
        }
        return Hsl;
    }

    public static Matrix createInterpolasi(Matrix M) {
        // Menghasilkan Matriks Interpolasi dari titik-titik yang sebelumnya sudah dibaca
        Matrix Hsl = new Matrix(M.rowNum, M.rowNum + 1);
        int i, j;
        for (i = 0; i < Hsl.rowNum; i++) {
            for (j = 0; j < Hsl.colNum; j++) {
                if (j == 0) {
                    Hsl.data[i][j] = 1.0;
                } else if (j == Hsl.colNum - 1) {
                    Hsl.data[i][j] = M.data[i][1];
                } else {
                    Hsl.data[i][j] = pangkat(j, M.data[i][0]);
                }
            }
        }
        return Hsl;
    }

    public static Matrix SplitHasilInterpolasi(Matrix M) {
        // 
        // pre kondisi kolom matriks berjumlah genap
        Matrix Hsl = new Matrix(M.rowNum, 1);
        for (int i = 0; i < M.rowNum; i++) {
            Hsl.data[i][0] = M.data[i][M.colNum - 1];

        }
        return Hsl;
    }

    public static Matrix Interp(Matrix M) {
        // Melakukan interpolasi,memanggil method createInterpolasi, 
        // Gauss Jordan di file MatriksBalikan.java, dan SplitHasilInterpolasi, sehingga didapatkan Matriks Hasil Interpolasi
        Matrix Minter, Hsl;
        Minter = createInterpolasi(M);
        MatriksBalikan.GaussJordan(Minter);
        Hsl = SplitHasilInterpolasi(Minter);
        return Hsl;
    }

    public static hasilOutput solution(Matrix M) {
        // Menghasilkan fungsi Interpolasi dalam bentuk f(x) =a +bx^2 +...
        hasilOutput res = new hasilOutput();
        int i, j;
        String s = "";
        s += ("Persamaan interpolasinya adalah: \n");
        s += "f(x) = ";
        for (i = 0; i < M.rowNum; i++) {
            if (M.data[i][0] != 0.0) {
                s += Double.toString(M.data[i][0]);
                if (i == 1) {
                    s += "x";
                }
                if (i != 0 && i != 1) {
                    s += "x^" + i;
                }
                j = i + 1;
                if (j < M.rowNum) {
                    if (M.data[j][0] != 0) {
                        s += " + ";
                    }
                }
            }
        }
        res.add(s);
        return res;
    }

    public static hasilOutput estimate(Matrix M, double x) {
        // Menhasilkan nilai taksiran dari x terhadap fungsi interpolasi f(x)
        hasilOutput res = new hasilOutput();
        int i;
        Double sum = M.data[0][0];
        for (i = 1; i < M.rowNum; i++) {
            sum += M.data[i][0] * pangkat(i, x);
        }
        res.add(String.format("Hasil taksiran untuk x = %f : f(%f) = %f", x, x, sum));
        return res;
    }

    public static Double findY(Double x, Matrix M){
        // Menghasilkan nilai taksiran x pada persamaan Interpolasi f(x) 
        int i;
        Double sum;
        sum = M.data[0][0];
        for (i=1;i<M.rowNum;i++){
            sum = sum + M.data[i][0]*x;
            x = x*x;
        }
        return sum;
    }

    
    public static hasilOutput Run(int inputType) {
        hasilOutput res = new hasilOutput();
        Scanner sc = new Scanner(System.in);
        Matrix M = new Matrix();
        Matrix Hsl;
        boolean repeat = true;
        switch (inputType) {
            case 1:
                readTitik(M);
                break;

            case 2:
                M.parseFile();
                break;
        }
        Hsl = Interp(M);
        res = solution(Hsl);
        System.out.println("");
        while (repeat ==true){
            System.out.print("Nilai x yang akan ditaksir : ");
            res.addHasil(estimate(Hsl, sc.nextDouble()));
            System.out.print("uji x kembali (y/n): ");
            char c = sc.next().charAt(0);
            // Java bandingin Stringnya pakai c.equals(String s)
            // karena kalau kayak c != b itu cuma bandingin addressnya aja
            if (c!='y' && c!='Y'){
                repeat = false;
            }
        }
        return res;
    }
}