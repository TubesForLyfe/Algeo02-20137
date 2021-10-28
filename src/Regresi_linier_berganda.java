import java.util.Scanner;

class Regresi_linier_berganda {

    public static void SWAP(Double a, Double b) {
        Double temp;
        temp = a;
        a = b;
        b = temp;
    }

    public static hasilOutput RegresiLinear(int inputType) {
        Scanner input = new Scanner(System.in);
        hasilOutput res = new hasilOutput();
        int n, k;

        Double[][] X;
        Double[][] MATRIKS;
        Double[] array_of_nilai;
        Double[] uji;
        Double nilai_taksiran;
        Matrix M = new Matrix();
        if (inputType == 1) {
            System.out.print("n (Banyak percobaan) : ");
            n = input.nextInt();
            System.out.print("k (Banyak variabel bebas): ");
            k = input.nextInt();
            X = new Double[n][k + 1];
            MATRIKS = new Double[k + 1][k + 2];
            array_of_nilai = new Double[k + 1];
            uji = new Double[k];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < k + 1; j++) {
                    X[i][j] = input.nextDouble();
                }
            }
        } else {
            M.parseFile();
            n = M.rowNum;
            k = M.colNum - 1;
            X = M.data;
            MATRIKS = new Double[k + 1][k + 2];
            array_of_nilai = new Double[k + 1];
            uji = new Double[k];
        }

        for (int i = 0; i < k; i++) {
            uji[i] = input.nextDouble();
        }
        for (int i = 0; i < k + 1; i++) {
            for (int j = 0; j < k + 2; j++) {
                MATRIKS[i][j] = 0.0;
            }
        }
        for (int i = 0; i <= k; i++) {
            for (int j = 0; j <= k + 1; j++) {
                for (int u = 0; u < n; u++) {
                    if (i == 0) {
                        if (j == 0) {
                            MATRIKS[i][j] += 1;
                        } else {
                            MATRIKS[i][j] += X[u][j - 1];
                        }
                    } else {
                        if (j == 0) {
                            MATRIKS[i][j] += X[u][i - 1];
                        } else {
                            MATRIKS[i][j] += X[u][i - 1] * X[u][j - 1];
                        }
                    }
                }
            }
        }

        /* Aply Gauss Elimination */
        int u = 0;
        Double konstanta;
        for (int i = 0; i < k + 1; i++) {
            if (u <= k) {
                if (MATRIKS[u][i] == 0) {
                    for (int j = u + 1; j < k + 1; j++) {
                        if (MATRIKS[j][i] != 0) {
                            for (int t = 0; t < k + 2; t++) {
                                /* SWAP elemen yang nol dengan yang bukan nol */
                                SWAP(MATRIKS[u][t], MATRIKS[j][t]);
                            }
                            u += 1;
                            break;
                        }
                    }
                }
                if (MATRIKS[u][i] != 0) {
                    for (int t = 0; t < k + 2; t++) {
                        MATRIKS[u][k + 1 - t] = MATRIKS[u][k + 1 - t] / MATRIKS[u][i];
                    }
                    for (int j = u + 1; j < k + 1; j++) {
                        konstanta = (MATRIKS[j][i]) / (MATRIKS[u][i]);
                        for (int t = 0; t < k + 2; t++) {
                            MATRIKS[j][k + 1 - t] = MATRIKS[j][k + 1 - t] - konstanta * MATRIKS[u][k + 1 - t];
                        }
                    }
                    u += 1;
                }
            } else {
                break;
            }
        }

        for (int i = 0; i < k + 1; i++) {
            Double sum = 0.0;
            for (int j = 0; j < k + 1; j++) {
                if (MATRIKS[k - i][j] != 0) {
                    for (int t = j + 1; t < k + 1; t++) {
                        sum += (MATRIKS[k - i][t] * array_of_nilai[t]);
                    }
                    array_of_nilai[j] = MATRIKS[k - i][k + 1] - sum;
                    break;
                }
            }
        }
        // System.out.println("Persamaan regresi : ");
        res.add("Persamaan regresi :");

        // System.out.print("Y = ");

        String s;
        s = "Y = ";
        for (int i = 0; i < k + 1; i++) {
            if (i == 0) {
                s += array_of_nilai[i] + " X_" + (i + 1);
            } else {
                if (array_of_nilai[i] >= 0) {
                    s += " + " + array_of_nilai[i] + " X_" + (i + 1);
                } else {
                    s += " - " + (-array_of_nilai[i]) + " X_" + (i + 1);
                }
            }
        }
        res.add(s);
        nilai_taksiran = 0.0;
        for (int i = 0; i < k + 1; i++) {
            if (i < k) {
                nilai_taksiran += uji[i] * array_of_nilai[i];
            } else {
                nilai_taksiran += array_of_nilai[i];
            }
        }
        res.add("Nilai taksiran : " + nilai_taksiran);
        // System.out.println(nilai_taksiran);
        return res;

    }

}
