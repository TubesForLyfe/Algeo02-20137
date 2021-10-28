import java.util.MissingFormatArgumentException;
import java.util.Scanner;

import javax.lang.model.element.VariableElement;
import javax.print.attribute.HashAttributeSet;

class SPL extends Matrix {
    static Scanner input = new Scanner(System.in);

    public static void SWAP(Double a, Double b) {
        Double temp;
        temp = a;
        a = b;
        b = temp;
    }

    public static void FillMatrix() {
        int m, n;
        System.out.printf("Masukkan m: ");
        m = input.nextInt();
        System.out.printf("Masukkan n: ");
        n = input.nextInt();
        System.out.println("Matriks: ");
        Double[][] MATRIKS = new Double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                MATRIKS[i][j] = input.nextDouble();
            }
        }
    }

    public static void Gauss_elimination(int m, int n, Matrix MATRIKS) {
        /* Aplying OBE */
        int u = 0;
        Double konstanta;
        for (int i = 0; i < n - 1; i++) {
            if (u <= m - 1) {
                if (MATRIKS.data[u][i] == 0) {
                    for (int j = u + 1; j < m; j++) {
                        if (MATRIKS.data[j][i] != 0) {
                            for (int t = 0; t < n; t++) {
                                /* SWAP elemen yang nol dengan yang bukan nol */
                                SWAP(MATRIKS.data[u][t], MATRIKS.data[j][t]);
                            }
                            u += 1;
                            break;
                        }
                    }
                }
                if (MATRIKS.data[u][i] != 0) {
                    for (int t = 0; t < n; t++) {
                        MATRIKS.data[u][n - 1 - t] = MATRIKS.data[u][n - 1 - t] / MATRIKS.data[u][i];
                    }
                    for (int j = u + 1; j < m; j++) {
                        konstanta = (MATRIKS.data[j][i]) / (MATRIKS.data[u][i]);
                        for (int t = 0; t < n; t++) {
                            MATRIKS.data[j][n - 1 - t] = MATRIKS.data[j][n - 1 - t]
                                    - konstanta * MATRIKS.data[u][n - 1 - t];
                        }
                    }
                    u += 1;
                }
            } else {
                break;
            }
        }
    }

    public static boolean isSingular(int m, int n, Matrix MATRIKS) {
        if (Determinan.detOBE(MATRIKS) != 0) {
            return false;
        } else {
            return true;
        }
    }

    public static void Gauss_Jordan_elimination(int m, int n, Matrix MATRIKS) {
        Double konstanta;
        Gauss_elimination(m, n, MATRIKS);
        for (int i = 0; i < m; i++) {
            // int jumlah_loop = 0;
            for (int j = 0; j < n - 1; j++) {
                if (MATRIKS.data[m - 1 - i][j] != 0) {
                    for (int t = i + 1; t < m; t++) {
                        konstanta = (MATRIKS.data[m - 1 - t][j]) / (MATRIKS.data[m - 1 - i][j]);
                        for (int u = 0; u < n; u++) {
                            MATRIKS.data[m - 1 - t][u] = MATRIKS.data[m - 1 - t][u]
                                    - konstanta * MATRIKS.data[m - 1 - i][u];
                        }
                    }
                    break;
                }
            }
        }
    }

    public static int banyaknya_baris_nol(Matrix MATRIKS, int n, int m) {
        int jumlah_baris_nol = 0;
        boolean cek_baris_nol = true;
        int indeks = 0;
        while (indeks < m && (cek_baris_nol == true)) {
            for (int j = 0; j < n - 1; j++) {

                if (MATRIKS.data[m - 1 - indeks][j] != 0) {
                    cek_baris_nol = false;
                    break;
                }

            }
            if (cek_baris_nol == true) {
                jumlah_baris_nol += 1;
            }
            indeks += 1;
        }
        return (jumlah_baris_nol);
    }

    public static boolean Apakah_ada_Solusi(int n, int m, Matrix MATRIKS) {
        for (int i = 0; i < m; i++) {
            int jumlah_loop = 0;
            for (int j = 0; j < n - 1; j++) {
                if (MATRIKS.data[m - 1 - i][j] != 0) {
                    return true;
                } else {
                    jumlah_loop += 1;
                }
            }
            if (MATRIKS.data[m - 1 - i][n - 1] != 0) {
                if (jumlah_loop == n - 1) {
                    return false;
                }
            }
        }
        return false;
    }

    public static int BanyaknyaVariabel_Parametrik(int m, int n, Matrix MATRIKS) {
        return ((n - (m - banyaknya_baris_nol(MATRIKS, n, m))) - 1);
    }

    public static hasilOutput Gauss_Solution(int m, int n, Matrix MATRIKS) {
        // Ini support untuk print ke file
        hasilOutput res = new hasilOutput();
        Double[] array_of_nilai = new Double[n - 1];

        if (BanyaknyaVariabel_Parametrik(m, n, MATRIKS) >= 0) {
            if (Apakah_ada_Solusi(n, m, MATRIKS) == true) {
                // Inisialisasi Matriks Solusi menjadi matriks nol, yang menampung nilai
                // x_1,x_2,...,x_n dengan masing-masing mengandung beberapa komponen seperti
                // variabel parameter atau pun konstanta
                int Jumlah_Variabel_Parametrik = BanyaknyaVariabel_Parametrik(m, n, MATRIKS);
                Double[][] MATRIKS_SOLUSI = new Double[n - 1][Jumlah_Variabel_Parametrik + 1];
                Double[] Tampung = new Double[Jumlah_Variabel_Parametrik + 1];
                Integer[] Array_indeks = new Integer[n - 1 - Jumlah_Variabel_Parametrik];
                for (int i = 0; i < n - 1; i++) {
                    for (int j = 0; j < Jumlah_Variabel_Parametrik + 1; j++) {
                        MATRIKS_SOLUSI[i][j] = 0.0;
                    }
                }

                // Inisialiasi array tampungan untuk menampung penjumlah dari masing-masing
                // komponen Matriks solusi yang dikali dengan koefisien SPL.
                for (int i = 0; i < Jumlah_Variabel_Parametrik + 1; i++) {
                    Tampung[i] = 0.0;
                }
                // Jika solusi SPL tidak unik (berupa solusi parametrik).
                if (m - banyaknya_baris_nol(MATRIKS, n, m) < n - 1) {
                    // Inisialisasi Variabel Parametrik
                    for (int i = 0; i < (m - banyaknya_baris_nol(MATRIKS, n, m)); i++) {
                        for (int j = 0; j < n - 1; j++) {
                            if (MATRIKS.data[i][j] != 0) {
                                Array_indeks[i] = j;
                                break;
                            }
                        }
                    }
                    // CEK Indeks di array Indeks
                    int p = 1;
                    for (int i = 0; i < n - 1; i++) {
                        int u = 0;
                        for (int j = 0; j < Array_indeks.length; j++) {
                            if (Array_indeks[j] == i) {
                                break;
                            } else {
                                u++;
                            }
                        }
                        if (u == Array_indeks.length) {
                            MATRIKS_SOLUSI[i][p] = 1.0;
                            p++;
                        }
                    }

                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < n - 1; j++) {
                            if (MATRIKS.data[m - 1 - i][j] != 0) {
                                for (int t = j + 1; t < n - 1; t++) {
                                    for (int u = 0; u < Jumlah_Variabel_Parametrik + 1; u++) {
                                        Tampung[u] = 0.0;
                                        Tampung[u] += (MATRIKS.data[m - 1 - i][t] * MATRIKS_SOLUSI[t][u]);
                                    }
                                }
                                for (int u = 0; u < Jumlah_Variabel_Parametrik + 1; u++) {
                                    if (u == 0) {
                                        MATRIKS_SOLUSI[j][u] = MATRIKS.data[m - 1 - i][n - 1] - Tampung[u];
                                    } else {
                                        MATRIKS_SOLUSI[j][u] = -Tampung[u];
                                    }
                                }
                                break;
                            }
                        }
                    }
                    // System.out.println("Solusinya : ");
                    res.add("Solusinya :");
                    for (int i = 0; i < n - 1; i++) {
                        int indeks = 0;
                        // System.out.print("X_" + (i + 1) + " = " + tampungan_solusi[i] + ", ");
                        String UntukOutput = "";
                        UntukOutput += "X_" + (i + 1) + " = ";
                        for (int y = 0; y < Jumlah_Variabel_Parametrik + 1; y++) {
                            if (y == 0) {
                                if (MATRIKS_SOLUSI[i][y] != 0.0) {
                                    UntukOutput += Double.toString(MATRIKS_SOLUSI[i][y]);
                                    indeks += 1;
                                }
                            } else {
                                if (indeks == 0) {
                                    if (MATRIKS_SOLUSI[i][y] > 0.0) {
                                        if (MATRIKS_SOLUSI[i][y] == 1.0) {
                                            UntukOutput += "a_" + y;
                                        } else {
                                            UntukOutput += Double.toString(MATRIKS_SOLUSI[i][y]) + "a_" + y;
                                        }
                                        indeks += 1;
                                    } else if (MATRIKS_SOLUSI[i][y] < 0.0) {
                                        if (MATRIKS_SOLUSI[i][y] == -1.0) {
                                            UntukOutput += "-a_" + y;
                                        } else {
                                            UntukOutput += Double.toString(MATRIKS_SOLUSI[i][y]) + "a_" + y;
                                        }
                                        indeks += 1;
                                    }
                                } else {
                                    if (MATRIKS_SOLUSI[i][y] > 0.0) {
                                        if (MATRIKS_SOLUSI[i][y] == 1.0) {
                                            UntukOutput += " + " + "a_" + y;
                                        } else {
                                            UntukOutput += " + " + Double.toString(MATRIKS_SOLUSI[i][y]) + "a_" + y;
                                        }
                                        indeks += 1;
                                    } else if (MATRIKS_SOLUSI[i][y] < 0.0) {
                                        if (MATRIKS_SOLUSI[i][y] == -1.0) {
                                            UntukOutput += " - " + "a_" + y;
                                        } else {
                                            UntukOutput += " - " + Double.toString(-MATRIKS_SOLUSI[i][y]) + "a_" + y;
                                        }
                                        indeks += 1;
                                    }
                                }
                            }
                        }
                        if (indeks == 0) {
                            UntukOutput += 0.0;
                        }
                        res.add(UntukOutput);
                    }
                } else { // Jika Solusi SPL adalah unik.

                    for (int i = 0; i < m; i++) {
                        Double sum = 0.0;
                        for (int j = 0; j < n - 1; j++) {
                            if (MATRIKS.data[m - 1 - i][j] != 0) {
                                for (int t = j + 1; t < n - 1; t++) {
                                    sum += (MATRIKS.data[m - 1 - i][t] * array_of_nilai[t]);
                                }
                                array_of_nilai[j] = MATRIKS.data[m - 1 - i][n - 1] - sum;
                                break;
                            }
                        }
                    }
                    // System.out.println("Solusinya : ");
                    res.add("Solusinya :");
                    for (int i = 0; i < n - 1; i++) {
                        if (i < n - 2) {
                            // System.out.print("X_" + (i + 1) + " = " + array_of_nilai[i] + ", ");
                            res.add("X_" + (i + 1) + " = " + array_of_nilai[i] + ", ");
                        } else {
                            // System.out.print("X_" + (i + 1) + " = " + array_of_nilai[i]);
                            res.add("X_" + (i + 1) + " = " + array_of_nilai[i]);
                        }
                    }
                }
            } else {
                res.add("Tidak Ada Solusi");
            }
        } else {
            // System.out.println("Tidak Ada Solusi");
            res.add("Tidak Ada Solusi");
        }
        return res;
    }

    public static hasilOutput SPL_Matriks_balikan(int m, int n, Matrix MATRIKS) {
        hasilOutput res = new hasilOutput();
        Double sum;
        if (m == n - 1) {
            if (isSingular(m, n, MATRIKS) == false) {
                for (int i = 0; i < m; i++) {
                    sum = 0.0;
                    for (int j = 0; j < n - 1; j++) {
                        // Aku tambahin [i][j]
                        sum += MatriksBalikan.InversGaussJordan(MATRIKS).data[i][j] * MATRIKS.data[j][n - 1];
                    }
                    // System.out.println("X_" + (i + 1) + " = " + sum);
                    res.add("X_" + (i + 1) + " = " + sum);
                }
            } else {
                // System.out.println("Matriks Singular");
                res.add("Matriks Singular");
            }
        } else {
            // System.out.println("Maaf tidak bisa memakai metode ini, silahkan pakai metode
            // gauss elimination atau gauss jordan elimination");
            res.add("Maaf tidak bisa memakai metode ini, silahkan pakai metode gauss elimination atau gauss jordan elimination");
        }
        return res;
    }

    public static hasilOutput SPL_Cramer(int m, int n, Matrix MATRIKS) {
        // Double[][] MATRIKS_Copy = new Double[m][n];
        hasilOutput res = new hasilOutput();
        Matrix MATRIKS_Copy = new Matrix(m, n);
        if (m != n - 1) {
            // System.out.println("Maaf tidak bisa memakai metode ini, silahkan pakai metode
            // gauss elimination atau gauss jordan elimination");
            res.add("Maaf tidak bisa memakai metode ini, silahkan pakai metode gauss elimination atau gauss jordan elimination");
        } else {
            if (Determinan.detOBE(MATRIKS) != 0) {
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        MATRIKS_Copy.data[i][j] = MATRIKS.data[i][j];
                    }
                }
                for (int i = 0; i < n - 1; i++) {
                    for (int j = 0; j < m; j++) {
                        Double temp;
                        temp = MATRIKS_Copy.data[j][i];
                        MATRIKS_Copy.data[j][i] = MATRIKS_Copy.data[j][n - 1];
                        MATRIKS_Copy.data[j][n - 1] = temp;
                        // Tukar semua
                        // kolom pada baris ke-i
                        // dengan
                        // konstantanya
                    }
                    // System.out.println("X_" + (i + 1) + " = " + Determinan.detOBE(MATRIKS_Copy) /
                    // Determinan.detOBE(MATRIKS));
                    res.add("X_" + (i + 1) + " = " + Determinan.detOBE(MATRIKS_Copy) / Determinan.detOBE(MATRIKS));
                    // Balikin lagi
                    for (int j = 0; j < m; j++) {
                        Double temp;
                        temp = MATRIKS_Copy.data[j][i];
                        MATRIKS_Copy.data[j][i] = MATRIKS_Copy.data[j][n - 1];
                        MATRIKS_Copy.data[j][n - 1] = temp;
                    }
                }
            } else {
                // System.out.println("Matriks Singular");
                res.add("Matriks Singular");
            }
        }
        return res;
    }

    public static hasilOutput SPL_Gauss(int m, int n, Matrix MATRIKS) {
        hasilOutput res = new hasilOutput();
        Gauss_elimination(m, n, MATRIKS);
        res = Gauss_Solution(m, n, MATRIKS);
        return res;
    };

    public static hasilOutput SPL_Gauss_Jordan(int m, int n, Matrix MATRIKS) {
        hasilOutput res = new hasilOutput();
        Gauss_Jordan_elimination(m, n, MATRIKS);
        res = Gauss_Solution(m, n, MATRIKS);
        return res;
    };

    public static hasilOutput Run(int submenu, int inputType) {
        Scanner sc = new Scanner(System.in);
        Matrix M = new Matrix();
        hasilOutput res = new hasilOutput();
        switch (inputType) {
            case 1:
                M.readMatrix();
                break;

            case 2:
                M.parseFile();
                break;
        }
        switch (submenu) {
            case 1:
                res = SPL_Gauss(M.rowNum, M.colNum, M);
                break;
            case 2:
                res = SPL_Gauss_Jordan(M.rowNum, M.colNum, M);
                break;
            case 3:
                res = SPL_Matriks_balikan(M.rowNum, M.colNum, M);
                break;
            case 4:
                res = SPL_Cramer(M.rowNum, M.colNum, M);
                break;
        }
        return res;

    }
}
