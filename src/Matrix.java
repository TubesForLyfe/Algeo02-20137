import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

public class Matrix {

    public Double[][] data;
    public Integer colNum;
    public Integer rowNum;
    static Scanner sc = new Scanner(System.in);

    public Matrix(Double[][] input) {
        data = input;
        rowNum = data.length;
        colNum = data[0].length;
    }

    public Matrix(int inputRow, int inputCol) {
        rowNum = inputRow;
        colNum = inputCol;
        data = new Double[inputRow][inputCol];
        for (int i = 0; i < inputRow; i++) {
            for (int j = 0; j < inputCol; j++) {
                data[i][j] = null;
            }
        } // kalo diisi nul ga bisa diinput
          // lah kok aku coba bisa ?

    }

    public Matrix() {
        // Matrix kosong beneran kosong
        data = null;
        colNum = null;
        rowNum = null;
    }

    public void display() {
        int i, j;
        for (i = 0; i < rowNum; i++) {
            for (j = 0; j < colNum; j++) {
                System.out.printf("%f ", data[i][j]);
            }
            System.out.println("");
        }
    }

    public void deepclone(Matrix M) {
        int i, j;
        data = new Double[M.rowNum][M.colNum];
        rowNum = M.rowNum;
        colNum = M.colNum;
        for (i = 0; i < rowNum; i++) {
            for (j = 0; j < colNum; j++) {
                data[i][j] = M.data[i][j];
            }
        }
    }

    public boolean isSquare() {
        return rowNum == colNum;
    }

    public Matrix transpose() {

        Double[][] data_transpose = new Double[colNum][rowNum];

        int i, j;
        for (i = 0; i < rowNum; i++) {
            for (j = 0; j < colNum; j++) {
                data_transpose[j][i] = data[i][j];
            }
        }
        return new Matrix(data_transpose);
    }

    public Double determinant(int row) {
        Double res;
        if (colNum != rowNum) {
            System.out.println("Matriks tidak persegi!");
            res = null;
        } else if (row > rowNum) {
            System.out.println("Baris diluar matrix!");
            res = null;
        } else if (colNum == 1) {
            res = data[0][0];
        } else {
            int i;
            res = 0.0;
            for (i = 0; i < colNum; i++) {
                res += data[row][i] * subMatrix(row, i).determinant() * Math.pow(-1, i + row);
            }
        }
        return res;
    }

    public Double determinant() {
        return determinant(0);
    }

    public Matrix subMatrix(int rowIdx, int colIdx) {
        int i, j, subCol = 0, subRow = 0;
        Double[][] subData = new Double[rowNum - 1][colNum - 1];
        for (i = 0; i < rowNum; i++) {
            subCol = 0;
            for (j = 0; j < colNum; j++) {
                if (i != rowIdx && j != colIdx) {
                    subData[subRow][subCol++] = data[i][j];
                }
            }
            if (i != rowIdx) {
                subRow++;
            }
        }
        return new Matrix(subData);
    }

    public Matrix minor() {
        Double[][] new_data = new Double[rowNum][colNum];
        int i, j;
        for (i = 0; i < rowNum; i++) {
            for (j = 0; j < colNum; j++) {
                new_data[i][j] = subMatrix(i, j).determinant();
            }

        }
        return new Matrix(new_data);
    }

    public Matrix cofactor() {
        Double[][] new_data = new Double[rowNum][colNum];
        int i, j;
        for (i = 0; i < rowNum; i++) {
            for (j = 0; j < colNum; j++) {
                new_data[i][j] = subMatrix(i, j).determinant() * Math.pow(-1, i + j);
            }

        }
        return new Matrix(new_data);
    }

    public void readMatrix() {
        // KAMUS LOKAL
        int i, j;
        // ALGORITMA

        // Kalau matrix polos, bakalan minta baris dan kolom
        // kalau udah ada ukuran, ditimpa
        if (rowNum == null || colNum == null) {
            System.out.printf("Masukkan Jumlah Baris: ");
            rowNum = sc.nextInt();
            System.out.printf("Masukkan Jumlah Kolom: ");
            colNum = sc.nextInt();
            data = new Double[rowNum][colNum];
        }
        System.out.println("Matrix: ");
        for (i = 0; i < rowNum; i++) {
            for (j = 0; j < colNum; j++) {
                data[i][j] = sc.nextDouble();
            }
        }
    }

    public void parseFile() {
        Scanner sc = new Scanner(System.in);
        System.out.printf("File path : ");
        String pathToFile = sc.nextLine();
        File file = new File(pathToFile);
        while(!file.exists()){
            System.out.println("File not found!");
            System.out.printf("File path : ");
            pathToFile = sc.nextLine();
            file = new File(pathToFile);
        }
        try {
            Scanner scan = new Scanner(file);
            String[] s;
            rowNum = fileRowNum(pathToFile);
            colNum = fileColNum(pathToFile);
            data = new Double[rowNum][colNum];
            int i = 0, j;
            while (scan.hasNextLine()) {
                s = scan.nextLine().split(" ");
                for (j = 0; j < colNum; j++) {
                    data[i][j] = Double.parseDouble(s[j]);
                }
                i++;
            }
        } catch (FileNotFoundException e) {
            System.out.println("File not found!");
        }
    }

    public int fileRowNum(String path) {
        File file = new File(path);
        int i = 0;
        try {
            Scanner scan = new Scanner(file);
            while (scan.hasNextLine()) {
                scan.nextLine();
                i++;
            }
        } catch (FileNotFoundException e) {
            System.out.println("File invalid!");
        }
        return i;
    }

    public int fileColNum(String path) {
        File file = new File(path);
        int i = 0;
        try {
            Scanner scan = new Scanner(file);
            i = scan.nextLine().split(" ").length;
        } catch (FileNotFoundException e) {
            System.out.println("File invalid!");
        }
        return i;
    }

}