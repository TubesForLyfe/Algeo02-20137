public class SPL_Balikan extends Matrix {

    public static boolean isEqualRowandCol(Matrix A, Matrix B){
        return ( A.rowNum == B.colNum);
    }
    public static Matrix spl_Balikan(Matrix A, Matrix B){
        Matrix Hsl;
        Matrix Ainv;
        Ainv = MatriksBalikan.inversAdjoin(A);
        Hsl = Multiply(Ainv, B);
        return Hsl;
    }

    public static void solusi(Matrix M){
        int i,j;
        for (i=0;i<M.rowNum;i++){
            for (j=0;j<M.colNum;j++){
                System.out.println("x" + i + " = " + M.data[i][j] + " ");
            }
        }
    }
    public static Matrix Multiply (Matrix A, Matrix B){
        Matrix Hsl = new Matrix(A.rowNum,B.colNum);
        for(int i=0;i<A.rowNum;i++){    
            for(int j=0;j<B.colNum;j++){    
                Hsl.data[i][j]=0.0;      
                for(int k=0;k<B.rowNum;k++)      
                {      
                    Hsl.data[i][j]+=A.data[i][k]*B.data[k][j];      
                }//end of k loop  
            }
        }
        return Hsl; 
    }

//     public static void main(String[] args) {
//         cek coba jalanin
//         Matrix M= new Matrix();
//         M.readMatrix();
//         Matrix N= new Matrix();
//         N.readMatrix();
//         M.display();
//         System.out.println("\n");
//         N.display();
//         System.out.println("\n");
//         Matrix Hsl;
//         if (isEqualRowandCol(M, N)){
//             Hsl = spl_Balikan(M, N);
//             Hsl.display();
//             solusi(Hsl);
//         }
//         else {
//             System.out.println("Persamaan Tidak bisa diselesaikan dengan menggunakan Metode Balikan");
//         }
//     }
}
