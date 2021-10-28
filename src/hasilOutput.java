import java.util.ArrayList;
import java.util.Scanner;
import java.io.*;

public class hasilOutput {
    ArrayList<String> output = new ArrayList<String>();

    public hasilOutput() {

    }

    public void add(String s) {
        output.add(s);
    }

    public void addMatrix(Matrix M){
        int i, j;
        String s;
        for (i = 0; i < M.rowNum; i++) {
            s = "";
            for (j = 0; j < M.colNum; j++) {
                // System.out.printf("%f ", M.data[i][j]);
                s += " " + M.data[i][j];
            }
            add(s);
            //System.out.println("");
        }
    }

    public void addHasil(hasilOutput res){
        output.addAll(res.output);
    }

    public void display() {
        int i;
        for (i = 0; i < output.size(); i++) {
            System.out.println(output.get(i));
        }
    }

    public void toFile() {
        Scanner sc = new Scanner(System.in);
        System.out.printf("File Path: ");
        String path = sc.nextLine();
        File file = new File(path);

        while (file.exists()) {
            System.out.println("File exists!");
            path = sc.nextLine();
            file = new File(path);
        }
        
        try {
            file.createNewFile();
            FileWriter fw = new FileWriter(file.getAbsolutePath());
            BufferedWriter bw = new BufferedWriter(fw);
            int i;
            for (i = 0; i < output.size(); i++) {
                bw.append(output.get(i));
            }
            bw.close();
            System.out.println("Penyimpanan berhasil!");
        } catch (IOException e) {
            System.out.println("I/O Error");
        }


    }

}
