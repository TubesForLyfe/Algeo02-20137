import java.util.Scanner;

public class Menu {
    
    public static int mainMenu() {
        Scanner scanner = new Scanner(System.in);
        int mainmenu = 0;
        System.out.println("\nMENU");
        System.out.println("1. Sistem Persamaan Linier");
        System.out.println("2. Determinan");
        System.out.println("3. Matriks Balikan");
        System.out.println("4. Interpolasi Polinom");
        System.out.println("5. Regresi linier berganda");
        System.out.println("6. Keluar");

        System.out.print("Masukan pilihan : ");
        mainmenu = scanner.nextInt();
        if (notValid(1, 6, mainmenu)){
            System.out.println("Input salah!");
            mainmenu = mainMenu();
        }
        return mainmenu;
    }

    public static int submenu_spl(){
        Scanner scanner = new Scanner(System.in);
        int submenu = 0;

        System.out.println("\n Pilih metode untuk menyelesaikan Sistem Persamaan Linier");
        System.out.println("1. Metode eliminasi Gauss");
        System.out.println("2. Metode eliminasi Gauss-Jordan");
        System.out.println("3. Metode matriks balikan");
        System.out.println("4. Kaidah Cramer");

        System.out.print("Masukan pilihan : ");
        submenu = scanner.nextInt();
        if (notValid(1, 4, submenu)){
            System.out.println("Input salah!");
            submenu = submenu_spl();
        }
        return submenu;
    }

    public static int submenu_determinan(){
        Scanner scanner = new Scanner(System.in);
        int submenu = 0;

        System.out.println("\n Pilih metode untuk menghitung Determinan");
        System.out.println("\n1. Metode reduksi baris");
        System.out.println("2. Metode ekspansi kofaktor");

        System.out.print("Masukan pilihan : ");
        submenu = scanner.nextInt();
        if (notValid(1, 2, submenu)){
            System.out.println("Input salah!");
            submenu = submenu_determinan();
        }
        return submenu;
    }

    public static int submenu_invers(){
        Scanner scanner = new Scanner(System.in);
        int submenu = 0;

        System.out.println("\n Pilih metode untuk menghitung inver");
        System.out.println("\n1. Metode Eliminasi Gauss-Jordan");
        System.out.println("2. Metode Adjoin");

        System.out.print("Masukan pilihan : ");
        submenu = scanner.nextInt();
        if (notValid(1, 2, submenu)){
            System.out.println("Input salah!");
            submenu = submenu_invers();
        }
        return submenu;
    }

    public static int input() {
        Scanner scanner = new Scanner(System.in);
        int input = 0;
        System.out.println("\n1. Input dari Keyboard");
        System.out.println("2. Input dari File txt");
        System.out.print("Masukan pilihan: ");

        input = scanner.nextInt();
        if (notValid(1, 2, input)){
            System.out.println("Input salah!");
            input = input();
        }
        return input;
    }

    public static boolean notValid(int LBound, int UBound, int menu){
        return menu < LBound || menu > UBound;
    }

}
