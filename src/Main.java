import java.util.Scanner;
public class Main {
    public static void main(String[] args){
        Scanner scanner = new Scanner(System.in);
        boolean loop = true;
        int mainmenu, submenu, inputType;
        hasilOutput hasil = new hasilOutput();
        while(loop) {
            mainmenu = Menu.mainMenu();
            switch (mainmenu) {
                case 1:
                    submenu = Menu.submenu_spl();
                    inputType = Menu.input();
                    hasil = SPL.Run(submenu, inputType);
                    break;
                case 2:
                    submenu = Menu.submenu_determinan();
                    inputType = Menu.input();
                    hasil = Determinan.Run(submenu, inputType);
                    break;
                case 3:
                    submenu = Menu.submenu_invers();
                    inputType = Menu.input();
                    hasil = MatriksBalikan.Run(submenu, inputType);
                    break;
                case 4:
                    inputType = Menu.input();
                    hasil = Interpolasi.Run(inputType);
                    break;
                case 5:
                    inputType = Menu.input();
                    hasil = Regresi_linier_berganda.RegresiLinear(inputType);
                    break;
                case 6:
                    loop = false;
                    break;
                default:
                    System.out.println("Masukkan input yang benar");
                    break;
            }
            if (mainmenu >= 1 && mainmenu <= 5){
                hasil.display();
                System.out.printf("Simpan ke file? (y/n): ");
                String s = scanner.nextLine();
                if (s.equals("y")){
                    hasil.toFile();
                }
            }
            
        }
    }
}
