# Tubes-Algeo_2_IF_2123# Algeo01-20116

Tugas Besar 2 IF 2123 Aljabar Linier dan Geometri Sistem Persamaan Linier, Determinan, dan Aplikasinya Semester I Tahun 2021/2022

Kelompok GWF

13520137 Muhammad Gilang R.

13520152 Muhammad Fahmi Irfan

13520160 Willy Wilsen

## How to Run

### Build

Sebelum mengeksekusi program, program harus di-compile terlebih dahulu. File `build.sh` merupakan file bash script yang dapat membantu dalam meng-compile program di Linux. Cara menggunakan build.sh adalah dengan run di terminal.

```bash
$ ./build.sh
```

Jika muncul error `bash: ./build.sh: Permission denied`, beri izin dulu dengan 

```bash
$ chmod +x build.sh
```

File kemudian akan di-compile dan disimpan di directory yang sama dengan `build.sh` dengan nama `Algeo01.jar`, yang bisa diganti dengan mengganti `build.sh`.

### Run

File `.jar` yang telah dicompile dapat dieksekusi dengan

```bash
$ java -jar Algeo01.jar
```

`Algeo01` dapat disesuaikan dengan nama file setelah di-compile


## Fitur

### Menu Utama

```bash
1. Sistem Persamaan Linier
2. Determinan
3. Matriks Balikan        
4. Interpolasi Polinom
5. Regresi linier berganda
6. Keluar
```

pilih 1-5 untuk mengakses fitur

pilih 6 untuk keluar atau menghentikan program

### Sistem Persamaan Linear

pilih angka 1-4 dari metode yang ingin digunakan

```bash
1. Metode eliminasi Gauss
2. Metode eliminasi Gauss-Jordan
3. Metode matriks balikan
4. Kaidah Cramer
```

pilih metode input dari keyboard atau file txt yang telah ada

jika dari keyboard, maka masukan jumlah baris dan kolom spl yang akan diproses

Ketik SPL dalam bentuk Matriks yang akan diolah dengan spasi disetiap kolom untuk kolom selanjutnya dan enter untuk ke baris selanjutnya

### Determinan

pilih 1 atau 2 sesuai metode yang ingin digunakan

```bash
1. Metode reduksi baris
2. Metode ekspansi kofaktor
```
pilih metode input dari keyboard atau file txt yang telah ada

Jika Input dari Keyboard, masukkan dimensi Matriks yang akan diinput

Ketik Matriks yang akan diolah dengan spasi disetiap kolom untuk ke kolom selanjutnya dan enter untuk ke baris selanjutnya

### Matriks Balikan

pilih 1 atau 2 sesuai metode yang ingin digunakan

```bash
1. Metode Eliminasi Gauss-Jordan
2. Metode Adjoin
```

pilih metode input dari keyboard atau file txt yang telah ada

Jika Input dari Keyboard, masukkan dimensi Matriks yang akan diinput

Ketik Matriks yang akan diolah dengan spasi disetiap kolom untuk ke kolom selanjutnya dan enter untuk ke baris selanjutnya

### Interpolasi Polinom

pilih metode input dari keyboard atau file txt yang telah ada

Masukkan jumlah titik yang akan dibaca

Masukkan titik-titik sesuai dengan jumlah yang sebelumnya sudah diinputkan

Masukkan nilai x yang akan ditaksir nilai y atau f(x)

jika ingin menguji kembali nilai x maka masukan 'y' pada pilihan, jika tidak maka pilih 'n'

```bash
uji x kembali (y/n):
```

Maka akan ditampilkan fungsi interpolasi polinom dan nilai taksiran dari x

### Regresi Linear Berganda

pilih metode input dari keyboard atau file txt yang telah ada

Masukkan 'n' atau banyaknya percobaan

Masukan 'k' atau banyaknya variabel bebas


### Input data 

pilih 1 jika ingin input dari keyboard, pilih 2 jika ingin input dari file txt

```bash
1. Input dari Keyboard
2. Input dari File txt
```
