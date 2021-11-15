### Tubes-Algeo_2_IF_20137

Tugas Besar 2 IF 2123 Aljabar Linier dan Geometri Aplikasi Nilai Eigen dan Vektor Eigen dalam Kompresi Gambar Semester I Tahun 2021/2022

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

File kemudian akan di-compile dan disimpan di directory yang sama dengan `build.sh` dengan nama `app.py`, yang bisa diganti dengan mengganti `build.sh`.

### Run

File `.py` dapat dieksekusi pada terminal dengan

```bash
python app.py
```

Kemudian, jalankan `localhost` pada peramban.


## Fitur

```
1. Pilih File
2. Pilih persentase kompresi
3. Kompresi File
```

Pilih metode input bisa langsung dari web dengan mengklik tombol pilih file dan mengisi persentase kompresi

File yang telah dikompress bisa disimpan dengan cara mengklik tombol save yang ada pada plot gambar yang telah dikompress
