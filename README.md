# Indobertplag  
Teks referensi yang digunakan: https://drive.google.com/drive/folders/1KUs98vnF1IrxiehZEy-U5Hr4xeGJuH5Q?usp=sharing  
**Aplikasi Cek Plagiarisme Karya Ilmiah Berbasis Web**  
Indobertplag adalah sebuah aplikasi cek plagiarisme berbasis web yang khusus digunakan untuk karya ilmiah. Aplikasi ini mengeluarkan tingkat kemiripan antara teks input dengan teks-teks yang terdaftar pada database. Dari nilai kemiripan tersebut, sebuah teks dapat diklasifikasi sebagai:  
- **Non Plagiat** (0 - 9.9%)  
- **Plagiat Ringan** (10 - 29.9%)  
- **Plagiat Berat** (30 - 100%)  

Aplikasi ini menggunakan model **IndoBERT**, sebuah model LLM berbasis arsitektur transformers, dan metode cosine similarity untuk menghitung tingkat kemiripan.  

---

## Instalasi  
### Prasyarat  
Sebelum memulai instalasi, pastikan telah mengunduh:  
- [Python versi 3.10.8](https://www.python.org/downloads/release/python-3108/)  
- [PostgreSQL versi 17.0](https://www.postgresql.org/download/)  
- [pgAdmin 4](https://www.pgadmin.org/download/)  
- IDE seperti [Visual Studio Code versi 19.5](https://code.visualstudio.com/docs/?dv=win64user)  
  
### Langkah Instalasi  
1. **Framework Django dan Library**  
   Program terdiri dari folder `indobertplag`, file `requirements.txt`, dan file `README.md`.  
   - Buka folder program dalam IDE seperti Visual Studio Code
   - Buka terminal pada IDE (untuk Visual Studio Code dapat menggunakan shortcut CTRL + Shift + `),
   - Buat sebuah virtual environment dengan menjalankan 2 command berikut: 
     ```bash  
     python -m venv Venv
     ```
     ```bash
     .\Venv\Scripts\Activate  
     ```  
   - Instal library yang diperlukan:  
     ```bash  
     pip install -r requirements.txt  
     ```  

   Command diatas akan menginsall library yang terdaftar di `requirements.txt`.  

2. **Model dan Cosine Similarity**  
   - Model **IndoBERT** (all-indobert-base-v4) diunduh otomatis oleh HuggingFace.  
   - Model tidak di download dalam bentuk file secara manual, tetapi langsung menggunakan library huggingface. Model dinyatakan dalam file utils.py (indobertplag -> plagiarism -> utils.py).
   - Running pertama kali akan mendownload model ke cache lokal yang disimpan dalam C:\Users\username\.cache\huggingface\hub.
   - Untuk menjalankan secara offline, daftarkan environment variable HF_HUB_OFFLINE=1.

3. **Konfigurasi Database**  
   - Buat database `indobertplag_db` di **pgAdmin 4**.
   - Update pengaturan `DATABASE` di `settings.py`.
   - Jalankan migrasi:
     ```bash
     python manage.py migrate
     ```

4. **Buat Superuser**  
   Tambahkan admin:
   ```bash
   python manage.py createsuperuser
   ```
   Terminal akan meminta untuk mengisi username, email, dan password. Username dan password akan digunakan untuk halaman login.

5. **Jalankan Aplikasi**  
   Untuk menjalankan aplikasi, pastikan bahwa terminal berada dalam folder indobertplag (C:\Users\user\folder_program\indobertplag).
   - Apabila terminal masih di dalam folder program, pindah ke folder indobertplag dengan menggunakan command berikut:
   ```bash
   cd indobertplag
   ```
   - Jalankan aplikasi dengan menggunakan command:
   ```bash
   python manage.py runserver
   ```

