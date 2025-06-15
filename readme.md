# Laporan Proyek Machine Learning - Muhammad Ikram Sabila Rasyad

## 📌 **Project Overview**

Sistem rekomendasi telah menjadi bagian penting dalam berbagai platform digital, termasuk industri buku. Seiring pertumbuhan pesat data pengguna dan katalog buku yang sangat besar, pengguna kesulitan menemukan bacaan yang sesuai dengan preferensi mereka secara efisien. Oleh karena itu, pengembangan sistem rekomendasi buku menjadi sangat penting untuk meningkatkan pengalaman pengguna serta mendorong keterlibatan dan loyalitas.

Menurut penelitian oleh Ricci et al. (2011), sistem rekomendasi mampu meningkatkan penjualan dan retensi pengguna dengan menyajikan konten yang relevan. Secara khusus dalam domain literatur, Goodreads, Amazon, dan Google Books menggunakan sistem ini untuk mempertemukan pembaca dengan buku yang sesuai. Mengadaptasi pendekatan yang serupa dapat memberikan nilai tambah dalam skenario serupa, seperti katalog digital atau platform e-learning.

**Mengapa masalah ini penting untuk diselesaikan?**

* Pengguna tidak mungkin secara manual mengevaluasi ribuan buku untuk menemukan yang cocok.
* Memberikan rekomendasi yang personal dapat meningkatkan waktu keterlibatan pengguna dan kepuasan mereka.
* Dalam konteks bisnis, sistem rekomendasi dapat mendorong pembelian dan peminatan konten spesifik.

📚 **Referensi:**

* Ricci, F., Rokach, L., & Shapira, B. (2011). *Introduction to Recommender Systems Handbook*. Springer. [https://doi.org/10.1007/978-0-387-85820-3\_1](https://doi.org/10.1007/978-0-387-85820-3_1)
* Schafer, J. B., Konstan, J., & Riedl, J. (2001). *E-Commerce Recommendation Applications*. *Data Mining and Knowledge Discovery*, 5(1–2), 115–153. [https://doi.org/10.1023/A:1009804230409](https://doi.org/10.1023/A:1009804230409)

---

## 📌 **Business Understanding**

### 🎯 Problem Statements

1. **Bagaimana sistem dapat merekomendasikan buku yang relevan untuk pengguna baru maupun pengguna lama?**
   Banyak pengguna memiliki preferensi berbeda, dan buku-buku yang populer belum tentu cocok untuk semua orang. Sistem perlu mampu memahami selera personal dari data historis.

2. **Bagaimana memanfaatkan metadata buku seperti judul dan genre untuk memberikan rekomendasi ketika data pengguna terbatas?**
   Dalam kasus cold-start (data pengguna terbatas), perlu pendekatan konten untuk tetap memberi rekomendasi masuk akal.

3. **Bagaimana mengukur performa model rekomendasi secara objektif?**
   Evaluasi perlu dilakukan secara sistematis menggunakan metrik-metrik yang telah diakui dalam dunia sistem rekomendasi.

---

### 🎯 Goals

1. **Mengembangkan dua pendekatan sistem rekomendasi**: Content-Based Filtering (CBF) dan User-Based Collaborative Filtering (UBCF) untuk memberikan rekomendasi buku personal.
   ➤ CBF digunakan untuk cold-start atau user baru, UBCF untuk user dengan cukup histori.

2. **Memanfaatkan metadata buku (judul) dan histori interaksi pengguna (rating)** sebagai sumber data utama untuk membangun model.
   ➤ Menggunakan TF-IDF + cosine similarity untuk CBF, dan K-Nearest Neighbors (KNN) berbasis vektor user-item untuk UBCF.

3. **Melakukan evaluasi objektif terhadap performa model menggunakan metrik** seperti Precision\@10, Recall\@10, MAP\@10, dan NDCG\@10.
   ➤ Evaluasi dilakukan dengan ground-truth data dari pembagian data training-test pengguna.

---

### 🛠️ Solution Approach

**Solution 1: Content-Based Filtering (CBF)**

* Menggunakan TF-IDF untuk memodelkan representasi vektor judul buku.
* Untuk setiap user, diambil 1 buku dengan rating tertinggi sebagai query, lalu dicari buku-buku serupa.
* Cocok untuk user dengan sedikit interaksi atau dalam situasi cold-start.

**Solution 2: User-Based Collaborative Filtering (UBCF)**

* Membangun matriks user-item dari data rating pengguna.
* Menggunakan algoritma KNN dengan metrik cosine similarity untuk mencari user serupa.
* Buku yang disukai oleh user-user serupa akan direkomendasikan.

---

## 📊 **Data Understanding**

### 🗃️ Deskripsi Dataset

Dataset yang digunakan dalam proyek ini merupakan dataset buku dan rating pengguna yang berasal dari [Book-Crossing: User review ratings](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset/data).

Dataset ini terdiri dari tiga file utama:

* `BX_Books.csv` — berisi informasi detail tentang buku.
* `BX_Users.csv` — berisi informasi tentang pengguna.
* `BX-Book-Ratings.csv` — berisi data interaksi pengguna dalam bentuk rating buku.

📦 **Ukuran data setelah pemrosesan awal (filtering user aktif dan buku populer)**:

* Jumlah pengguna: 271.379 
* Jumlah buku: 278.858 
* Jumlah interaksi rating: 1.149.780 

---

### 📄 Deskripsi Fitur

Berikut adalah penjelasan fitur untuk masing-masing file:

#### **1. BX-Books.csv**

| Fitur                                       | Deskripsi                                             |
| ------------------------------------------- | ----------------------------------------------------- |
| `ISBN`                                      | Kode unik buku (International Standard Book Number)   |
| `Book-Title`                                | Judul buku                                            |
| `Book-Author`                               | Penulis buku                                          |
| `Year-Of-Publication`                       | Tahun terbit                                          |
| `Publisher`                                 | Nama penerbit                                         |
| `Image-URL-S`, `Image-URL-M`, `Image-URL-L` | Link gambar sampul (tidak digunakan dalam proyek ini) |

#### **2. BX-Users.csv**

| Fitur      | Deskripsi                                                    |
| ---------- | ------------------------------------------------------------ |
| `User-ID`  | ID unik pengguna                                             |
| `Location` | Lokasi pengguna (kota, negara bagian, negara)                |
| `Age`      | Usia pengguna (dalam beberapa kasus kosong atau tidak valid) |

#### **3. BX-Book-Ratings.csv**

| Fitur         | Deskripsi                                                                                |
| ------------- | ---------------------------------------------------------------------------------------- |
| `User-ID`     | ID pengguna yang memberi rating                                                          |
| `ISBN`        | Kode buku yang diberi rating                                                             |
| `Book-Rating` | Skor rating dari 0–10. Nilai 0 umumnya berarti implicit feedback (tanpa opini eksplisit) |

---

### 🔎 Exploratory Data Analysis (EDA)

Beberapa tahapan eksplorasi data yang telah dilakukan:

* **Distribusi Rating**:
  Mayoritas rating berada pada rentang 5–10. Rating 0 cukup dominan karena mewakili implicit feedback.

* **Distribusi Usia Pengguna**:
  Terlihat banyak data usia tidak valid (seperti 0 atau >100). Data usia dibersihkan sebelum digunakan.

* **Frekuensi Rating oleh Pengguna**:
  Hanya sebagian kecil pengguna yang aktif (banyak memberi rating). Oleh karena itu, filtering dilakukan untuk hanya mempertahankan pengguna yang memberi ≥10 rating dan buku yang mendapat ≥10 rating.

```python
# Contoh filtering
active_users = ratings['User-ID'].value_counts()
active_users = active_users[active_users >= 10].index

popular_books = ratings['ISBN'].value_counts()
popular_books = popular_books[popular_books >= 10].index

df_filtered = ratings[
    ratings['User-ID'].isin(active_users) &
    ratings['ISBN'].isin(popular_books)
]
```

---

## 📊 Data Preparation

Tahapan *data preparation* sangat penting dalam proses pengembangan sistem rekomendasi, khususnya untuk pendekatan *Content-Based Filtering (CBF)*. Dalam proyek ini, data preparation dilakukan secara bertahap dan sistematis agar model dapat bekerja secara optimal dan akurat. Berikut adalah tahapan yang dilakukan:

---

### 1. **Seleksi Kolom Relevan**

Langkah pertama adalah memilih hanya kolom yang relevan dari dataset buku:

📌 **Tujuan**: Mengurangi kompleksitas dan hanya mempertahankan fitur-fitur yang dibutuhkan untuk membangun fitur konten.

---

### 2. **Menghapus Nilai Kosong (Missing Values)**

📌 **Tujuan**: Menghindari error saat proses pemrosesan teks dan membangun model. Data yang tidak lengkap berisiko menurunkan akurasi sistem.

---

### 3. **Normalisasi Teks (Lowercasing)**

📌 **Tujuan**: Menyamakan format teks agar tidak terjadi perbedaan karena huruf kapital, yang dapat memengaruhi hasil tokenisasi dan pencocokan teks saat menghitung kemiripan.

---

### 4. **Membuat Fitur Gabungan Konten**

📌 **Tujuan**: Menggabungkan fitur-fitur penting menjadi satu representasi teks untuk digunakan dalam proses *TF-IDF vectorization*. Ini adalah inti dari sistem CBF.

---

### 5. **Filtering Buku Berdasarkan Popularitas**

📌 **Tujuan**: Membatasi data hanya pada 500 buku yang paling sering dirating. Ini dilakukan untuk mengurangi noise dan meningkatkan efisiensi komputasi tanpa mengorbankan kualitas rekomendasi.

---

### 6. **Membangun Indeks Judul Buku**

📌 **Tujuan**: Membuat indeks pencarian cepat berdasarkan judul buku. Ini sangat membantu dalam proses pencarian kemiripan konten menggunakan cosine similarity.

---

## ✍️ Kesimpulan

Setiap langkah dalam proses *data preparation* dirancang untuk meningkatkan kualitas data yang digunakan oleh algoritma *Content-Based Filtering*. Normalisasi, penggabungan fitur, hingga filtering data bertujuan untuk memastikan bahwa sistem rekomendasi yang dibangun mampu memberikan hasil yang relevan, efisien, dan bebas dari error akibat kualitas data yang buruk.

---

## 🤖 Modeling

Pada tahap ini, kami membangun dan membandingkan dua pendekatan sistem rekomendasi untuk menghasilkan **Top-N Recommendation** bagi pengguna. Kedua pendekatan tersebut adalah:

1. **Content-Based Filtering (CBF)**
2. **User-Based Collaborative Filtering (UBCF)**

---

### 📌 1. Content-Based Filtering (CBF)

CBF merekomendasikan item berdasarkan kemiripan konten item itu sendiri. Dalam proyek ini, kami membangun representasi konten buku dengan menggabungkan fitur teks seperti judul, penulis, dan penerbit, lalu mengubahnya menjadi vektor menggunakan **TF-IDF Vectorizer**.

* **Proses:**

  * Menggabungkan fitur teks buku menjadi satu fitur konten.
  * Menghitung bobot kata menggunakan TF-IDF.
  * Mengukur kemiripan antar buku menggunakan **cosine similarity**.
  * Untuk setiap user, sistem mengambil satu buku yang paling tinggi rating-nya sebagai *query*, lalu mencari buku serupa berdasarkan konten.

* **Contoh Output (Top-5 Recommendation):**

```python
cb_recommend("cat & mouse (alex cross novels)", 5)
```

* **Kelebihan:**

  * Tidak bergantung pada interaksi pengguna lain.
  * Cocok untuk pengguna baru yang sudah memiliki sedikit interaksi.
  * Mampu menjelaskan alasan rekomendasi (“karena mirip dengan buku sebelumnya”).

* **Kekurangan:**

  * Tidak bisa merekomendasikan item di luar yang sudah pernah dilihat pengguna (sering disebut *serendipity problem*).
  * Tidak memanfaatkan informasi preferensi komunitas pengguna lain.

---

### 📌 2. User-Based Collaborative Filtering (UBCF)

UBCF merekomendasikan item berdasarkan kesamaan perilaku antar pengguna. Jika dua pengguna memiliki preferensi yang serupa, maka sistem akan merekomendasikan buku yang disukai oleh salah satu ke pengguna lainnya.

* **Proses:**

  * Membentuk matriks pengguna-item dari data rating.
  * Menghitung kesamaan antar pengguna menggunakan **cosine similarity** pada data sparse matrix.
  * Mengambil user-user terdekat (nearest neighbors) dan merekomendasikan buku yang belum pernah dibaca tetapi populer di kalangan tetangga tersebut.

* **Contoh Output:**
  Rekomendasi top-N buku untuk user tertentu berdasarkan rating user-user serupa.

* **Kelebihan:**

  * Menangkap pola preferensi dari komunitas secara global.
  * Tidak membutuhkan data fitur item (judul, genre, dll).

* **Kekurangan:**

  * Sulit bekerja dengan user baru yang belum pernah memberikan rating (*cold start problem*).
  * Performanya menurun jika data sangat sparse (banyak user tapi sedikit interaksi).

---

### ✅ Kesimpulan

Berdasarkan hasil evaluasi, pendekatan **User-Based Collaborative Filtering** menghasilkan metrik yang lebih tinggi dibandingkan dengan Content-Based Filtering, khususnya dalam hal precision dan NDCG. Hal ini menunjukkan bahwa UBCF lebih efektif dalam memberikan rekomendasi yang relevan dalam dataset ini.

Namun, kedua pendekatan memiliki kekuatan masing-masing dan dapat dikombinasikan (misalnya dalam hybrid system) untuk menghasilkan rekomendasi yang lebih personal dan akurat di berbagai kondisi pengguna.

---

## 📊 Evaluation

Untuk mengevaluasi performa sistem rekomendasi yang dikembangkan, kami menggunakan empat metrik utama yang umum digunakan dalam evaluasi sistem rekomendasi berbasis **Top-N Recommendation**:

---

### 🔢 Metrik Evaluasi

1. **Precision\@K**
   Mengukur proporsi item yang direkomendasikan dalam Top-K yang benar-benar relevan bagi pengguna.

   $$
   \text{Precision@K} = \frac{\text{Jumlah item relevan dalam Top-K}}{K}
   $$

2. **Recall\@K**
   Mengukur proporsi item relevan yang berhasil direkomendasikan dalam Top-K dari seluruh item relevan yang tersedia.

   $$
   \text{Recall@K} = \frac{\text{Jumlah item relevan dalam Top-K}}{\text{Total item relevan}}
   $$

3. **MAP\@K (Mean Average Precision)**
   Mengukur rata-rata precision pada setiap posisi item relevan dalam daftar rekomendasi Top-K. MAP memperhitungkan urutan (ranking) item yang direkomendasikan.

4. **NDCG\@K (Normalized Discounted Cumulative Gain)**
   Mengukur relevansi item dengan memperhatikan posisi item pada daftar rekomendasi. Relevansi yang muncul di urutan lebih atas dihargai lebih tinggi.

   $$
   \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
   $$

---

### 📈 Hasil Evaluasi

Berikut adalah hasil evaluasi dari kedua metode yang digunakan:

| Metode   | Precision\@10 | Recall\@10 | MAP\@10 | NDCG\@10 |
| -------- | ------------- | ---------- | ------- | -------- |
| **CBF**  | 0.0014        | 0.0039     | 0.0017  | 0.0031   |
| **UBCF** | 0.0061        | 0.0050     | 0.0043  | 0.0091   |

---

### 📌 Interpretasi Hasil

* **User-Based Collaborative Filtering (UBCF)** menghasilkan skor evaluasi yang lebih tinggi dibandingkan Content-Based Filtering (CBF) pada semua metrik, terutama pada MAP dan NDCG.
* Hal ini menunjukkan bahwa UBCF lebih mampu menangkap relevansi item berdasarkan preferensi komunitas pengguna.
* **CBF**, meskipun lebih lemah dalam hal metrik, masih berguna untuk pengguna baru yang belum memiliki banyak interaksi.

---

### ✅ Kesimpulan Proyek

* Dua pendekatan rekomendasi telah berhasil dibangun dan dievaluasi menggunakan metrik yang sesuai.
* **UBCF** terbukti lebih unggul pada dataset ini, namun pendekatan **CBF** tetap berguna sebagai alternatif atau komponen hybrid.
* Sistem rekomendasi ini dapat dikembangkan lebih lanjut dengan menambahkan:

  * Hybrid model (menggabungkan CBF dan UBCF)
  * Informasi tambahan (genre, tahun, dll.)
  * Deep learning untuk embedding item dan user

---