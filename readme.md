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

Dari hasil eksplorasi awal, ditemukan beberapa masalah pada data. Pada data buku, terdapat missing value sebanyak 2 entri pada kolom `Book-Author` dan `Publisher` akan tetapi tidak memiliki nilai duplikat. Sedangkan pada data pengguna, kolom `Age` memiliki missing value yang cukup signifikan, yakni sekitar 110.762 dari total 278.858 pengguna (sekitar 40%) walau demikian data pengguna tidak memiliki duplikat. Namun, data rating tidak memiliki missing value maupun duplikat. Selain itu, tidak ditemukan entri duplikat pada ketiga file utama.

📦 **Ukuran data**:

* Jumlah pengguna: 278.858  
* Jumlah buku: 271.379 
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

## 📊 Data Preparation - Content Based Filtering

Tahapan *data preparation* sangat penting dalam proses pengembangan sistem rekomendasi, khususnya untuk pendekatan *Content-Based Filtering (CBF)*. Dalam proyek ini, data preparation dilakukan secara bertahap dan sistematis agar model dapat bekerja secara optimal dan akurat. Berikut adalah tahapan yang dilakukan:

---

### 1. **Seleksi Kolom Relevan**

Langkah pertama adalah memilih hanya kolom yang relevan dari dataset buku:

📌 **Tujuan**: Langkah pertama dalam proses data preparation adalah melakukan seleksi kolom dari dataset buku (`df_book`). Dari sekian banyak kolom yang tersedia, hanya empat kolom yang dipilih, yaitu `ISBN`, `Book-Title`, `Book-Author`, dan `Publisher`. Keempat kolom ini dianggap sebagai fitur yang paling relevan untuk membangun sistem rekomendasi berbasis konten (content-based filtering), karena berisi identitas unik buku dan informasi tekstual yang dapat digunakan untuk mengekstrak representasi konten buku. Seleksi ini juga bertujuan untuk menyederhanakan data dan mengurangi noise dari kolom-kolom lain yang tidak diperlukan dalam pemodelan.

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

Berikut versi yang lebih terstruktur dan lebih baik untuk bagian **Data Preparation** yang menjelaskan kode yang kamu berikan:

---

## 📦 Data Preparation - User Collaborative Filtering

### 1. **Filtering Rating Positif**

Langkah awal dalam tahap persiapan data adalah menyaring data rating untuk hanya menyertakan interaksi yang bermakna. Pada dataset ini, nilai rating `0` dianggap tidak merepresentasikan opini pengguna (misalnya, karena pengguna hanya melihat buku tanpa memberikan penilaian). Oleh karena itu, hanya rating dengan nilai lebih dari 0 yang dipertahankan untuk dianalisis lebih lanjut:

```python
ratings_filtered = df_rating[df_rating['Book-Rating'] > 0]
```

### 2. **Agregasi Rating Ganda**

Dalam beberapa kasus, satu pengguna dapat memberikan lebih dari satu rating untuk buku yang sama. Untuk menghindari duplikasi dalam data interaksi, dilakukan agregasi menggunakan **mean rating** per kombinasi `User-ID` dan `ISBN`.

```python
ratings_grouped = ratings_filtered.groupby(['User-ID', 'ISBN'], as_index=False)['Book-Rating'].mean()
```

### 3. **Seleksi Pengguna dan Buku Terpopuler**

Karena sistem rekomendasi berbasis memori (seperti KNN) membutuhkan matriks yang padat dan representatif, dilakukan penyaringan terhadap data berdasarkan popularitas:

* Dipilih **1.000 buku** dengan jumlah interaksi terbanyak.
* Dipilih **1.000 pengguna** dengan jumlah aktivitas (rating) terbanyak.

Langkah ini bertujuan untuk meningkatkan efisiensi dan menghindari sparsity ekstrem pada data.

```python
top_isbn = ratings_grouped['ISBN'].value_counts().head(1000).index
top_users = ratings_grouped['User-ID'].value_counts().head(1000).index
```

### 4. **Membangun Dataset Skala Kecil**

Setelah daftar pengguna dan buku terpopuler ditentukan, dibuat subset data yang hanya memuat interaksi antara pengguna dan buku yang ada dalam daftar tersebut. Dataset ini akan menjadi dasar pembentukan matriks interaksi.

```python
ratings_small = ratings_grouped[
    ratings_grouped['ISBN'].isin(top_isbn) & ratings_grouped['User-ID'].isin(top_users)
]
```

### 5. **Membangun Matriks Interaksi Pengguna-Buku**

Data kemudian diubah menjadi **user-item interaction matrix**, di mana baris mewakili pengguna dan kolom mewakili buku. Nilai dalam matriks adalah skor rating yang telah diberikan oleh pengguna. Untuk sel kosong (tidak ada interaksi), diisi dengan `0` untuk menjaga struktur data numerik.

```python
user_item_matrix = ratings_small.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
```

### 6. **Konversi ke Format Sparse Matrix**

Karena sebagian besar nilai dalam matriks interaksi adalah nol (sparse), digunakan representasi **compressed sparse row (CSR)** untuk menghemat memori dan mempercepat komputasi.

```python
matrix_sparse = csr_matrix(user_item_matrix.values)
```
---

## 🤖 Modeling

Pada tahap ini, kami membangun dan membandingkan dua pendekatan sistem rekomendasi untuk menghasilkan **Top-N Recommendation** bagi pengguna. Kedua pendekatan tersebut adalah:

1. **Content-Based Filtering (CBF)**
2. **User-Based Collaborative Filtering (UBCF)**

---

### 📌 1. Content-Based Filtering (CBF)

Content-Based Filtering adalah metode sistem rekomendasi yang merekomendasikan item berdasarkan kemiripan konten antar item itu sendiri, bukan dari preferensi pengguna lain.

Dalam proyek ini, konten buku direpresentasikan menggunakan kombinasi fitur teks seperti **judul buku**, **penulis**, dan **penerbit**. Fitur-fitur ini digabungkan menjadi satu kolom teks, lalu diubah menjadi representasi numerik menggunakan **TF-IDF (Term Frequency–Inverse Document Frequency)**. Proses ini dilakukan pada tahap *data preparation*.

#### ✅ **Modeling dengan Cosine Similarity**

Setelah mendapatkan representasi vektor dari setiap buku, kemiripan antar buku dihitung menggunakan **cosine similarity**. Cosine similarity mengukur sudut antara dua vektor dalam ruang fitur, sehingga cocok untuk mengukur kemiripan dokumen berbasis teks.

#### 🔍 **Proses Rekomendasi**

1. Untuk setiap pengguna, sistem mengambil satu buku dengan rating tertinggi (paling disukai) sebagai referensi.
2. Sistem mencari buku-buku lain yang **paling mirip** dengan buku tersebut berdasarkan konten (judul, penulis, penerbit).
3. Sistem menyajikan **Top-N buku serupa** sebagai rekomendasi.

Pendekatan ini memiliki keunggulan dalam memahami konten item, tetapi memiliki keterbatasan seperti cold-start terhadap item atau user baru tanpa riwayat.

* **Contoh Output (Top-5 Recommendation):**

Berikut adalah hasil rekomendasi menggunakan metode **Content-Based Filtering (CBF)** untuk buku *"cat & mouse (alex cross novels)"*:

```python
cb_recommend("cat & mouse (alex cross novels)", 5)
```

| Book-Title                              | Book-Author     |
| --------------------------------------- | --------------- |
| jack & jill (alex cross novels)         | james patterson |
| along came a spider (alex cross novels) | james patterson |
| roses are red (alex cross novels)       | james patterson |
| the beach house                         | james patterson |
| kiss the girls                          | james patterson |

Semua buku yang direkomendasikan berasal dari penulis yang sama dan memiliki genre/seri yang mirip, menunjukkan bahwa model berhasil menangkap kemiripan konten antar buku dengan baik.

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
  Rekomendasi top-10 buku untuk user dengan ID `82893` berdasarkan preferensi user-user serupa:

```python
ubcf_recommend(82893, 10)
```

| Book-Title                                                   | Book-Author            |
| ------------------------------------------------------------ | ---------------------- |
| seabiscuit: an american legend                               | laura hillenbrand      |
| deception point                                              | dan brown              |
| the lovely bones: a novel                                    | alice sebold           |
| the no. 1 ladies' detective agency (today show book club #1) | alexander mccall smith |
| harry potter and the sorcerer's stone (harry potter, #1)     | j. k. rowling          |
| mystic river                                                 | dennis lehane          |
| saving faith                                                 | david baldacci         |
| year of wonders                                              | geraldine brooks       |

Model ini dapat menangkap kesamaan preferensi antar pengguna dan merekomendasikan buku yang populer di kalangan pengguna dengan minat serupa.


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
