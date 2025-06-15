#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from sklearn.model_selection import train_test_split


# Mengimpor seluruh pustaka yang dibutuhkan untuk analisis data, visualisasi, pemrosesan teks, pembuatan sistem rekomendasi berbasis content dan user (collaborative filtering), serta pemisahan data latih dan uji.

# # Data Understanding

# In[4]:


df_book = pd.read_csv('BX_Books.csv', sep= ';', encoding= 'latin-1')
df_book


# Membaca dataset berisi informasi buku dari file BX_Books.csv dengan pemisah ; dan encoding 'latin-1', lalu menampilkannya dalam bentuk DataFrame df_book. Data ini akan digunakan dalam sistem rekomendasi berbasis konten.

# In[9]:


df_book.info()


# Menampilkan ringkasan struktur DataFrame df_book, termasuk jumlah entri, jumlah non-null di setiap kolom, tipe data, dan penggunaan memori. Digunakan untuk memahami kelengkapan dan tipe data pada dataset buku.

# In[5]:


df_book.isnull().sum()


# Menghitung jumlah nilai yang hilang (null) di setiap kolom pada DataFrame. Berguna untuk mengidentifikasi kolom mana yang memiliki data tidak lengkap sebelum melakukan pembersihan data.

# In[8]:


df_book.duplicated().sum()


# Menghitung jumlah baris duplikat pada DataFrame. Duplikasi bisa mengganggu proses analisis atau pelatihan model, sehingga perlu dihapus jika ditemukan.

# In[6]:


df_user = pd.read_csv('BX_Users.csv', sep= ';', encoding= 'latin-1')
df_user


# Membaca dataset berisi informasi user dari file BX_Users.csv dengan pemisah ; dan encoding 'latin-1', lalu menampilkannya dalam bentuk DataFrame df_user. Data ini akan digunakan dalam sistem rekomendasi berbasis konten.

# In[10]:


df_user.info()


# Menampilkan ringkasan struktur DataFrame df_user, termasuk jumlah entri, jumlah non-null di setiap kolom, tipe data, dan penggunaan memori. Digunakan untuk memahami kelengkapan dan tipe data pada dataset buku.

# In[11]:


df_user.isnull().sum()


# Menghitung jumlah nilai yang hilang (null) di setiap kolom pada DataFrame. Berguna untuk mengidentifikasi kolom mana yang memiliki data tidak lengkap sebelum melakukan pembersihan data.

# In[12]:


df_user.duplicated().sum()


# Menghitung jumlah baris duplikat pada DataFrame. Duplikasi bisa mengganggu proses analisis atau pelatihan model, sehingga perlu dihapus jika ditemukan.

# In[13]:


df_rating = pd.read_csv('BX-Book-Ratings.csv', sep= ';', encoding= 'latin-1')
df_rating


# Membaca dataset berisi informasi rating dari file BX-Book-Ratings.csv dengan pemisah ; dan encoding 'latin-1', lalu menampilkannya dalam bentuk DataFrame df_rating. Data ini akan digunakan dalam sistem rekomendasi berbasis konten.

# In[94]:


df_rating.info()


# Menampilkan ringkasan struktur DataFrame df_rating, termasuk jumlah entri, jumlah non-null di setiap kolom, tipe data, dan penggunaan memori. Digunakan untuk memahami kelengkapan dan tipe data pada dataset buku.

# In[95]:


df_rating.isnull().sum()


# Menghitung jumlah nilai yang hilang (null) di setiap kolom pada DataFrame. Berguna untuk mengidentifikasi kolom mana yang memiliki data tidak lengkap sebelum melakukan pembersihan data.

# In[96]:


df_rating.duplicated().sum()


# Menghitung jumlah baris duplikat pada DataFrame. Duplikasi bisa mengganggu proses analisis atau pelatihan model, sehingga perlu dihapus jika ditemukan.

# In[15]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df_rating, x='Book-Rating')
plt.title('Distribution of Book Ratings')  
plt.xlabel('Book Rating')
plt.ylabel('Count')
plt.show()


# Menampilkan grafik batang distribusi rating buku dari dataset df_rating. Grafik ini membantu memahami sebaran nilai rating (misalnya, apakah data condong ke rating rendah atau tinggi) dan mendeteksi potensi ketidakseimbangan pada data.

# In[ ]:


top_books = df_rating['ISBN'].value_counts().head(10)

top_books_df = df_book[df_book['ISBN'].isin(top_books.index)][['ISBN', 'Book-Title']]
top_books_df = top_books_df.merge(top_books.rename('Rating Count'), left_on='ISBN', right_index=True)

top_books_df


# Menyusun daftar 10 buku teratas berdasarkan jumlah rating terbanyak dari df_rating, lalu mencocokkannya dengan judul buku dari df_book. Output berupa DataFrame yang menampilkan ISBN, judul buku, dan jumlah rating yang diterima oleh masing-masing buku. Berguna untuk mengetahui buku-buku paling populer.

# In[19]:


top_users = df_rating['User-ID'].value_counts().head(10)
top_users_df = df_user[df_user['User-ID'].isin(top_users.index)][['User-ID', 'Location']]
top_users_df = top_users_df.merge(top_users.rename('Rating Count'), left_on='User-ID', right_index=True)
top_users_df


# Menampilkan 10 pengguna teratas yang memberikan rating terbanyak. Data mencakup User-ID, Location, dan jumlah rating yang diberikan. Berguna untuk mengidentifikasi pengguna paling aktif yang mungkin berpengaruh besar terhadap sistem rekomendasi.

# # Data Preperation - Content Base Filtering

# ### Feature Selection

# In[ ]:


books = df_book[['ISBN', 'Book-Title', 'Book-Author', 'Publisher']]
books


# Membuat subset DataFrame books yang hanya menyimpan kolom penting: ISBN, Book-Title, Book-Author, dan Publisher. Digunakan untuk menyederhanakan referensi buku dalam proses pemodelan dan rekomendasi.

# ### Null Cleaning

# In[24]:


books = books.dropna()

books.isnull().sum()


# Menghapus baris yang memiliki nilai kosong (NaN) pada data books, lalu mengecek ulang jumlah nilai kosong untuk memastikan data sudah bersih sebelum digunakan dalam proses rekomendasi.

# ### Lowercase and Merge Column

# In[25]:


books['Book-Author'] = books['Book-Author'].str.lower()
books['Book-Title'] = books['Book-Title'].str.lower()
books['Publisher'] = books['Publisher'].str.lower()

books['content'] = books['Book-Title'] + ' ' + books['Book-Author'] + ' ' + books['Publisher']
books


# Melakukan normalisasi teks (menjadi huruf kecil) pada kolom Book-Author, Book-Title, dan Publisher, lalu menggabungkannya menjadi kolom content sebagai fitur gabungan untuk pemodelan Content-Based Filtering (CBF).

# ### Sampling / Filtering

# In[39]:


top_isbn = df_rating['ISBN'].value_counts().head(500).index
books = books[books['ISBN'].isin(top_isbn)]
books.reset_index(drop=True, inplace=True)
books


# Memilih 500 buku paling populer berdasarkan jumlah rating tertinggi, lalu memfilter books agar hanya menyisakan buku-buku tersebut. Data di-reset index-nya agar rapi sebelum digunakan dalam model.

# ### Index Mapping

# In[40]:


indices = pd.Series(books.index, index=books['Book-Title']).drop_duplicates()
indices


# Membuat Series yang memetakan judul buku (Book-Title) ke indeks baris pada DataFrame books. Digunakan untuk lookup cepat saat melakukan rekomendasi berbasis konten. Duplikat judul dihapus agar satu judul hanya punya satu indeks.

# # Modeling and Result

# ## Model A - Content Based Filtering

# #### Training

# In[41]:


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['content'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Mengubah konten buku menjadi representasi vektor menggunakan TF-IDF, lalu menghitung kemiripan antar buku berdasarkan cosine similarity. Hasilnya disimpan dalam cosine_sim dan digunakan untuk Content-Based Filtering (CBF).

# #### Inference Top-N

# In[91]:


def cb_recommend(title: str, top_n: int = 10):
    """
    title: judul buku (case‑insensitive)
    return: list ISBN dari buku mirip (top_n)
    """
    key = title.lower().strip()
    if key not in indices:
        return []

    idx = indices[key]
    if not np.isscalar(idx):
        idx = int(idx.iloc[0])

    sims = cosine_sim[idx].A1 if hasattr(cosine_sim[idx], "A1") else cosine_sim[idx]
    rec_idx = sims.argsort()[::-1][1:top_n + 1]

    return books.loc[rec_idx, 'ISBN'].tolist()


# Fungsi cb_recommend menghasilkan rekomendasi buku berbasis kemiripan konten. Pencarian dilakukan berdasarkan judul buku (title), kemudian menggunakan skor cosine similarity untuk mengambil top_n buku paling mirip (selain dirinya sendiri). Hasilnya adalah daftar ISBN dari buku-buku tersebut.

# #### Result

# In[44]:


cb_recommend("cat & mouse (alex cross novels)", 5)


# Menjalankan fungsi rekomendasi berbasis konten (cb_recommend) dengan input judul buku "cat & mouse (alex cross novels)", dan menghasilkan 5 buku paling mirip berdasarkan kontennya (judul, penulis, dll). Hasilnya berupa daftar ISBN dari buku-buku yang direkomendasikan.

# ## Model B - User Collaborative Filtering

# #### Training

# In[51]:


ratings_filtered = df_rating[df_rating['Book-Rating'] > 0]
ratings_grouped = ratings_filtered.groupby(['User-ID', 'ISBN'], as_index=False)['Book-Rating'].mean()

top_isbn = ratings_grouped['ISBN'].value_counts().head(1000).index
top_users = ratings_grouped['User-ID'].value_counts().head(1000).index

ratings_small = ratings_grouped[
    ratings_grouped['ISBN'].isin(top_isbn) & ratings_grouped['User-ID'].isin(top_users)
]

user_item_matrix = ratings_small.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
matrix_sparse = csr_matrix(user_item_matrix.values)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(matrix_sparse)


# 1. Filter rating > 0: Hanya menyimpan interaksi pengguna dengan rating positif.
# 2. Group rating per user & buku: Menghitung rata-rata rating tiap (User-ID, ISBN).
# 3. Ambil 1000 user & ISBN teratas: Untuk mengurangi sparsity dan mempercepat pelatihan.
# 4. Buat matrix user-item: Dengan user sebagai baris, ISBN sebagai kolom, dan nilai rating.
# 5. Sparsify matrix: Mengubah matrix ke bentuk sparse agar efisien.
# 5. Latih model KNN: Menggunakan cosine similarity untuk menghitung kedekatan antar user.

# In[57]:


top_users


# Menampilkan daftar 1.000 pengguna (User-ID) yang paling banyak memberikan rating dalam dataset, digunakan untuk membatasi jumlah user dalam model UBCF agar komputasi lebih efisien. Hasilnya berupa Index dari User-ID yang sering berinteraksi dengan buku.

# #### Inference Top-N

# In[ ]:


def ubcf_recommend(user_id, n=10, k=20):
    if user_id not in user_item_matrix.index:
        return[]
    
    idx = user_item_matrix.index.get_loc(user_id)
    indices = model_knn.kneighbors(
        matrix_sparse[idx], n_neighbors=k+1
    )
    neighbor_ids = user_item_matrix.index[indices.flatten()[1:]]
    
    neighbor_ratings = user_item_matrix.loc[neighbor_ids].mean(axis=0)
    
    already_rated = user_item_matrix.loc[user_id]
    candidates = neighbor_ratings[already_rated == 0]
    
    top_isbn = candidates.sort_values(ascending=False).head(n).index
    return books[books['ISBN'].isin(top_isbn)][['Book-Title','Book-Author']]


# Fungsi ini mengembalikan rekomendasi buku untuk seorang user berdasarkan pendekatan User-Based Collaborative Filtering:
# 1. Mencari user tetangga paling mirip (berdasarkan cosine similarity) menggunakan NearestNeighbors.
# 2. Menghitung rata-rata rating dari user tetangga.
# 3. Memfilter buku yang belum dirating oleh user saat ini.
# 4. Mengembalikan n buku teratas berdasarkan estimasi rating tertinggi.
# 5. Digunakan untuk merekomendasikan buku berdasarkan kesamaan perilaku antar pengguna.

# #### Result

# In[62]:


ubcf_recommend(82893, 10)


# Menampilkan 10 rekomendasi buku untuk user dengan User-ID = 82893 menggunakan User-Based Collaborative Filtering (UBCF).
# 
# Fungsi ini memanfaatkan kemiripan antara user tersebut dengan user lain berdasarkan pola rating, lalu merekomendasikan buku yang belum pernah dirating oleh user tersebut tetapi disukai oleh user-user serupa.

# # Evaluation

# In[74]:


def evaluate_model(model_func, user_ids, ground_truth, k=10):
    prec, rec, ap, ndcg = [], [], [], []
    for u in user_ids:
        pred  = model_func(u, k)
        true  = set(ground_truth.get(u, []))
        if not true:
            continue

        hit   = [1 if p in true else 0 for p in pred]
        prec.append(sum(hit) / k)
        rec.append(sum(hit) / len(true))

        # AP
        cum, ap_u = 0, 0
        for i, h in enumerate(hit, 1):
            if h:
                cum += 1
                ap_u += cum / i
        ap.append(ap_u / min(len(true), k))

        # NDCG
        dcg = sum(h / np.log2(i+1) for i, h in enumerate(hit, 1))
        idcg = sum(1 / np.log2(i+1) for i in range(1, min(len(true), k)+1))
        ndcg.append(dcg / idcg if idcg else 0)

    return {
        "Precision@{}".format(k): np.mean(prec),
        "Recall@{}".format(k): np.mean(rec),
        "MAP@{}".format(k): np.mean(ap),
        "NDCG@{}".format(k): np.mean(ndcg),
    }


# Fungsi untuk mengevaluasi performa sistem rekomendasi menggunakan metrik top-k:
# * **Input**:
#   * `model_func`: fungsi rekomendasi (menerima `user_id` dan `k`).
#   * `user_ids`: daftar user yang dievaluasi.
#   * `ground_truth`: dict berisi item relevan dari setiap user.
#   * `k`: jumlah item teratas yang dievaluasi.
# * **Proses**:
#   * Hitung **Precision\@k**, **Recall\@k**, **Mean Average Precision (MAP\@k)**, dan **Normalized Discounted Cumulative Gain (NDCG\@k)**.
# * **Output**:
#   * Dictionary berisi skor rata-rata dari keempat metrik evaluasi di atas.
# 

# In[ ]:


# === 1) SPLIT TRAIN / TEST PER‑USER ======================================
user_groups = df_rating.groupby('User-ID')
train_list, test_ground_truth = [], {}

for uid, grp in user_groups:
    if len(grp) < 5:             
        continue
    tr, te = train_test_split(grp, test_size=0.2, random_state=42)
    train_list.append(tr)
    test_ground_truth[uid] = te['ISBN'].tolist()     

df_train = pd.concat(train_list, ignore_index=True)
print("Train shape :", df_train.shape)
print("#Users in GT :", len(test_ground_truth))


# Membagi data rating menjadi data latih dan uji per pengguna:
# 1. Hanya user yang memiliki minimal 5 rating yang disertakan.
# 2. 80% rating digunakan untuk latih (df_train), 20% disimpan sebagai ground truth (test_ground_truth) untuk evaluasi model.
# 3. Output: menampilkan ukuran data latih dan jumlah user yang memiliki data uji.

# In[89]:


def cbf_wrapper(user_id: str, top_n: int = 10):
    """
    Mengembalikan daftar ISBN hasil CBF untuk satu user.
    - Mencari buku dengan rating tertinggi user pada df_train
    - Mencari judul buku itu di df_book  ➜  query ke cb_recommend()
    """
    user_ratings = df_train[df_train['User-ID'] == user_id]
    if user_ratings.empty:
        return []                     

    best_isbn = (
        user_ratings
        .sort_values('Book-Rating', ascending=False)
        .iloc[0]['ISBN']
    )

    row = books[books['ISBN'] == best_isbn]
    if row.empty:
        return []                   

    title = row.iloc[0]['Book-Title']
    return cb_recommend(title, top_n)  


# Fungsi wrapper untuk Content-Based Filtering (CBF) berbasis buku favorit pengguna:
# 1. Untuk setiap user_id, ambil 1 buku dengan rating tertinggi pada df_train.
# 2. Ambil judulnya dari books, lalu gunakan sebagai query ke fungsi cb_recommend().
# 3. Mengembalikan daftar rekomendasi top_n ISBN.
# 4. Jika user belum memberi rating atau buku tidak ditemukan di metadata, hasilkan list kosong.

# In[78]:


# === 2) ----- RE‑BUILD USER‑BASED KNN PADA df_train ----- ================
user_counts = df_train['User-ID'].value_counts()
item_counts = df_train['ISBN'].value_counts()

min_user_ratings = 10
min_item_ratings = 20

filtered_train = df_train[
    df_train['User-ID'].isin(user_counts[user_counts >= min_user_ratings].index) &
    df_train['ISBN'].isin(item_counts[item_counts >= min_item_ratings].index)
]

print(f"Filtered Train shape : {filtered_train.shape}")

R = filtered_train.groupby(['User-ID', 'ISBN'], as_index=False)['Book-Rating'].mean()
user_item = R.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

matrix_sparse = csr_matrix(user_item.values)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(matrix_sparse)


def ubcf_recommend(user_id, top_n=10, k=20):
    uid = user_item.index.intersection([user_id])
    if uid.empty:
        return []
    row = user_item.index.get_loc(uid[0])
    _, idxs = model_knn.kneighbors(matrix_sparse[row], n_neighbors=k+1)
    neigh_ids = user_item.index[idxs.flatten()[1:]]  # drop self

    scores = user_item.loc[neigh_ids].mean(axis=0)
    already = user_item.loc[uid[0]] > 0
    recs = scores[~already].sort_values(ascending=False).head(top_n).index.tolist()
    return recs    


# Membangun kembali model **User-Based Collaborative Filtering (UBCF)** dengan KNN berdasarkan data latih yang telah difilter:
# 
# * **Filter** user yang memberi ≥ 10 rating dan item yang mendapat ≥ 20 rating.
# * Bentuk **user-item matrix** dari rating rata-rata, lalu ubah ke bentuk sparse matrix.
# * Latih model **KNN** berbasis cosine similarity dengan `NearestNeighbors`.
# 
# **Fungsi `ubcf_recommend(user_id, top_n=10, k=20)`**:
# 
# * Input: `user_id`, jumlah rekomendasi `top_n`, dan jumlah tetangga `k`.
# * Output: daftar ISBN hasil rekomendasi.
# * Proses:
# 
#   1. Ambil tetangga terdekat user dengan KNN.
#   2. Hitung skor rata-rata dari item yang disukai tetangga.
#   3. Buang item yang sudah pernah dirating user.
#   4. Kembalikan `top_n` item dengan skor tertinggi.
# 

# In[92]:


users_eval = list(test_ground_truth.keys())

cbf_eval  = evaluate_model(cbf_wrapper,
                           users_eval,
                           test_ground_truth,
                           k=10)

ubcf_eval = evaluate_model(lambda u, top_n: ubcf_recommend(u, top_n),
                           users_eval,
                           test_ground_truth,
                           k=10)

print("CBF  :", cbf_eval)
print("UBCF :", ubcf_eval)


# Mengevaluasi dan membandingkan performa dua metode rekomendasi:
# 1. CBF (Content-Based Filtering) dengan cbf_wrapper().
# 2. UBCF (User-Based Collaborative Filtering) dengan ubcf_recommend().
# 3. Evaluasi dilakukan terhadap top-10 rekomendasi untuk setiap user di test_ground_truth.
# 4. Menggunakan metrik: Precision@10, Recall@10, MAP@10, dan NDCG@10.
# 5. Hasil evaluasi ditampilkan di konsol.
