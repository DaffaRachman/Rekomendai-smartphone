# Proyek Pertama Machine Learning Terapan | smartphone recommendation system

###### Disusun oleh : Muhammad Daffa Rachman

Proyek ini membangun model *Sistem rekomendasi* yang dapat memberikan rekomendasi smartphone

## 1. Project Domain

Pada era digital saat ini, smartphone telah menjadi bagian penting dari kehidupan sehari-hari. Dengan berbagai pilihan yang tersedia di pasar, pemilihan smartphone yang sesuai dengan kebutuhan pribadi menjadi semakin kompleks. Smartphone tidak hanya berfungsi sebagai alat komunikasi, tetapi juga sebagai perangkat multifungsi untuk hiburan, pekerjaan, dan gaya hidup. Dalam situasi ini, pengguna cenderung mencari rekomendasi yang dapat membantu mereka membuat keputusan pembelian yang tepat.

Seiring dengan berkembangnya teknologi, sistem rekomendasi berbasis data mulai mendapatkan perhatian besar dalam memberikan solusi kepada pengguna untuk menemukan smartphone yang sesuai dengan preferensi dan anggaran mereka. Sistem rekomendasi ini dapat dibangun dengan menggunakan berbagai pendekatan, termasuk collaborative filtering, content-based filtering, dan hybrid systems. Salah satu teknik yang paling umum digunakan dalam sistem rekomendasi adalah content-based filtering menggunakan informasi tentang atribut atau fitur dari smartphone, seperti RAM, processor, harga, dan rating, dll, untuk memberikan rekomendasi berdasarkan kesamaan fitur antara smartphone yang tersedia di pasar.

### Bagaimana Masalah tersebut Diselesaikan?

#### 1. Pemanfaatan Teknologi Sistem Rekomendasi Smartphone
Untuk mengatasi tantangan dalam memilih smartphone yang tepat di pasar yang sangat kompetitif, digunakan sistem rekomendasi berbasis algoritma machine learning seperti Content-Based Filtering dan Cosine Similarity. Dalam penelitian ini, algoritma Cosine Similarity digunakan untuk mengukur kesamaan antara berbagai smartphone berdasarkan fitur-fitur teknis seperti processor, RAM, battery, display, dan camera, serta fitur numerik seperti price dan rating. Sistem rekomendasi ini memungkinkan pengguna untuk mendapatkan rekomendasi yang lebih relevan dengan smartphone yang mereka minati berdasarkan spesifikasi dan preferensi serupa.

#### Analisis Fitur Smartphone dan Rekomendasi Berdasarkan Similarity
Dengan memanfaatkan dataset smartphone yang mencakup berbagai spesifikasi perangkat seperti processor (jenis dan kecepatan), RAM (kapasitas), battery (kapasitas dan jenis pengisian daya), display (ukuran dan resolusi), dan camera (spesifikasi kamera), algoritma Content-Based Filtering dapat memberikan rekomendasi berdasarkan kesamaan fitur. Fitur-fitur ini diubah menjadi representasi numerik menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) untuk fitur berbasis teks dan StandardScaler untuk fitur numerik. Kemudian, Cosine Similarity dihitung antara smartphone dalam dataset untuk memberikan rekomendasi yang paling mirip.

#### 3. Dampak yang Diharapkan
Pendekatan berbasis Content-Based Filtering dan Cosine Similarity diharapkan dapat meningkatkan pengalaman pengguna dalam memilih smartphone yang sesuai dengan preferensi pribadi mereka. Dengan memberikan rekomendasi yang relevan berdasarkan fitur-fitur utama dan harga, pengguna dapat lebih mudah menemukan smartphone yang memenuhi kebutuhan mereka tanpa merasa kewalahan dengan banyaknya pilihan yang tersedia di pasar. Selain itu, peningkatan akurasi rekomendasi harga dan rating diharapkan dapat mengurangi ketidakpuasan pengguna terhadap rekomendasi yang diberikan, yang sering kali didorong oleh perbedaan harga yang signifikan. Dengan ini, diharapkan sistem rekomendasi smartphone ini dapat membantu meningkatkan kepuasan pengguna dan membantu mereka membuat keputusan pembelian yang lebih tepat 【3】.


## 2. Business Understanding

Pengembangan sistem rekomendasi smartphone berbasis machine learning memiliki potensi besar untuk meningkatkan pengalaman pengguna dalam memilih smartphone yang sesuai dengan kebutuhan pribadi mereka. Dengan berbagai pilihan smartphone yang sangat beragam di pasar, model ini dapat membantu konsumen dalam menemukan perangkat yang memenuhi preferensi teknis, harga, dan rating mereka secara akurat. Sistem ini, yang memanfaatkan algoritma seperti Content-Based Filtering dan Cosine Similarity, memungkinkan pengguna untuk mendapatkan rekomendasi yang lebih relevan berdasarkan fitur-fitur teknis dan harga dari smartphone.

### Problem Statements
Berdasarkan latar belakang tersebut, masalah yang dapat diselesaikan dalam proyek ini adalah:
- Bagaimana cara membangun sistem rekomendasi berbasis machine learning yang dapat memberikan rekomendasi smartphone yang relevan?
- Bagaimana sistem dapat merekomendasikan smartphone secara otomatis?

### Goals
Tujuan dari proyek ini meliputi:
- Membangun model rekomendasi sistem content-based filtering.
- Membuat rekomendasi smartphone.

## Data Understanding
### EDA - Deskripsi Variabel
**Informasi Datasets**


| Jenis | Keterangan |
| ------ | ------ |
| Title | _Smartphone Specifications and Prices in India_ |
| Source | [Kaggle](https://www.kaggle.com/datasets/shrutiambekar/smartphone-specifications-and-prices-in-india) |
| Maintainer | [shruti ambekar](https://www.kaggle.com/shrutiambekar) |
| License | Data files © Original Authors |
| Visibility | Publik |
| Tags | _Data Visualization, Data Cleaning, Exploratory Analysis, Feature Engineering, Recomender Systems_ |
| View | 9572 |


Tabel 1. Informasi Dataset

![data info](https://github.com/user-attachments/assets/35fc3de0-4b35-4ccb-ab4c-0424c643f044)

Dilihat dari _Tabel 1. Informasi Dataset_ dataset ini berisi informasi sebagai berikut ini : 
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 1020 sample dengan 11 fitur.
- Dataset memiliki 1 fitur bertipe float64, dan 10 fitur bertipe object.

### Variable - variable pada dataset
- model - Nama model mobile phone
- price - harga mobie phone dalam currency India
- rating - rata - rata penilaian yang diberikan oleh customer yang sudah membeli dan sudah menggunakan
- sim - Jumlah kartu SIM yang disupport (single sim, dual sim, dll)
- processor - nama model processor (Qualcom Snapdragon, Apple Bionic, dll)
- ram - jumlah ram dalam GB
- battery - Jumlah kapasistas baterai dalam mAh
- display - Ukuran layar dalam inch dan resolusi layar
- camera - Jumlah kamera dan spesifikasi kamera
- card - tipe dan jumlah maksimal memory card yang dapat digunakan

### Pengecekan Data Duplikat dan Missing Value
-	Data Duplikat

![Data Duplikat](https://github.com/user-attachments/assets/2ec40689-22ae-4ca5-b854-6c7e4de98d86).

Gambar 1. Data Duplikat.

Pada gambar tersebut, menjelaskan bahwa pada dataset ini tidak memiliki data yang terduplikat.

-	Missing Value

![Missing value](https://github.com/user-attachments/assets/6a707b44-c025-45b4-b1cc-865cde8a983e).


Gambar 2. Missing Value

Pada dataset anime terdapat beberapa kolom yang mempunyai data missing value, yakni pada kolom rating sebanyak 141 entri, camera sebanyak 1 entri, card sebanyak 1 entri, dan os sebanyak 17 entri.

### Pengecekan Value Unik yang Ada Pada Dataset

![value unik](https://github.com/user-attachments/assets/c0cdb42e-cf48-4e19-9c9b-90a2ee1a124b)

Gambar 3. Value Unik

Berdasarkan gambar tersebut, berikut adalah penjelasan singkat tentang nilai unik (unique values) untuk fitur-fitur dalam dataset tersebut:

- 1020 nilai unik pada kolom model, ini menunjukkan bahwa ada 1020 model perangkat yang terdaftar dalam dataset.

- 412 nilai unik pada kolom price, ini menunjukkan bahwa harga perangkat bervariasi di antara 412 nilai yang berbeda.

- 369 nilai unik pada kolom display, ini menunjukkan variasi pada tampilan perangkat dan ukuran layar.

- processor: 298 nilai unik – ini menunjukkan berbagai jenis prosesor yang digunakan dalam perangkat.

- camera: 285 nilai unik – ini mencakup variasi jenis atau konfigurasi kamera dalam perangkat.

- battery: 256 nilai unik – menunjukkan berapa banyak variasi dalam kapasitas baterai perangkat.

- card: 63 nilai unik – ini mengacu pada jumlah dalam GB kartu memori.

- ram: 58 nilai unik – mengindikasikan jumlah variasi dalam kapasitas RAM.

- os: 48 nilai unik – ini menunjukkan jumlah variasi sistem operasi yang digunakan oleh perangkat.

- rating: 30 nilai unik – ini menunjukkan variasi pada rating perangkat.

- sim: 28 nilai unik – ini menunjukkan jumlah variasi dalam jumlah SIM yang didukung perangkat.

Nilai unik ini mencerminkan variasi atau kategori yang ada pada setiap fitur, yang membantu dalam memahami pola atau karakteristik data yang terkandung di dalamnya.

### EDA - Univariate Analysis

![infoDataset](https://github.com/user-attachments/assets/b18bc4a2-eb4e-43e0-9a5f-469445c11fa2)

Gambar 4. Informasi Dataset

Gambar 4 merupakan informasi mengenai dataset yang digunakan :
Tabel diatas merupakan informasi mengenai dataset yang sigunakan :
- Harga smartphone bervariasi sangat besar, dari harga terendah ₹99 hingga harga tertinggi ₹65,000. Sebagian besar smartphone berada di kisaran harga menengah, dengan harga rata-rata sekitar ₹31,371.

- Rating smartphone memiliki distribusi yang lebih terkonsentrasi, dengan rata-rata 78.26 dan mayoritas smartphone mendapatkan rating yang lebih baik, di atas 75.

![outlier](https://github.com/user-attachments/assets/3d73399b-6bf2-4942-86d8-c93629982e8f)

Gambar 5. Pengecekan outlier pada dataset

Gambar 5 merupakan visualisasi exploratory data analysis dari pengecekan outlier pada dataset yang digunakan. Berdasarkan boxplot yang ditampilkan, berikut adalah fitur yang menunjukkan adanya outlier yang jelas pada dataset ini:

- Terdapat banyak outliers yang terlihat pada harga smartphone. Ini menunjukkan bahwa ada beberapa smartphone dengan harga yang sangat tinggi (di atas ₹200,000), yang kemungkinan merupakan perangkat premium atau flagship.

Outliers pada kolom harga (misalnya, harga smartphone yang sangat tinggi atau rendah) bisa menyebabkan kesalahan perhitungan kesamaan antara item. Misalnya, sebuah smartphone dengan harga sangat tinggi akan memiliki kesamaan yang lebih rendah dengan smartphone yang memiliki harga sangat rendah. Karena akan menggunakan teknik seperti Cosine Similarity untuk mengukur kemiripan antara item, fitur yang memiliki rentang nilai yang sangat besar (seperti harga) bisa mendominasi perhitungan kesamaan, sehingga menyebabkan rekomendasi yang tidak sesuai. Untuk itu, perlu menggunakan metode seperti IQR (Interquartile Range) untuk menangani outlier dan memastikan data yang digunakan untuk analisis lebih konsisten dan valid.

![sebaran](https://github.com/user-attachments/assets/17ef1fcc-cd7f-4a37-8134-e8e15b39517c)

Gambar 6. Persebaran data

Gambar 6 merupakan visualisasi exploratory data analysis dari persebaran data pada dataset yang digunakan pada project ini adalah weather type. 
Berdasarkan visualisasi distribusi pada gambar, berikut adalah penjelasan untuk beberapa fitur yang ada dalam dataset :

- Rating Distribution:
Distribusi rating menunjukkan bahwa mayoritas perangkat memiliki rating yang lebih tinggi, dengan puncak pada nilai sekitar 80 hingga 85. Grafik ini menunjukkan distribusi yang lebih merata dan sedikit condong ke angka yang lebih tinggi, menunjukkan bahwa kebanyakan perangkat memiliki rating yang relatif baik. Kurva distribusi mendukung pola ini, menandakan bahwa data mengikuti pola distribusi normal (meskipun ada beberapa fluktuasi pada angka-angka tertentu).

- Price Distribution:
Distribusi harga menunjukkan adanya sebaran harga yang sangat miring ke kanan, dengan mayoritas perangkat memiliki harga yang sangat rendah, namun ada beberapa perangkat dengan harga yang sangat tinggi. Grafik ini memperlihatkan bahwa sebagian besar harga perangkat terpusat di bagian bawah rentang harga, namun ada sedikit data dengan harga yang sangat tinggi, yang membuat distribusi ini tampak sangat condong ke kiri. Kurva distribusi yang sangat tinggi pada harga rendah ini menunjukkan bahwa sebagian besar perangkat dalam dataset mungkin berada dalam kategori harga rendah.

## Data Preparation
Berikut merupakan data preparation yang diterapkan pada project ini :

1. Pembersihan data Null
Membersihkan nilai Missing value dengan cara dropna
2. Penanganan Outlier Menggunakan IQR
Penggunaan metode Interquartile Range (IQR) adalah salah satu cara yang efektif untuk menangani data outlier dalam analisis statistik. IQR mengukur rentang antara kuartil pertama (Q1) dan kuartil ketiga (Q3) dalam sebuah dataset, yang mencakup 50% data tengah. Outlier dapat diidentifikasi dengan memeriksa nilai yang berada di luar rentang yang ditentukan oleh IQR.
3. Mengkoversi harga dari Rupee ke IDR
Karena data yang digunakan bersala dari negara Indonesia, maka akan dikonversi ke mataung IDR 
4. Pembersihan teks dan pengolahan fitur teks menggunakan TF-IDF
Dalam hal fitur teks seperti processor, ram, battery, display, dan camera, Anda menggunakan TfidfVectorizer untuk mengubah teks menjadi representasi numerik. Proses ini akan mengonversi teks dalam kolom-kolom tersebut menjadi vektor numerik berdasarkan frekuensi kata.
5. Encoding Data
Untuk memastikan bahwa data numerik seperti harga dan rating berada dalam skala yang seragam, Anda menggunakan StandardScaler untuk menstandarisasi nilai-nilai ini.
6. Penggabungan fitur Numerik dan TF-IDF
Menggabungkan hasil dari representasi TF-IDF dan fitur numerik yang telah distandarisasi 

## Modeling
Pada project ini menggunakan Content-Based Filtering dengan Cosine Similarity

Content-Based Filtering adalah teknik rekomendasi yang berfokus pada fitur atau atribut dari item yang sedang dianalisis (dalam hal ini, smartphone). Dengan metode ini, setiap smartphone dianalisis berdasarkan fitur-fitur spesifik seperti harga, rating, processor, ram, battery, display, dan camera. Rekomendasi kemudian diberikan berdasarkan kesamaan fitur antara smartphone yang dipilih dengan smartphone lainnya.

#### Cara Kerja pada Dataset Smartphone
Pada metode Content-Based Filtering, smartphone dibandingkan dengan menggunakan cosine similarity untuk mengukur kemiripan antara smartphone berdasarkan fitur-fitur yang ada. Setiap smartphone diwakili sebagai vektor dalam ruang fitur, dan rekomendasi diberikan berdasarkan kemiripan vektor ini. Cosine Similarity menghitung sudut antara dua vektor dalam ruang fitur. Vektor yang lebih mirip (sudut lebih kecil) akan memiliki nilai similarity lebih tinggi.

#### Langkah Kerja:
Menghitung Cosine Similarity:
- Setiap smartphone direpresentasikan sebagai vektor fitur.
- Cosine similarity dihitung antara vektor smartphone yang dipilih dan semua smartphone lain di dalam dataset.
- Setelah menghitung similarity, smartphone akan diurutkan berdasarkan tingkat kesamaan (dari yang paling mirip ke yang paling tidak mirip).
- Rekomendasi diberikan berdasarkan nilai similarity tertinggi (dengan pengecualian model itu sendiri).

#### Parameter yang Digunakan:
- features: Matriks fitur yang digabungkan, yang mencakup representasi numerik dari harga, rating, dan fitur teks seperti processor, ram, display.
- cosine_similarity(features, features): Menghitung nilai kesamaan antara semua smartphone dalam dataset.
- model_index: Indeks model yang dipilih sebagai referensi untuk rekomendasi.
- top_n=5: Menentukan jumlah smartphone yang akan direkomendasikan berdasarkan kesamaan tertinggi.

### Hasil rekomendasi
![Rekomendasi](https://github.com/user-attachments/assets/bb809853-461a-460f-9f5b-d158b8e0e757)

Gambar 7. Hasil rekomendasi

Dari gambar tersebut, sistem berhasil menghasilkan rekomendasi top 5 smartphone

## Evaluation
Menggunakan matriks evaluasi MAE

![MAE](https://github.com/user-attachments/assets/17c15219-a086-47e6-8639-3cc740265a87)

Nilai MAE untuk harga menunjukkan kesalahan rata-rata sebesar 1,090,189.8 IDR antara harga asli dan harga yang diprediksi oleh sistem rekomendasi. Nilai MAE untuk rating adalah 0.0, yang menunjukkan bahwa rekomendasi yang diberikan memiliki rating yang persis sama dengan smartphone yang diberikan sebagai input. Model memberikan rekomendasi yang baik dalam hal rating, tetapi kurang akurat dalam memprediksi harga. Hal ini menunjukkan bahwa content-based filtering dengan cosine similarity lebih fokus pada kesamaan rating, namun kurang mempertimbangkan variasi harga.

## Kesimpulan
Content based filtering dengan  cosine similiarity cukup baik dalam sistem rekomendasi dan dapat memberikan hasil rekomendasi. Dari hasil yang tertera, terlihat bahwa MAE untuk harga cukup tinggi, yakni sekitar 1.09 juta IDR, yang menunjukkan adanya perbedaan harga yang signifikan antar smartphone dalam dataset. Ini kemungkinan disebabkan oleh perbedaan harga yang besar antara smartphone flagship dan tipe lain. 

## Kesimpulan Dampak Model Terhadap Business Understanding

## 1. Menjawab Problem Statements
Model yang dievaluasi telah berhasil menjawab sebagian besar problem statements.

- System dapat memberikan hasil rekomendasi, berdasarkan MAE system dapat cukup baik merekomendasikan smartphone untuk pengguna

- System telah berhasil secara otomatis menampilkan top 5 rekomendasi untuk pengguna 

## 2. Mencapai Goals
Tujuan dari proyek ini berhasil dicapai.

- Content-based filtering dengan consine similiarity telah berhasil membangun system rekomendasi
- system sudah berhasil menampilkan rekomendasi smartphone
