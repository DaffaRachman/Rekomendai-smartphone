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
- Bagaimana cara membangun sistem rekomendasi berbasis machine learning yang dapat memberikan rekomendasi smartphone yang relevan, berdasarkan kesamaan harga dan rating?
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

![data info](https://github.com/user-attachments/assets/f8d6a72f-61a5-4980-b65a-e15cb18e7672)



Dilihat dari _Tabel 1. Informasi Dataset_ dataset ini berisi informasi sebagai berikut ini : 
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 13200 sample dengan 11 fitur.
- Dataset memiliki 5 fitur bertipe float64, 2 fitur bertipe o=int64 dan 4 fitur bertipe object.

### Variable - variable pada dataset
- Temperature (numeric): Suhu dalam derajat Celsius, dengan rentang dari suhu sangat dingin hingga sangat panas.
- Humidity (numeric): Persentase kelembaban, termasuk nilai di atas 100% untuk memperkenalkan nilai pencilan (outlier).
- Wind Speed (numeric): Kecepatan angin dalam kilometer per jam.
- Precipitation (%) (numeric): Persentase curah hujan, termasuk nilai outlier.
- Cloud Cover (categorical): Deskripsi tutupan awan.
- Atmospheric Pressure (numeric): Tekanan atmosfer dalam hPa, dengan rentang yang luas.
- UV Index (numeric): Indeks UV, menunjukkan kekuatan radiasi ultraviolet.
- Season (categorical): Musim saat data dicatat.
- Visibility (km) (numeric): Jarak pandang dalam kilometer, termasuk nilai yang sangat rendah atau sangat tinggi.
- Location (categorical): Tipe lokasi tempat data dicatat.
- Weather Type (categorical): Variabel target untuk klasifikasi, yang menunjukkan tipe cuaca.

### Pengecekan Data Duplikat dan Missing Value
-	Data Duplikat
  

![Data Duplikat](https://github.com/user-attachments/assets/37a175dc-0bb1-4f2b-adc3-62633b70910b).

Gambar 1. Data Duplikat.

Pada gambar tersebut, menjelaskan bahwa pada dataset ini tidak memiliki data yang terduplikat.

-	Missing Value
  
![Missing Value](https://github.com/user-attachments/assets/d26dab42-0e22-46db-a4c1-12d9c9a29c09).

Gambar 2. Missing Value

Pada gambar tersebut, menjelaskan bahwa pada dataset ini tidak memiliki Missing value.

### Pengecekan Value Unik yang Ada Pada Dataset

![Value Unik](https://github.com/user-attachments/assets/6d5c13c1-3b5f-46a3-a3da-b1d4bea41f92).

Gambar 3. Value Unik

Berdasarkan gambar tersebut, berikut adalah penjelasan singkat tentang nilai unik (unique values) untuk fitur-fitur dalam dataset tersebut:

- Atmospheric Pressure (5456 unique values): Menunjukkan variasi nilai tekanan atmosfer, dengan 5456 nilai unik yang mencerminkan fluktuasi tekanan udara dalam dataset.

- Temperature (126 unique values): Menampilkan variasi suhu yang tercatat, dengan 126 nilai unik yang menggambarkan rentang suhu yang cukup luas.

- Precipitation (110 unique values): Mengindikasikan variasi curah hujan, dengan 110 nilai unik yang menunjukkan perubahan tingkat presipitasi sepanjang waktu.

- Wind Speed (97 unique values): Menunjukkan kecepatan angin, dengan 97 nilai unik yang mencerminkan perbedaan dalam kecepatan angin yang tercatat.

- Humidity (90 unique values): Menampilkan variasi kelembaban udara, dengan 90 nilai unik yang menunjukkan fluktuasi kelembaban di lokasi tertentu.

- Visibility (41 unique values): Mengindikasikan jarak pandang yang tercatat, dengan 41 nilai unik yang menggambarkan berbagai kondisi visibilitas.

- UV Index (15 unique values): Menunjukkan nilai indeks UV yang berkisar dari 1 hingga 15, yang mencerminkan tingkat paparan sinar ultraviolet.

-- Cloud Cover (4 unique values): Mengindikasikan tingkat penutupan awan, dengan 4 nilai unik yang menggambarkan kategori ketebalan atau intensitas awan.

- Season (4 unique values): Mengacu pada musim, dengan 4 nilai unik yang merepresentasikan musim-musim tertentu dalam set data.

- Weather Type (4 unique values): Menyatakan jenis cuaca yang tercatat, dengan 4 nilai unik yang menggambarkan kondisi cuaca seperti cerah, hujan, berawan, dll.

- Location (3 unique values): Mengindikasikan lokasi yang tercatat dalam dataset, dengan 3 nilai unik yang menunjukkan lokasi berbeda di mana data tersebut dikumpulkan.

Nilai unik ini mencerminkan variasi atau kategori yang ada pada setiap fitur, yang membantu dalam memahami pola atau karakteristik data yang terkandung di dalamnya.

### EDA - Univariate Analysis

![EDA -univariate](https://github.com/user-attachments/assets/a16f241c-5bf7-481e-981a-4a12de5b5b95)

Gambar 4. Informasi Dataset

Gambar 4 merupakan informasi mengenai dataset yang digunakan :
Tabel diatas merupakan informasi mengenai dataset yang sigunakan :
- Kolom Temperature mencatat suhu dalam derajat Celsius. Rata-rata suhu adalah 19,13°C, dengan nilai minimum mencapai -25°C yang menunjukkan adanya suhu ekstrem yang mungkin disebabkan oleh kesalahan data atau kondisi yang sangat jarang. Nilai maksimum tercatat mencapai 109°C, yang merupakan nilai ekstrem dan perlu diperiksa lebih lanjut karena suhu tersebut sangat tinggi untuk kondisi normal. Median suhu berada pada angka 21°C, menunjukkan sebagian besar data berada pada kisaran suhu yang lebih normal.

- Kolom Humidity mencatat persentase kelembapan udara. Rata-rata kelembapan adalah 68,71%, dengan nilai minimum 20% dan maksimum 109%, yang menunjukkan adanya outlier. Nilai kelembapan di atas 100% bisa menunjukkan adanya kesalahan pengukuran atau data yang tidak konsisten. Median kelembapan berada pada 70%, menunjukkan bahwa setengah dari data berada pada tingkat kelembapan di bawah 70%.

- Kecepatan angin tercatat dengan rata-rata 9,83 km/jam. Nilai minimum menunjukkan kecepatan angin yang sangat rendah (0 km/jam), sedangkan nilai maksimum menunjukkan kecepatan angin yang tidak realistis (48,5 km/jam). Rata-rata kecepatan angin cukup moderat, namun data ini juga memiliki outlier yang perlu diperhatikan lebih lanjut.
- Kolom Precipitation (%) mengukur persentase curah hujan. Nilai rata-rata curah hujan adalah 53,64%, dengan nilai minimum 0% (tidak ada hujan) dan nilai maksimum 109%, yang menandakan adanya outlier. Ini menunjukkan bahwa sebagian besar data mencatatkan hujan dengan persentase moderat, namun juga terdapat outlier yang menunjukkan curah hujan yang sangat tinggi.
- Tekanan atmosfer tercatat dengan rata-rata 1005,83 hPa, dengan nilai minimum 800,12 hPa dan maksimum 1199,21 hPa. Rata-rata menunjukkan tekanan atmosfer yang cukup stabil, tetapi rentang nilai yang luas dapat menunjukkan variasi besar dalam data, termasuk kemungkinan outlier pada nilai maksimum.
- indeks UV, yang menunjukkan intensitas radiasi ultraviolet, memiliki rata-rata 4,01, dengan nilai minimum 0 dan maksimum 14. Nilai maksimum yang tinggi perlu diperiksa lebih lanjut, karena mungkin mencerminkan kejadian ekstrim atau kesalahan data. Median indeks UV adalah 3, yang menunjukkan kondisi UV pada kisaran sedang di sebagian besar data.
- Kolom Visibility mencatatkan jarak pandang dalam kilometer, dengan nilai rata-rata 5,46 km. Nilai minimum adalah 0 km dan maksimum mencapai 20 km, menunjukkan rentang jarak pandang yang sangat bervariasi, dengan kemungkinan besar adanya outlier.

![gambar outlier](https://github.com/user-attachments/assets/ac5c168c-5b54-49a4-9a76-818910235c61).

Gambar 5. Pengecekan outlier pada dataset

Gambar 5 merupakan visualisasi exploratory data analysis dari pengecekan outlier pada dataset yang digunakan pada project ini adalah weather type. Berdasarkan boxplot yang ditampilkan, berikut adalah fitur yang menunjukkan adanya outlier yang jelas pada dataset ini:

- Terlihat jelas adanya outlier pada Temperature yang sangat tinggi (di atas 60°C) dan sangat rendah (di bawah 0°C). Nilai-nilai ini tidak realistis dan harus diperiksa lebih lanjut.

- Wind Speed (Kecepatan) angin menunjukkan adanya outlier yang cukup signifikan, dengan beberapa nilai mencapai lebih dari 30 km/jam, yang tidak lazim pada sebagian besar data.

- Terlihat outlier pada nilai UV Index yang sangat tinggi, mencapai 14. Sebagian besar data berada pada kisaran 0-7, sementara nilai lebih dari 7 adalah outlier yang perlu diperhatikan.

- Visibility (Jarak pandang) menunjukkan adanya outlier pada nilai yang sangat rendah (mungkin mendekati 0 km) dan sangat tinggi (mendekati 20 km). Ini menandakan bahwa beberapa nilai sangat jauh dari nilai-nilai lainnya.

- Atmospheric Pressure menunjukkan beberapa outlier pada nilai yang sangat rendah (di bawah 900 hPa) dan sangat tinggi (di atas 1100 hPa), yang mungkin mencerminkan kondisi cuaca yang tidak biasa atau kesalahan pengukuran.

Outlier pada fitur seperti Temperature, Wind Speed, UV Index, dan Visibility perlu dianalisis lebih lanjut, karena mereka bisa merusak distribusi data dan memengaruhi model prediksi. Untuk itu, perlu menggunakan metode seperti IQR (Interquartile Range) untuk menangani outlier dan memastikan data yang digunakan untuk analisis lebih konsisten dan valid.

![persebaran data](https://github.com/user-attachments/assets/4d527e17-515d-47d7-aa22-e8899fc6222e)

Gambar 6. Persebaran data

Gambar 6 merupakan visualisasi exploratory data analysis dari persebaran data pada dataset yang digunakan pada project ini adalah weather type. 
Berdasarkan visualisasi distribusi pada gambar, berikut adalah penjelasan untuk beberapa fitur yang ada dalam dataset :

Distribusi variabel dalam dataset menunjukkan variasi yang baik dalam kebanyakan fitur, meskipun ada beberapa outlier yang perlu ditangani, seperti pada fitur Temperature, Wind Speed, Precipitation, dan UV Index. Selain itu, terdapat ketidakseimbangan pada fitur Season Count yang sangat didominasi oleh Winter, dan Cloud Cover yang lebih sering mencatatkan kondisi berawan. Metode SMOTE akan dilakukan untuk menangani value yang tidak seimbang.

### EDA - Multivariate Analysis

![matriks korelasi](https://github.com/user-attachments/assets/ada04ab3-2ef3-4bf6-b706-0b1551337910)

Gambar 7. Analisis Multivariate Matriks korelasi

Pada Gambar 7 Analisis Multivariate, dengan menggunakan matriks korelasi. beriku adalah penjelasan dari matriks korelasi :
- Humidity dan Precipitation memiliki korelasi positif yang kuat (0.64), yang menunjukkan bahwa semakin tinggi kelembaban, semakin tinggi pula kemungkinan curah hujan.
- Wind Speed memiliki korelasi positif dengan Precipitation (0.44), yang mengindikasikan bahwa kecepatan angin mungkin sedikit berpengaruh terhadap curah hujan.
- Temperature dan UV Index memiliki korelasi positif (0.37), yang menunjukkan bahwa suhu yang lebih tinggi cenderung terkait dengan peningkatan intensitas radiasi UV.
- Visibility memiliki korelasi negatif dengan Humidity (-0.48) dan Precipitation (-0.46), yang menunjukkan bahwa pada kondisi kelembaban atau hujan yang lebih tinggi, jarak pandang cenderung berkurang.

### EDA - Multivariate Analysis

![multivariate](https://github.com/user-attachments/assets/d203fcc1-0eb0-4ce8-a5c6-dcc2e0e43857)

Gambar 8. Multivariate Analysis

Pada Gambar 8 Multivariate analysis digunakan untuk melihat hubungan setiap kolom. berikut adalah penjelasannya :
- -Beberapa kolom seperti Wind Speed dan Precipitation menunjukkan distribusi yang sangat terpusat, sedangkan variabel lain memiliki distribusi yang lebih normal.

- Beberapa variabel seperti Temperature dan Humidity atau Precipitation dan Wind Speed menunjukkan hubungan yang cukup kuat, yang dapat membantu dalam membangun model yang lebih efisien.

Dengan menggunakan Multivariate Analysis, dapat membantu mengevaluasi hubungan antar variabel dan distribusi data dalam dataset. Ini juga sangat membantu dalam mendeteksi pola dan korelasi yang bisa digunakan dalam analisis lebih lanjut atau pemodelan.

## Data Preparation
Berikut merupakan data preparation yang diterapkan pada project ini :

1. Penggunaan Metode IQR untuk menangani data outlier
Penggunaan metode Interquartile Range (IQR) adalah salah satu cara yang efektif untuk menangani data outlier dalam analisis statistik. IQR mengukur rentang antara kuartil pertama (Q1) dan kuartil ketiga (Q3) dalam sebuah dataset, yang mencakup 50% data tengah. Outlier dapat diidentifikasi dengan memeriksa nilai yang berada di luar rentang yang ditentukan oleh IQR.
2. Encoding data
Label Encoding dan One-Hot Encoding. Label Encoding diterapkan pada fitur season dan weather type, yang bersifat ordinal, untuk mewakili urutan kategori secara numerik (misalnya, musim dan jenis cuaca yang memiliki urutan tertentu). Sementara itu, One-Hot Encoding digunakan pada fitur cloud cover dan location, yang bersifat nominal, dengan mengubah setiap kategori menjadi kolom biner (0 atau 1), karena tidak ada hubungan urutan antar kategori.
3. Split Dataset
Pada tahapan ini membadi data menjadi x seagai fitur dan y seagai target. Pada projek ini menggunakan fitur weather_type sebagai target. Lalu, melakukan drop kolom pada fitur wether_type karena akan digunakan sebagai y atau target dan melakukan drop pada fitur Cloud_cover dan location karena kedua fitur tersebut sudah dilakukan encoding OnehotEncoding.
4. Penanganan Ketidakseimbangan Data dengan SMOTE
Ketidakseimbangan kelas sering menjadi masalah dalam dataset, terutama jika salah satu kelas target jauh lebih sedikit dibandingkan kelas lainnya. Dalam kasus ini, digunakan SMOTE (Synthetic Minority Over-sampling Technique) untuk menyeimbangkan data. Teknik ini menghasilkan sampel sintetik dari kelas minoritas dengan cara menginterpolasi antara contoh yang ada. Hasilnya adalah dataset yang lebih seimbang, memungkinkan model untuk mempelajari pola dari kedua kelas secara lebih efektif.

## Modeling
Pada project ini menggunakan 8 algoritma machine learning yang diantaranya sebagai berikut :

## Model yang Digunakan

### 1. **Logistic Regression**

**Logistic Regression** digunakan untuk memprediksi probabilitas kejadian tertentu (misalnya, apakah cuaca akan hujan atau tidak) berdasarkan variabel input seperti **temperature**, **humidity**, **wind speed**, dan lainnya.

#### Cara Kerja pada Dataset Cuaca
Logistic Regression memetakan data fitur yang ada ke dalam probabilitas terjadinya hujan atau tidak dengan fungsi sigmoid yang menghasilkan nilai antara 0 dan 1.

#### Parameter yang Digunakan
- **`C=1.0`**: Mengatur kekuatan regularisasi untuk menghindari overfitting.
- **`penalty='l2'`**: Menggunakan penalti L2 untuk regularisasi, yang membantu mencegah model overfit.
- **`solver='liblinear'`**: Digunakan untuk dataset kecil hingga menengah, efektif dalam mengoptimalkan fungsi logistik.

---

### 2. **K-Nearest Neighbors (KNN)**

**KNN** digunakan untuk mengklasifikasikan cuaca berdasarkan jarak terdekat dengan data historis cuaca lainnya. Dengan menghitung jarak antara data baru dan data historis, model menentukan kelas mayoritas dari `k` tetangga terdekat.

#### Cara Kerja pada Dataset Cuaca
KNN menghitung jarak antar titik data dan mengklasifikasikan cuaca berdasarkan mayoritas kelas dari tetangga terdekat.

#### Parameter yang Digunakan
- **`n_neighbors=5`**: Jumlah tetangga terdekat yang digunakan untuk menentukan prediksi.
- **`metric='euclidean'`**: Menggunakan jarak Euclidean untuk mengukur kedekatan antar data.
- **`weights='uniform'`**: Memberikan bobot yang sama untuk semua tetangga terdekat.

---

### 3. **Support Vector Machine (SVM)**

**SVM** digunakan untuk memisahkan data cuaca menjadi dua kelas (misalnya, hujan dan tidak hujan) dengan mencari hyperplane yang memisahkan kedua kelas tersebut.

#### Cara Kerja pada Dataset Cuaca
SVM mencari hyperplane yang memaksimalkan margin antara dua kelas (hujan dan tidak hujan). Dengan menggunakan kernel, SVM dapat menangani data non-linier.

#### Parameter yang Digunakan
- **`C=1.0`**: Mengontrol penalti kesalahan pada data pelatihan.
- **`kernel='rbf'`**: Menggunakan kernel Radial Basis Function (RBF) untuk menangani data non-linier.
- **`gamma='scale'`**: Mengatur gamma untuk kernel, mempengaruhi bentuk fungsi keputusan.

---

### 4. **Naive Bayes (Bernoulli)**

**Naive Bayes (Bernoulli)** digunakan untuk memprediksi cuaca berdasarkan fitur biner, seperti apakah suhu lebih tinggi dari 25°C atau tidak, dan apakah kelembaban lebih dari 80% atau tidak.

#### Cara Kerja pada Dataset Cuaca
Model ini mengasumsikan bahwa fitur-fitur yang ada saling independen dan mengikuti distribusi Bernoulli.

#### Parameter yang Digunakan
- **`alpha=1.0`**: Smoothing Laplace untuk menghindari probabilitas nol pada fitur yang tidak ada dalam data pelatihan.
- **`binarize=0.0`**: Batas binarisasi untuk fitur input.

---

### 5. **Naive Bayes (Gaussian)**

**Naive Bayes (Gaussian)** digunakan untuk memprediksi cuaca dengan mengasumsikan bahwa fitur-fitur numerik (seperti suhu dan kelembaban) mengikuti distribusi Gaussian.

#### Cara Kerja pada Dataset Cuaca
Model ini menghitung probabilitas setiap kelas berdasarkan asumsi distribusi normal dari setiap fitur cuaca.

#### Parameter yang Digunakan
- **`var_smoothing=1e-9`**: Menambahkan nilai kecil untuk menghindari pembagian dengan nol dalam perhitungan varians.

---

### 6. **Decision Tree**

**Decision Tree** digunakan untuk membagi dataset berdasarkan fitur-fitur cuaca yang paling mempengaruhi prediksi, seperti suhu, kelembaban, dan kecepatan angin.

#### Cara Kerja pada Dataset Cuaca
Model ini membuat keputusan bertingkat berdasarkan nilai fitur untuk memprediksi kelas. Setiap node dalam pohon memisahkan data berdasarkan fitur yang paling informatif.

#### Parameter yang Digunakan
- **`max_depth=5`**: Membatasi kedalaman pohon untuk mencegah overfitting.
- **`min_samples_split=10`**: Minimum jumlah sampel yang diperlukan untuk membagi sebuah node.
- **`criterion='gini'`**: Menggunakan Gini impurity sebagai kriteria untuk pemisahan.

---

### 7. **Random Forest**

**Random Forest** adalah metode ensemble yang menggunakan banyak decision tree untuk meningkatkan akurasi prediksi cuaca dan mengurangi risiko overfitting.

#### Cara Kerja pada Dataset Cuaca
Dengan membangun banyak decision tree dan menggabungkan hasilnya, Random Forest memberikan prediksi yang lebih stabil dan robust terhadap fluktuasi dalam data cuaca.

#### Parameter yang Digunakan
- **`n_estimators=100`**: Jumlah pohon dalam ensemble.
- **`max_depth=5`**: Menentukan kedalaman maksimum pohon untuk menghindari overfitting.

---

### 8. **XGBoost**

**XGBoost** adalah algoritma boosting yang digunakan untuk mengoptimalkan prediksi cuaca dengan meminimalkan kesalahan secara iteratif.

#### Cara Kerja pada Dataset Cuaca
XGBoost membangun model secara bertahap dengan mengurangi kesalahan prediksi di setiap iterasi. Model ini efektif dalam menangani dataset besar dan menangani nilai yang hilang dengan baik.

#### Parameter yang Digunakan
- **`learning_rate=0.1`**: Menentukan ukuran langkah pembelajaran.
- **`max_depth=6`**: Kedalaman maksimum pohon dalam setiap iterasi boosting.

## Evaluation
Menggunakan matriks evaluasi Accuracy	K-Fold Mean Accuracy	Std. Deviation	Precision	Recall	F1

Tabel 2. Evaluasi
Berikut adalah hasil evaluasi berbagai model prediksi cuaca berdasarkan beberapa metrik kinerja:

| Model            | Accuracy  | K-Fold Mean Accuracy | Std. Deviation | Precision | Recall  | F1      |
|------------------|-----------|----------------------|----------------|-----------|---------|---------|
| XGBoost          | 92.31%    | 91.15%               | 0.745945       | 0.923285  | 0.923082 | 0.923047 |
| Random Forest    | 91.82%    | 91.46%               | 0.978454       | 0.918535  | 0.918217 | 0.918300 |
| Decision Tree    | 90.91%    | 90.45%               | 0.970170       | 0.909134  | 0.909252 | 0.909131 |
| KNeighbors       | 89.73%    | 88.37%               | 0.977537       | 0.898640  | 0.897037 | 0.897472 |
| Logistic Regresion | 86.25%  | 84.04%               | 0.886568       | 0.864687  | 0.862150 | 0.861645 |
| SVM              | 83.94%    | 82.50%               | 0.914010       | 0.845519  | 0.838243 | 0.838937 |
| GaussianNB       | 81.06%    | 81.08%               | 0.895425       | 0.828575  | 0.808969 | 0.809858 |
| BernoulliNB      | 66.89%    | 66.39%               | 0.804032       | 0.705650  | 0.668794 | 0.680106 |

## Kesimpulan
Model XGBoost menunjukkan performa terbaik dengan akurasinya 92.31%, diikuti oleh Random Forest dengan 91.82% dan Decision Tree dengan 90.9%. Model-model ini unggul berkat kemampuannya dalam menangkap pola kompleks dan mengatasi variabilitas dalam data, dengan XGBoost dan Random Forest menggunakan teknik ensemble untuk mengurangi risiko overfitting. KNeighbors juga menunjukkan kinerja yang baik dengan akurasi 89.73%, meskipun sedikit lebih rendah dibandingkan model ensemble.

Sementara itu, model Logistic Regression dan SVM memberikan hasil yang lebih rendah, dengan SVM mencapai 83.93% dan Logistic Regression 86.25%. Meskipun hasilnya cukup baik, keduanya kalah dibandingkan dengan model ensemble yang lebih kompleks. Model GaussianNB dan BernoulliNB memiliki performa terendah, dengan GaussianNB mencatat 81.06% dan BernoulliNB hanya 66.89%, yang menunjukkan keterbatasan mereka dalam menangani data yang lebih kompleks.

Secara keseluruhan, XGBoost dan Random Forest tetap menjadi pilihan terbaik berdasarkan akurasi dan keseimbangan antara Precision, Recall, dan F1-Score, menunjukkan kemampuan mereka dalam menangani dataset yang lebih besar dan lebih beragam. Model seperti Decision Tree dan KNeighbors memberikan alternatif yang baik, tetapi dengan sedikit penurunan dalam performa, sedangkan model Logistic Regression dan SVM lebih cocok untuk kasus yang lebih sederhana dengan dataset yang lebih kecil.

## Kesimpulan Dampak Model Terhadap Business Understanding

## 1. Menjawab Problem Statements
Model yang dievaluasi telah berhasil menjawab sebagian besar problem statements.

- **Efektivitas Model Machine Learning**  
  Model XGBoost dan Random Forest menunjukkan performa terbaik dengan akurasi tinggi 92.31%.  
  Ini menunjukkan bahwa algoritma ensemble sangat efektif dalam memprediksi cuaca berdasarkan data historis.

- **Identifikasi Algoritma Terbaik**  
  Berdasarkan evaluasi, XGBoost dan Random Forest adalah algoritma unggulan yang memberikan hasil terbaik.  

## 2. Mencapai Goals
Tujuan dari proyek ini berhasil dicapai.

- **Pembangunan Model Akurasi Tinggi**  
  Dengan akurasi hingga 92.31%, proyek ini telah menghasilkan model prediksi cuaca yang sangat andal.

- **Perbandingan Performansi Algoritma**  
  Evaluasi menyeluruh terhadap berbagai algoritma memberikan pemahaman yang jelas tentang kekuatan dan kelemahan masing-masing metode.

## 3. Solusi Statement
Solusi yang dirancang berdampak signifikan terhadap hasil proyek.

- **Analisis Data dan Preprocessing**  
  Proses ini membantu dalam mengidentifikasi fitur-fitur penting yang relevan dengan weather type, temperature, wind speed, season.

- **Penerapan Algoritma Machine Learning**  
  Penggunaan berbagai model memberikan wawasan luas tentang pola prediksi cuaca dalam data.  
  Model ensemble seperti XGBoost dan Random Forest memberikan hasil terbaik.

- **Penanganan Outlier dan Ketidakseimbangan Data**  
  Teknik SMOTE dan IQR memastikan bahwa model tidak mempunyai data outlier dan dapat menangani ketidak seimbangan dalam data, meningkatkan akurasi dalam prediksi.


## Daftar Pustaka

1. Panish | Shea | Ravipudi LLP. (2025). Aviation and Plane Crash Statistics. Diakses pada 18 Juni 2025, dari https://www.panish.law/aviation_accident_statistics.html
2. Weather Concerns for General Aviation. (2018). Flight Safety Foundation. Diakses pada 18 Juni 2025, dari https://flightsafety.org/asw-article/weather-concerns-for-general-aviation/
3. Basuki Rochmat, & Sukendra Martha. (2021). Pengaruh Faktor Geografis terhadap Keselamatan Penerbangan di Indonesia. Jurnal Lemhannas RI, 9(2), 13–23. https://doi.org/10.55960/jlri.v9i2.388
4. NCAS. (2024). What causes weather? Diakses pada 18 Juni 2025, dari https://ncas.ac.uk/learn/what-causes-weather/
5. Mahendra, M. F. R., Azizah, N. L., & Sumarno. (2024). Implementasi Machine Learning Untuk Memprediksi Cuaca Menggunakan Support Vector Machine. Jurnal Ilmiah Komputasi, 23(1), 45–50. https://doi.org/10.32409/jikstik.23.1.3499
6. Rangkuti, M. Y. R., Alfansyuri, M. V., & Gunawan, W. (2021). Penerapan Algoritma K-Nearest Neighbor (Knn) Dalam Memprediksi Dan Menghitung Tingkat Akurasi Data Cuaca Di Indonesia. Hexagon Jurnal Teknik Dan Sains, 2(2), 11–16. https://doi.org/10.36761/hexagon.v2i2.1082
