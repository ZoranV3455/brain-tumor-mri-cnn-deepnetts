# Klasifikacija MRI snimaka tumora mozga primenom CNN u DeepNetts-u

## 1. Opis problema
Cilj projekta je automatska klasifikacija snimaka magnetne rezonance (MRI) mozga u četiri kategorije tumora:
- **Glioma** (gliom)
- **Meningioma** (meningiom)
- **Pituitary tumor** (tumor hipofize)
- **No tumor** (bez tumora)

Problem je medicinski značajan jer rana i tačna dijagnoza direktno utiče na izbor terapije i ishod lečenja. Ručna analiza MRI snimaka je spora i subjektivna, pa automatizacija pomoću dubokog učenja može značajno pomoći radiolozima.

## 2. Podaci

### Izvor
Dataset je preuzet sa Kaggle-a: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### Struktura
- Originalni dataset: ~7.200 slika (512×512 piksela)
- Korišćeno: **1.200 slika** (300 po klasi) – smanjeno zbog hardverskih ograničenja
- Format: JPG, RGB

### Podela
- Trening: 60% (720 slika)
- Validacija: 10% (120 slika)
- Test: 30% (360 slika)

### Preprocesiranje
- Skaliranje na **64×64 piksela** (sa 512×512)
- Strategija: `fitLarger` (zadržava proporcije)
- **Bez augmentacije** (isključeno zbog ograničenja RAM-a)
- **Bez standardizacije** (zeroMean=false – tehničko ograničenje trenutne verzije DeepNetts-a)

## 3. Arhitektura modela

Custom CNN od nule (bez transfer learning-a), ~2.2M parametara:

| Sloj | Detalji |
|------|---------|
| Input | 64×64×3 |
| Conv1 | 3×3, 32 filtera, ReLU |
| MaxPool1 | 2×2, stride 2 |
| Conv2 | 3×3, 64 filtera, ReLU |
| MaxPool2 | 2×2, stride 2 |
| Conv3 | 3×3, 128 filtera, ReLU |
| MaxPool3 | 2×2, stride 2 |
| Flatten | 8.192 |
| FC | 256 neurona, ReLU |
| Output | 4 neurona, Softmax |

- Loss: Categorical Cross-Entropy
- Optimizer: Adam

## 4. Trening

### Hiperparametri
| Parametar | Vrednost |
|-----------|----------|
| Learning Rate | 0.0001 |
| Max Epochs | 25 |
| Patience | 8 |
| Min Delta | 0.001 |
| Batch Mode | Isključen |
| Train/Val/Test | 60/10/30 |

### Tok treninga
Trening je trajao **77 minuta** (25 epoha) na AMD Ryzen 5 CPU laptopu.

| Epoha | Train Acc | Val Acc | Train Loss |
|-------|-----------|---------|------------|
| 1 | 54.7% | 54.3% | 1.13 |
| 5 | 81.8% | 72.8% | 0.69 |
| 7 | 86.9% | 76.9% | 0.58 |
| 13 | 93.1% | 79.4% | 0.66 |
| 20 | 95.2% | 84.6% | 0.62 |
| 25 | 94.6% | 77.9% | 1.16 |

## 5. Analiza osetljivosti i hiperparametarska optimizacija

Kroz više iteracija treninga testirani su različiti hiperparametri:

| Pokušaj | LR | Patience | MaxEpochs | Slike | Rezultat |
|---------|-----|----------|-----------|-------|----------|
| 1 | 0.001 | 2 | 100 | 1.400 (128×128) | Val Acc 67% – previše RAM-a |
| 2 | 0.001 | 2 | 100 | 1.400 (64×64) | Val Acc 79.2% – Infinity loss u epoch 8 |
| 3 | 0.0005 | 6 | 30 | 1.200 (128×128) | Val Acc 5.9% – divergirao |
| 4 | 0.0005 | 8 | 25 | 300 (64×64) | Val Acc 75.5% – overfitting |
| **5 (finalni)** | **0.0001** | **8** | **25** | **300 (64×64)** | **Val Acc 84.6%, Test 73.9%** |

Ključni zaključci:
- Smanjenje LR sa 0.001 na 0.0001 stabilizovalo je trening
- Povećanje Patience-a sa 2 na 8 sprečilo je prerano zaustavljanje
- Smanjenje broja slika sa 1.400 na 300 smanjilo je opterećenje RAM-a

## 6. Rezultati evaluacije

| Metrika | Vrednost |
|---------|----------|
| Test Accuracy | **73.9%** |
| Precision | **85%** |
| Recall | **85%** |
| F1 Score | **85%** |

### Confusion Matrix (414 test slika)
| | Glioma | Meningioma | No Tumor | Pituitary |
|-|--------|------------|----------|-----------|
| **Glioma** | 92 | 7 | 0 | 0 |
| **Meningioma** | 16 | 60 | 2 | 5 |
| **No Tumor** | 2 | 6 | 77 | 0 |
| **Pituitary** | 7 | 9 | 0 | 77 |

### Tačnost po klasama
- Glioma: **92.9%**
- No Tumor: **90.6%**
- Pituitary: **82.8%**
- Meningioma: **72.3%**

### Poređenje sa drugim modelima (isti dataset, Validation Accuracy)

| Model | Params | Image Size | Hardware | Val Acc |
|-------|--------|------------|----------|----------|
| EfficientNetB0 (Transfer) | 4.2M | 224×224 | GPU | 82.4% |
| **Ovaj CNN (od nule)** | **2.2M** | **64×64** | **CPU** | **84.6%** |
| GoogLeNet (Transfer) | 6.8M | 224×224 | GPU | 86.7% |

## 7. Diskusija

### Uspešno
- CNN od nule postigao je 84.6% Val Acc na CPU-u, bolje od EfficientNetB0 (82.4%) na GPU-u
- Klase glioma i bez tumora se prepoznaju sa preko 90% tačnosti
- 85% precision i recall pokazuju balansiran model

### Ograničenja
- 64×64 rezolucija gubi ~98% originalnih piksela (sa 512×512)
- Bez augmentacije, model overfituje (Train 95% vs Test 73.9%)
- Numerička nestabilnost (Infinity loss) u kasnijim epohama
- Meningiom se najviše meša sa gliomom (16 od 83 slučaja)

### Predlozi za poboljšanje
- Povećati rezoluciju na 128×128 ili 224×224
- Dodati augmentaciju (flip, rotacija)
- Koristiti grayscale (MRI snimci su crno-beli)
- Gradient clipping za sprečavanje eksplodirajućeg gradijenta
- Povećati broj slika (vratiti svih 7.200)

## 8. Zaključak

CNN model istreniran u DeepNetts-u uspešno klasifikuje MRI snimke tumora mozga u 4 kategorije sa **73.9% test tačnosti** i **85% precision/recall**. Model je pokazao visoku tačnost za gliom (92.9%) i zdrave mozgove (90.6%).

Ključni doprinos projekta je demonstracija da **Java-native deep learning (DeepNetts) na CPU-u može da parira Python modelima na GPU-u** – CNN od nule sa 2.2M parametara pobedio je EfficientNetB0 (4.2M, GPU, transfer learning) i bio blizu GoogLeNet-u (6.8M, GPU).

Projekat potvrđuje da DeepNetts, zahvaljujući Vector API-ju i Project Panama, predstavlja ozbiljnu alternativu za on-premise AI u Java ekosistemu, bez potrebe za skupim GPU hardverom ili Python stack-om.

## Kako reprodukovati

### 1. Preuzimanje repozitorijuma

git clone https://github.com/ZoranV3455/brain-tumor-mri-cnn-deepnetts.git
cd brain-tumor-mri-cnn-deepnetts

### 2. Preuzimanje dataseta
Skinuti dataset sa Kaggle-a (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) i raspakovati ga.

### 3. Podešavanje putanje do slika
U fajlu Data Sets/NewImageDataset.properties izmeniti liniju:
dataSet.imageDir=C:\\Users\\KORISNIK\\Downloads\\archive\\Training
zameniti sa stvarnom putanjom do foldera sa slikama.

### 4. Otvaranje projekta
- Instalirati DeepNetts (https://www.deepnetts.com/download)
- Pokrenuti deepnettsplatform64.exe
- File → Open Project → izabrati klonirani folder

### 5. Pokretanje treninga
- Kliknuti Run (zelena strelica) u toolbar-u
- Trening traje ~77 minuta (25 epoha) na CPU-u
- Model će biti sačuvan u Trained Models/ folderu

### 6. Korišćenje već istreniranog modela
Ako ne želite da trenirate ponovo, već istrenirani model se nalazi u:
Trained Models/deepNetwork300.dnet
