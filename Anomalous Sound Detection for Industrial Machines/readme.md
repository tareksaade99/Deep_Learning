# Anomalous Sound Detection for Industrial Machines 🔊  
*A Deep Learning Project on Acoustic Anomaly Detection (DCASE Subset)*

## Overview
This project tackles the task of **Anomalous Sound Detection (ASD)** — identifying abnormal machine sounds that may indicate mechanical faults. Early anomaly detection can prevent costly downtimes and improve industrial reliability.  
We replicate a subset of the **DCASE 2020 Challenge**, focusing specifically on the **Slide Rail** machine type from the **ToyADMOS** and **MIMII** datasets.  
The dataset consists of **10-second mono audio recordings (16 kHz)** representing both *normal* and *anomalous* machine states. Since real anomalies are rare, models are trained **only on normal data** (unsupervised/semi-supervised setup) and evaluated on unseen normal and anomalous samples.

## Preprocessing
All audio files are:
- Loaded and resampled to **16 kHz**  
- Converted to **mono**  
- Cast to **float32** format  
These preprocessing steps ensure compatibility with deep learning feature extraction pipelines.

---

## Method I — CNN with VGGish + One-Class SVM
We first extract **128-dimensional embeddings** from each audio sample using **VGGish**, a CNN pre-trained on Google’s AudioSet.  
Embeddings are standardized using a `StandardScaler` (fit only on training data) to avoid data leakage.

An **One-Class SVM** with an **RBF kernel** is then trained exclusively on *normal* samples to learn their boundary in feature space. At inference, samples outside this boundary are classified as *anomalous*.  

- **Hyperparameters tuned:** `gamma`, `nu`  
- **Metrics used:** Accuracy, Precision, Recall, F1-score, AUC  

**Key findings:**  
High precision often led to low recall and vice versa — showing that while VGGish embeddings are semantically rich, they don’t perfectly separate normal and anomalous sounds.

---

## Method II — LSTM for Frame Prediction
This method models the **temporal evolution** of sounds.  
We convert each waveform into **log-mel spectrograms (128 frequency bins, 313 time frames)** and train a **2-layer LSTM** to predict the next frame from a sequence of 10 previous frames.

### Pipeline:
1. Train LSTM on *normal* data (input = 10 frames → output = next frame).  
2. Compute **prediction error** (MSE) on evaluation set to define a threshold `T`.  
3. Classify test samples as *anomalous* if their mean error exceeds `T`.  

- **Best threshold (T):** 0.001688  
- **AUC:** 0.9406  
- **Best results:**  
  | Class | Precision | Recall | F1-score |
  |:------|:-----------|:--------|:----------|
  | Normal | 0.76 | 0.90 | 0.82 |
  | Anomalous | 0.96 | 0.89 | 0.92 |

**Outcome:**  
The LSTM-based predictor achieved the **highest performance**, effectively capturing temporal dependencies in machine sounds.

---

## Method III — GRU Autoencoder for Sequence Reconstruction
The third method uses a **GRU-based autoencoder** trained to reconstruct sequences of spectrogram frames.  
The **reconstruction error (MSE)** serves as the anomaly score.

### Configuration:
- **Encoder/Decoder:** 2-layer GRUs (hidden size 256)  
- **Training:** Adam (lr=1e-3, batch size=64, epochs=10)  
- **Threshold (τ):** 0.000511 (TPR=0.8, FPR=0.123)  
- **Results:**
  | Class | Precision | Recall | F1-score |
  |:------|:-----------|:--------|:----------|
  | Normal | 0.62 | 0.87 | 0.72 |
  | Anomalous | 0.94 | 0.80 | 0.87 |
- **AUC:** 0.892  

**Observation:**  
The GRU autoencoder captures normal behavior patterns but is less discriminative than the LSTM predictor.

---

## Conclusion
This project explored **three strategies** for anomalous sound detection under unsupervised settings:
1. **VGGish + One-Class SVM** — robust feature-based baseline.  
2. **LSTM Frame Prediction** — best performing, AUC = **0.9406**, F1 = **0.92**.  
3. **GRU Autoencoder** — effective but slightly less accurate.  

Overall, **temporal modeling via LSTMs** proved most successful for detecting deviations in machine operation, outperforming static feature-based and reconstruction approaches.  

---
**Keywords:** Anomalous Sound Detection, LSTM, GRU, VGGish, One-Class SVM, DCASE, Unsupervised Learning, Audio Processing, PyTorch

