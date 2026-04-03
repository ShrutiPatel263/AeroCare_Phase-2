
# ✈️ AeroCare Phase-2: Advanced RUL Prediction using Deep Learning

> 🚀 Predicting **Remaining Useful Life (RUL)** of aircraft engines using advanced **Bidirectional LSTM and Novel Gated Ensemble Architectures** on NASA CMAPSS dataset.

---

## 📌 Overview

AeroCare Phase-2 is a deep learning–focused extension of AeroCare that explores **sequence modeling architectures** for predictive maintenance.

This project compares:

* A **strong baseline BiLSTM**
* A **novel multi-branch Gated Ensemble architecture**

Goal: Improve understanding of **temporal modeling strategies** for RUL prediction rather than just chasing accuracy.

---

## 🧠 Models Implemented

### 🔹 1. Baseline Model — Bidirectional LSTM

A clean and strong baseline using standard sequence modeling.

**Architecture:**

* BiLSTM (128 units) → BatchNorm → Dropout (0.3)
* BiLSTM (64 units) → BatchNorm → Dropout (0.3)
* Dense (50, ReLU)
* Output (Linear, RUL)

**Key Characteristics:**

* Simple sequential architecture
* No attention / gating / parallel branches
* Uses **Batch Normalization**
* Relies purely on **bidirectional temporal learning**

**Data Pipeline:**

* Sliding window: **30 cycles**
* Features: **16 selected sensors**
* Normalization: StandardScaler
* RUL cap: 125

**Training Strategy:**

* 5-Fold **GroupKFold Cross Validation**
* Best fold selected for inference

---

### 🔹 2. Novel Model — Gated Ensemble BiLSTM

A **multi-branch architecture** designed to learn diverse temporal representations.

**Architecture Overview:**

🧩 **Parallel Feature Extraction Branches:**

* LSTM branch → captures long-term dependencies
* GRU branch → efficient gated dynamics
* 1D CNN branch → local temporal patterns

Each branch:

* Outputs 64-d sequence
* Uses LayerNorm + Dropout

---

🧠 **Gating Mechanism (Core Innovation):**

Instead of simple concatenation:

* Learns **dynamic importance weights** for each branch
* Uses softmax-based gating
* Performs **weighted fusion of sequences**

👉 Model learns *which branch to trust more*

---

🔁 **Sequential Modeling:**

* BiLSTM (return_sequences=True)
* BiLSTM (return_sequences=False)

---

🎯 **Output Layer:**

* Dense layer
* Linear neuron → RUL prediction

---

📉 **Loss Function:**

* Huber Loss (δ = 1.0)
* Robust to outliers, sensitive to moderate errors

---

## 📊 Results

| Dataset | Model             | Accuracy |
| ------- | ----------------- | -------- |
| FD001   | BiLSTM (Baseline) | **87%**  |
| FD003   | BiLSTM (Baseline) | **86%**  |
| FD001   | Gated Ensemble    | 77%      |
| FD003   | Gated Ensemble    | 79%      |

---

## 📈 Key Insights

* ✅ **Baseline BiLSTM outperformed the complex model**
* ⚠️ Increased architectural complexity ≠ better performance
* 📉 Gating mechanism needs further tuning
* 🔍 Simpler models generalize better on CMAPSS

---

## 📂 Repository Structure

```
AeroCare_Phase-2/
│── Best_BidirectionalLSTM_FD001.ipynb
│── Best_BidirectionalLSTM_FD003.ipynb
│── Improvement_on_dataset_1.ipynb
│── Novel_Model_FD001_done.ipynb
│── Novel_model_FD003_done.ipynb
│── README.md
```

---

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Scikit-learn
* Matplotlib

---

## ▶️ How to Run

```bash
# Clone repo
git clone https://github.com/ShrutiPatel263/AeroCare_Phase-2.git

# Open notebooks
jupyter notebook
```

Run notebooks in order:

1. Baseline models
2. Novel model experiments


---

## 👩‍💻 Author

**Shruti Patel**
🎓 B.E. Computer Science And Engineering(Data Science)
💡 Machine Learning | Deep Learning | AI

