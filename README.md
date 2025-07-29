# 🛡️ Fraud Detection with Machine Learning: A Comparative Study of Resampling and Modeling Strategies

This project addresses credit card fraud detection using supervised machine learning. Due to the highly imbalanced nature of the dataset (\~0.17% fraud cases), it evaluates six resampling techniques applied across **five classifiers**: Logistic Regression, Random Forest, K-Nearest Neighbors, Neural Network, and Voting Classifier ensemble. The ensemble combines the strengths of the base models to enhance detection robustness.

---

## 📊 Dataset

The dataset is the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), containing 284,807 transactions, with only 492 labeled as fraud.

* Features: `V1` to `V28` (PCA-transformed), `Amount`, `Time`
* Target: `Class` (1 = fraud, 0 = legitimate)

---

## 🧠 Key Highlights

* Compared six resampling methods: No resampling, Random Undersampling, NearMiss, KMeans Undersampling (custom), Random Oversampling, and SMOTE.
* Evaluated **five machine learning models**: Logistic Regression, Random Forest, K-Nearest Neighbors, Neural Network, and Voting Classifier.
* Developed a custom KMeans-based undersampler
* Optimized thresholds per model based on F1 score using validation sets.
* Conducted feature importance analysis, t-SNE visualization, and engineered key interaction features.
* Carefully avoided data leakage and ensured evaluation mimics real-world deployment.
* Conducted EDA with correlation heatmaps and fraud/normal feature comparisons
* Documented results and analysis in a professional project report
*  Wrote modular and reusable training/testing scripts with CLI support

---

## 🧪 Model Performance Overview

| Model                            | F1 Score   | Precision  | Recall     |
| -------------------------------- | ---------- | ---------- | ---------- |
| Neural Network (no resampling)   | 85.56%     | 88.89%     | 82.47%     |
| Random Forest (baseline)         | 81.03%     | 80.61%     | 81.44%     |
| K-Nearest Neighbors (baseline)   | 81.77%     | 88.10%     | 76.29%     |
| **Voting Classifier (ensemble)** | **83.60%** | **85.87%** | **81.44%** |

---

## 📄 Full Report

The full report offers a detailed overview of the entire project, including:

* Motivation and goals of the fraud detection task
* EDA insights with visualizations
* Feature engineering strategies
* Comparison of six resampling techniques
* Evaluation of five models and a voting ensemble
* Threshold tuning and performance metrics
* Preprocessing best practices and reproducibility

It’s written for both technical and non-technical readers and reflects real-world modeling considerations.

👉 [**Review Full Report (PDF)**](project_report.pdf)

---

## 🧰 Tools and Technologies

* Python 3.9+
* scikit-learn (used for classifiers and feature importance analysis, e.g., Random Forest)
* imbalanced-learn (for resampling techniques)
* matplotlib, seaborn (for visualizations)
* numpy, pandas (for data manipulation)
* t-SNE (from `sklearn.manifold`) for dimensionality reduction

---

## 📂 Project Structure

```
Credit-Card-Fraud-Detection/
├── notebooks/
│   └── credit_fraud.ipynb               # EDA, training & evaluation
│
├── reports/
│   └── Project Report.pdf               # Final written report
│
├── scripts/
│   ├── train/
│   │   └── credit_fraud_train.py        # Model training script
│   ├── test/
│   │   └── credit_fraud_test.py         # Model testing script
│   └── utils/
│       ├── credit_fraud_utils_data.py   # Data loading, cleaning, and splitting
│       ├── credit_fraud_utils_eval.py   # Evaluation metrics and plots
│       ├── credit_fraud_utils.py        # Shared utilities (e.g., logging)
│       ├── kmeans_undersampler.py       # Custom KMeans undersampling method
│       └── cli_args.py                  # CLI argument parser
│
└── saved_models/                        # Trained model artifacts

```
---
## 🛠️ Installation

To install all required dependencies, run:

```bash
pip install -r requirements.txt
```

Make sure you're using **Python 3.9 or later** for full compatibility.
---

## 📬 Contact

**Author:** Taher Alabbar  
**Email:** t.alabbar.ca@gmail.com  
[**LinkedIn**](https://www.linkedin.com/in/taher-alabbar/)  


Feel free to reach out if you have questions or would like to collaborate!
