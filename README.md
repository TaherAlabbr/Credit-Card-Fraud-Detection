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
* Optimized thresholds per model based on F1 score using validation sets.
* The Voting Classifier ensemble achieved:
  * **F1 Score:** 83.60%
  * **Precision:** 85.87%
  * **Recall:** 81.44%
* Conducted feature importance analysis, t-SNE visualization, and engineered key interaction features.
* Carefully avoided data leakage and ensured evaluation mimics real-world deployment.

---

## 🧪 Model Performance Overview

| Model                            | F1 Score   | Precision  | Recall     |
| -------------------------------- | ---------- | ---------- | ---------- |
| Neural Network (no resampling)   | 85.56%     | 88.89%     | 82.47%     |
| Random Forest (baseline)         | 81.03%     | 80.61%     | 81.44%     |
| K-Nearest Neighbors (baseline)   | 81.77%     | 88.10%     | 76.29%     |
| **Voting Classifier (ensemble)** | **83.60%** | **85.87%** | **81.44%** |

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
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for exploration and analysis
├── src/                    # Source code for model training and evaluation
├── reports/                # Generated reports and visualizations
├── requirements.txt        # Project dependencies
└── README.md               # Project overview and instructions
```

