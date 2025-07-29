Got it! Here's your `README.md` rewritten without emojis, with clean and professional segmentation:

---

# Fraud Detection with Machine Learning

**A Comparative Study of Resampling and Modeling Strategies**

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions using advanced resampling techniques and ensemble modeling strategies.

---

## Overview

This project addresses credit card fraud detection using supervised machine learning on a highly imbalanced dataset (\~0.17% fraud cases). It implements a complete pipeline from:

* Data preprocessing and exploratory analysis
* Resampling strategy implementation
* Model training, threshold tuning, and evaluation
* Ensemble modeling and performance comparison
* Modular scripting with command-line interface (CLI) support

A custom KMeans-based undersampling method is also developed to handle class imbalance more intelligently than naive undersampling.

---

## Dataset

* Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* Total Transactions: 284,807
* Fraudulent Cases: 492
* Features:

  * V1 to V28 (PCA-transformed)
  * Amount, Time
  * Class (target: 1 = fraud, 0 = legitimate)

---

## Key Highlights

* Implemented six resampling techniques:

  * No resampling
  * Random Undersampling
  * NearMiss
  * KMeans Undersampling (custom)
  * Random Oversampling
  * SMOTE

* Evaluated five machine learning models:

  * Logistic Regression
  * Random Forest
  * K-Nearest Neighbors
  * Neural Network
  * Voting Classifier (ensemble)

* Developed a custom KMeans-based undersampler

* Performed threshold tuning based on validation F1 score

* Analyzed feature importance and conducted dimensionality reduction using t-SNE

* Avoided data leakage with proper separation and reproducibility

* Designed clean, modular, CLI-driven training/testing pipelines

* Created a comprehensive, professional project report

---

## Model Performance Overview

| Model                          | F1 Score | Precision | Recall |
| ------------------------------ | -------- | --------- | ------ |
| Neural Network (no resampling) | 85.56%   | 88.89%    | 82.47% |
| Random Forest (baseline)       | 81.03%   | 80.61%    | 81.44% |
| K-Nearest Neighbors (baseline) | 81.77%   | 88.10%    | 76.29% |
| Voting Classifier (ensemble)   | 83.60%   | 85.87%    | 81.44% |

---

## Full Report

The report includes:

* Motivation and problem framing
* Exploratory data analysis and class distribution
* Feature engineering strategies
* In-depth resampling method comparison
* Model evaluation and threshold tuning
* Metric interpretation and real-world considerations

[View Full Report (PDF)](project_report.pdf)

---

## Project Structure

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

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Python 3.9 or later is required for full compatibility.

---

## Tools and Technologies

* Python 3.9+
* scikit-learn
* imbalanced-learn
* pandas, numpy
* matplotlib, seaborn
* argparse
* t-SNE (`sklearn.manifold`)
* Jupyter Notebook

---
## Contact

**Author:** Taher Alabbar
**Email:** [t.alabbar.ca@gmail.com](mailto:t.alabbar.ca@gmail.com)
[**LinkedIn**](https://www.linkedin.com/in/taher-alabbar/)  
