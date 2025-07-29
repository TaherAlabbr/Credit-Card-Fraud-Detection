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
* Exploratory data analysis insights and class distribution
* Feature engineering strategies
* In-depth resampling method comparison
* Model evaluation and threshold tuning
* Metric interpretation and real-world considerations

[View Full Report (PDF)](reports/project_report.pdf)

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
│   ├── credit_fraud_train.py            # Model training script
│   ├── credit_fraud_test.py             # Model testing script
│
│   # Utility modules
│   ├── credit_fraud_utils_data.py       # Data loading, cleaning, and splitting
│   ├── credit_fraud_utils_eval.py       # Evaluation metrics and plots
│   ├── credit_fraud_utils.py            # Shared utilities (e.g., logging)
│   ├── kmeans_undersampler.py           # Custom KMeans undersampling method
│   └── cli_args.py                      # CLI argument parser
│
├── saved_models/                        # Trained model artifacts
├── README.md                            # Project overview and instructions
└── requirements.txt                     # Python package dependencies

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

## Command-Line Interface (CLI Options)

This project supports flexible configuration via command-line arguments using `argparse`. Below is a breakdown of available CLI options by category.

### File Paths

| Argument       | Description                                                                         |
| -------------- | ----------------------------------------------------------------------------------- |
| `--train-dir`  | Path to the training dataset CSV (default: `data/split/trainval.csv`)               |
| `--test-dir`   | Path to the test dataset CSV (default: `data/split/test.csv`)                       |
| `--save-dir`   | Directory to save trained model artifacts (default: `saved_models/`)                |
| `--load-dir`   | Path to load a saved model for evaluation (default: `saved_models/final_model.pkl`) |
| `--model-name` | 	Name used when saving the model file (e.g., `model_v1.pkl`)                        |

### Preprocessing Options

| Argument             | Description                                                                                                     |
| -------------------- | --------------------------------------------------------------------------------------------------------------- |
| `--scaling`          | Feature scaling method: `standard`, `minmax`, `robust`, or `raw`                                                |
| `--balance`          | Class balancing method: `none`, `undersample`, `nearmiss`, `kmeans`, `cluster_centroids`, `oversample`, `smote` |
| `--n-neighbors`      | Number of neighbors for SMOTE or NearMiss (default: `5`)                                                        |
| `--nearmiss-version` | Version of NearMiss to use (`1`, `2`, or `3`)                                                                   |
| `--n-clusters`       | Number of clusters for KMeans undersampling (default: `500`)                                                    |
| `--use-pca`          | Flag to apply PCA before training (default: `False`)                                                            |
| `--n-components`     | Number of PCA components to retain (default: `18`)                                                              |

### Model Selection

| Argument  | Description                                                                  |
| --------- | ---------------------------------------------------------------------------- |
| `--model` | Model to train: `LR` (Logistic Regression), `NN` (MLP), `RF`, `VC`, or `KNN` |

### Model Hyperparameters

#### K-Nearest Neighbors (KNN)

| Argument            | Description                               |
| ------------------- | ----------------------------------------- |
| `--knn-n-neighbors` | Number of neighbors to use (default: `7`) |
| `--knn-weights`     | Weight strategy: `uniform` or `distance`  |

#### Logistic Regression (LR)

| Argument        | Description                                       |
| --------------- | ------------------------------------------------- |
| `--lr-c`        | Inverse of regularization strength (default: `1`) |
| `--lr-max-iter` | Maximum iterations (default: `10000`)             |

#### Random Forest (RF)

| Argument            | Description                          |
| ------------------- | ------------------------------------ |
| `--rf-n-estimators` | Number of trees (default: `100`)     |
| `--rf-max-depth`    | Maximum tree depth (default: `None`) |

#### Neural Network (NN / MLP)

| Argument             | Description                                         |
| -------------------- | --------------------------------------------------- |
| `--nn-hidden-layers` | Comma-separated hidden layer sizes (e.g., `128,64`) |
| `--nn-max-iter`      | Maximum training iterations (default: `3000`)       |
| `--nn-activation`    | Activation function: `relu`, `tanh`, or `logistic`  |
| `--nn-lr`            | Learning rate (default: `0.001`)                    |
| `--nn-alpha`         | L2 regularization parameter (default: `0.001`)      |

### Additional Options

| Argument           | Description                                     |
| ------------------ | ----------------------------------------------- |
| `--grid-search`    | Enable `GridSearchCV` for hyperparameter tuning |
| `--cost-sensitive` | Use cost-sensitive learning via class weighting |

## Example Usage

Train a neural network with SMOTE and PCA:

```bash
python scripts/credit_fraud_train.py \
    --train-dir data/split/trainval.csv \
    --model NN \
    --balance smote \
    --scaling standard \
    --use-pca \
    --model-name smote_nn_pca
```
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
