<p align="center">

# Credit Card Fraud Detection: A Complete ML Pipeline

**A Comparative Study of Resampling and Modeling Strategies**

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions, leveraging advanced resampling techniques and ensemble modeling to tackle class imbalance and improve predictive performance.

</p>


---

## Overview

This project tackles credit card fraud detection using supervised and unsupervised machine learning on a highly imbalanced dataset (\~0.18% fraud cases). It builds a complete and modular pipeline that spans data preprocessing, exploratory analysis, feature engineering, and advanced resampling strategies to address class imbalance.

To improve model performance and robustness, multiple resampling techniques—such as random undersampling, SMOTE (Synthetic Minority Oversampling Technique), and a custom KMeans-based undersampling method—were applied selectively with different models. We experimented with various classifiers, including logistic regression, random forest, K-Nearest Neighbors (KNN), neural networks, and an ensemble voting classifier, each combined with the resampling technique best suited to its learning characteristics.

After thorough evaluation using threshold tuning and F1-score optimization on validation data, the best performing model was selected. Subsequent feature engineering was then applied specifically to this final model to further enhance its predictive performance and generalization.

The resulting pipeline is designed for reproducibility, supports command-line configuration, and follows modular, clean coding practices—making it both practical and adaptable for real-world fraud detection scenarios.

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

* Implemented six resampling techniques to address class imbalance:
  
  * baseline (cost-sensetive)
  * Random Undersampling
  * NearMiss
  * Custom KMeans-based undersampling
  * Random Oversampling
  * SMOTE
    
* Trained and compared five machine learning models:

  * Logistic Regression
  * Random Forest
  * K-Nearest Neighbors
  * Neural Network (MLPClassifier)
  * Ensemble Voting Classifier combining top performers
* Custom KMeans-based undersampler developed to improve representation of the majority class while preserving data structure
* Feature engineering to enhance model performance, supported by feature importance analysis to identify influential variables
* Applied threshold tuning based on F1-score to better balance precision and recall
* Modular, CLI-driven architecture for reproducible and configurable experimentation
* Professional final report and cleanly structured Jupyter notebooks for analysis, experimentation, and reproducibility
* Designed for easy extensibility to test new resampling or modeling approaches

---

## Model Performance Overview

| Model                              | F1 Score | Precision | Recall  | PR-AUC  |
| ---------------------------------- | -------- | --------- | ------- | ------- |
| Neural Network (threshold=0.606)   | 85.25%   | 90.70%    | 80.41%  | 85.70%  |
| Random Forest (threshold=0.364)    | 84.69%   | 83.84%    | 85.57%  | 85.01%  |
| K-Nearest Neighbors (threshold=0.414) | 87.05%   | 87.50%    | 86.60%  | 82.81%  |
| **Voting Classifier (threshold=0.283)** | **86.73%** | **85.86%** | **87.63%** | **87.73%** |

---

## 📄 Full Report

The report includes:

* Motivation and problem framing
* Exploratory data analysis insights and class distribution
* Feature engineering strategies
* In-depth resampling method comparison
* Model evaluation and threshold tuning
* Metric interpretation and real-world considerations

👉 [**View Full Report (PDF)**](reports/project_report.pdf)


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
**Email:** t.alabbar.ca@gmail.com, [**LinkedIn**](https://www.linkedin.com/in/taher-alabbar/)  
