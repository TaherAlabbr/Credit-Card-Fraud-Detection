# 🛡️ Fraud Detection with Machine Learning: A Comparative Study of Resampling and Modeling Strategies

This project explores the challenge of credit card fraud detection using supervised machine learning techniques. Given the highly imbalanced nature of fraud data (fraud cases ~0.17%), the study investigates and compares six resampling strategies across four core models: Logistic Regression, Random Forest, K-Nearest Neighbors, and Neural Networks. A soft Voting Classifier ensemble is then constructed to combine their strengths for robust fraud detection.

---

## 📊 Dataset

The dataset used is the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), consisting of 284,807 transactions, with only 492 labeled as fraud.

- Features: `V1` to `V28` (PCA-transformed), `Amount`, `Time`
- Target: `Class` (1 = fraud, 0 = normal)

---

## 🧠 Project Highlights

- ✅ **Compared 6 resampling techniques**: No resampling, Random Undersampling, NearMiss, KMeans Undersampling (custom), Random Oversampling, SMOTE.
- ✅ **Evaluated 4 ML models** with resampling and scaling combinations.
- ✅ **Optimized thresholds** per model using F1 score on validation sets.
- ✅ **Built a soft Voting Classifier ensemble**, achieving:
  - **F1 Score**: `83.60%`
  - **Precision**: `85.87%`
  - **Recall**: `81.44%`
- ✅ **Applied t-SNE**, feature importance analysis, and crafted a key feature interaction (`V17 × V14`) for final performance gains.
- ✅ **Handled data leakage risks**, preserved real-world deployment realism.

---

## 🧪 Model Performance Snapshot

| Model              | F1 Score | Precision | Recall  |
|-------------------|----------|-----------|---------|
| Logistic Regression (baseline) | 82.14%   | 88.46%    | 76.67%  |
| Neural Network (no resampling) | 86.42%   | 97.22%    | 77.78%  |
| Random Forest (baseline)       | 88.24%   | 93.75%    | 83.33%  |
| KNN (baseline)                 | 86.59%   | 95.95%    | 78.89%  |
| **Voting Classifier (ensemble)** | **83.60%** | **85.87%**  | **81.44%** |

---

## 🧰 Technologies Used

- Python 3.9+
- scikit-learn
- imbalanced-learn
- matplotlib, seaborn
- numpy, pandas
- t-SNE (from sklearn.manifold)

---

## 📁 Project Structure

