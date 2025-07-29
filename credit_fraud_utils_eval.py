from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score
import numpy as np

def find_best_threshold(model, X, y, num_thresholds=100):
    """
    Find the best threshold by testing evenly spaced thresholds from 0 to 1.

    Args:
        model: Trained binary classifier with predict_proba.
        X: Feature data.
        y: True binary labels.
        num_thresholds: Number of threshold candidates to evaluate.

    Returns:
        float: Threshold with highest F1 score.
    """
    probs = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0, 1, num_thresholds)

    f1_scores = [f1_score(y, (probs >= t).astype(int)) for t in thresholds]

    best_threshold = thresholds[np.argmax(f1_scores)]
    
    return best_threshold

def evaluate_with_threshold(model, X, y, data_name, threshold):
    """
    Evaluate classification metrics using a custom threshold.

    Parameters:
        model: trained classifier with predict/predict_proba
        X: input features
        y: true labels
        data_name: dataset label (e.g. 'Train', 'Test')
        threshold: probability cutoff for positive class 
    """
    y_scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    y_pred = (y_scores >= threshold).astype(int)

    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    pr_auc = average_precision_score(y, y_scores)

    print(f"""{data_name} Dataset Evaluation (threshold={threshold:.3f}):
        F1 Score     : {f1 * 100:.2f}%
        Precision    : {precision * 100:.2f}%
        Recall       : {recall * 100:.2f}%
        PR-AUC       : {pr_auc * 100:.2f}%""")