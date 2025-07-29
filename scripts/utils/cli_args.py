import argparse

def parse_hidden_layers(s):
    """
    Parse hidden layer sizes from a string.
    Accepts either:
    - A single integer: "100" → (100,)
    - A comma-separated list: "64,32" → (64, 32)
    """
    try:
        if ',' in s:
            return tuple(int(x) for x in s.split(','))
        return (int(s),)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Hidden layers must be an integer or a comma-separated list of integers, e.g., '64,32' or '100'"
        )

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Train and evaluate various models for fraud detection with preprocessing and resampling options.'
    )

    # === File Paths ===
    parser.add_argument('--train-dir', type=str, default='data/split/trainval.csv',
        metavar='TRAIN_FILE_PATH',
        help='Path to the training dataset CSV file (default: %(default)s)')

    parser.add_argument('--test-dir', type=str, default='data/split/test.csv',
        metavar='TEST_FILE_PATH',
        help='Path to the testing dataset CSV file (default: %(default)s)')

    parser.add_argument('--save-dir', type=str, default='saved_models',
        metavar='SAVE_MODEL_DIR',
        help='Directory to save trained model artifacts (default: %(default)s)')

    parser.add_argument('--load-dir', type=str, default='saved_models/model_bundel.pkl',
        metavar='LOAD_MODEL_PATH',
        help='Path to load a saved model for evaluation or inference (default: %(default)s)')

    parser.add_argument('--model-name', type=str, default='model_v1',
        metavar='MODEL_NAME',
        help='Model identifier name used during saving/loading (default: %(default)s)')

    # === Preprocessing Options ===
    parser.add_argument('--scaling', type=str, default='standard',
        choices=['standard', 'minmax', 'robust', 'raw'],
        metavar='SCALING',
        help=(
            'Feature scaling strategy:\n'
            '  standard → StandardScaler (zero mean, unit variance)\n'
            '  minmax   → MinMaxScaler (scales to [0, 1])\n'
            '  robust   → RobustScaler (median and IQR; better for outliers)\n'
            '  raw      → No scaling'
        )
    )

    parser.add_argument('--balance', type=str, default='none',
        choices=['none', 'undersample', 'nearmiss', 'kmeans', 'cluster_centroids', 'oversample', 'smote'],
        metavar='METHOD',
        help='Class imbalance strategy (default: %(default)s)')

    parser.add_argument('--n-neighbors', type=int, default=5,
        metavar='K',
        help='Number of neighbors for SMOTE or NearMiss (default: %(default)s)')

    parser.add_argument('--nearmiss-version', type=int, default=1,
        choices=[1, 2, 3],
        help='Version of NearMiss to use if selected (default: %(default)s)')

    parser.add_argument('--n-clusters', type=int, default=500,
        metavar='K',
        help='Number of clusters for KMeans resampling (default: %(default)s)')

    parser.add_argument('--use-pca', action='store_true',
        help='Enable PCA for dimensionality reduction before training')

    parser.add_argument('--n-components', type=int, default=18,
        metavar='N',
        help='Number of PCA components to keep (default: %(default)s)')

    # === Model Selection ===
    parser.add_argument('--model', type=str, default='LR',
        choices=['LR', 'NN', 'RF', 'VC', 'KNN'],
        help='Model type to train: LR (Logistic Regression), NN (Neural Network), RF (Random Forest), VC (Voting Classifier), or KNN (K-Nearest Neighbors) (default: %(default)s)')

    # === KNN Parameters ===
    parser.add_argument('--knn-n-neighbors', type=int, default=7,
        help='Number of neighbors for KNN classifier (default: %(default)s)')

    parser.add_argument('--knn-weights', type=str, default='distance',
        choices=['uniform', 'distance'],
        help='Weight function for KNN (default: %(default)s)')

    # === Logistic Regression Parameters ===
    parser.add_argument('--lr-c', type=float, default=1.0,
        help='Inverse of regularization strength for Logistic Regression (default: %(default)s)')

    parser.add_argument('--lr-max-iter', type=int, default=10000,
        help='Maximum iterations for Logistic Regression (default: %(default)s)')

    # === Random Forest Parameters ===
    parser.add_argument('--rf-n-estimators', type=int, default=100,
        help='Number of trees in the Random Forest (default: %(default)s)')

    parser.add_argument('--rf-max-depth', type=int, default=None,
        help='Maximum depth of trees in the Random Forest (default: %(default)s)')

    # === Neural Network Parameters ===
    parser.add_argument('--nn-hidden-layers', type=parse_hidden_layers, default='128',
        help='Comma-separated sizes for hidden layers in MLP (default: %(default)s)')

    parser.add_argument('--nn-max-iter', type=int, default=3000,
        help='Maximum training iterations for MLP (default: %(default)s)')

    parser.add_argument('--nn-activation', type=str, default='relu',
        choices=['relu', 'tanh', 'logistic'],
        help='Activation function for MLP (default: %(default)s)')

    parser.add_argument('--nn-lr', type=float, default=0.001,
        help='Initial learning rate for MLP (default: %(default)s)')

    parser.add_argument('--nn-alpha', type=float, default=0.001,
        help='L2 regularization strength (alpha) for MLP (default: %(default)s)')

    # === Tuning & Cost Sensitivity ===
    parser.add_argument('--grid-search', action='store_true',
        help='Enable hyperparameter tuning using GridSearchCV')

    parser.add_argument('--cost-sensitive', action='store_true',
        help='Enable cost-sensitive learning by adjusting class weights')

    return parser.parse_args()
