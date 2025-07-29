from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import GridSearchCV
from kmeans_undersampler import KMeansUnderSampler 
from imblearn.under_sampling import NearMiss
import pandas as pd
import numpy as np

def load_data(train_path, test_path):
    """
    Load and preprocess train and test datasets.

    - Removes duplicates.
    - Applies log transform to 'Amount'.
    - Drops 'Time' column.
    - Returns NumPy arrays.

    Parameters:
        train_path (str): Path to training CSV file.
        test_path (str): Path to testing CSV file.

    Returns:
        X_train, X_test, y_train, y_test: NumPy arrays.
    """

    def process(df):
        df = df.copy()
        #df = df.drop_duplicates() # keeping them give us a better performance!
        df['Amount'] = np.log1p(df['Amount'])
        df = df.drop('Time', axis=1)

        df['V17_V14'] = df['V17'] * df['V14']

        X = df.drop('Class', axis=1).values
        y = df['Class'].values
        return X, y

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train, y_train = process(train)
    X_test, y_test = process(test)

    return X_train, X_test, y_train, y_test

def get_scaler(scaling_strategy):
    """
    Returns a scaler object based on the scaling strategy.

    Parameters:
        scaling_strategy (str): Scaling method: 'raw', 'standard', or 'minmax'.

    Returns:
        scaler: Scaler object (StandardScaler, MinMaxScaler) or None if 'raw'.
    """
    if scaling_strategy == 'standard':
        return StandardScaler()
    elif scaling_strategy == 'minmax':
        return MinMaxScaler()
    elif scaling_strategy == 'robust':
        return RobustScaler()
    elif scaling_strategy == 'raw':
        return None  # No scaling applied
    else:
        raise ValueError(f"Unsupported scaling option: {scaling_strategy}")

def get_balance_strategy(args, random_state=42):
    """
    Returns a resampling strategy based on the specified balance method.

    Parameters:
    -----------
    args : Namespace
        Contains balance method and related parameters.

    random_state : int, default=42
        Seed for reproducibility.

    Returns:
    --------
    Sampler object, 'cost_sensitive' string, or None.

    Raises:
    -------
    ValueError if an unknown balance method is provided.
    """
    if args.balance == 'none':
        return None

    elif args.balance == 'undersample':
        return RandomUnderSampler(random_state=random_state)

    elif args.balance == 'nearmiss':
        return NearMiss(version=args.nearmiss_version, n_neighbors=args.n_neighbors)

    elif args.balance == 'kmeans':
        return KMeansUnderSampler(n_clusters=args.n_clusters, random_state=random_state)

    elif args.balance == 'oversample':
        return RandomOverSampler(random_state=random_state)

    elif args.balance == 'smote':
        return SMOTE(random_state=random_state, k_neighbors=args.n_neighbors)

    else:
        raise ValueError(
            "Unknown balance method '{}'. Choose from "
            "['none', 'cost_sensitive', 'undersample', 'nearmiss', 'kmeans', "
            "'oversample', 'smote']".format(args.balance)
        )

def setup_grid(model, args, cv=5, scoring='f1', n_jobs=-1 , verbose = 1):
    """
    Creates a GridSearchCV object for hyperparameter tuning.

    Parameters:
    -----------
    model : Pipeline
        The model or pipeline to tune.

    args : Namespace
        Contains model type and hyperparameter settings.

    cv : int, default=5
        Number of cross-validation folds.

    scoring : str, default='f1'
        Metric to optimize.

    n_jobs : int, default=-1
        Number of parallel jobs.

    verbose : int, default=1
        Controls verbosity.

    Returns:
    --------
    GridSearchCV object ready for fitting.
    """

    if args.model == 'LR':
        param_grid = { 
            'model__C': [0.1, 1, 10],  # Regularization strength: moderate to high range
            'model__class_weight': [ {0: 1, 1: 4}, 'balanced', None]  # Useful class weight options
        }
    elif args.model == 'KNN':
        param_grid = {
                'model__n_neighbors': [3, 5, 7, 9],
                'model__weights': ['uniform', 'distance']
        }

    elif args.model == 'RF':
        param_grid = {
            'model__n_estimators': [100],  # Number of trees: moderate values for speed & accuracy balance
            'model__max_depth': [ None],   # None (full depth) and moderate depth to control overfitting
            'model__class_weight': [None]
        }

    elif args.model == 'NN':
        param_grid = {
            'model__hidden_layer_sizes': [
                (128,),       
                (128, 64),    
                (256, 128),     
            ],
            'model__activation': ['relu'],      # 'relu' usually better for most tasks
            'model__alpha': [0.0001, 0.001],    # Regularization to prevent overfitting
            'model__learning_rate_init': [0.001, 0.01]  # A good default learning rate
        }

    elif args.model == 'VC':
        param_grid = {
            # RandomForestClassifier
            'model__rf__n_estimators': [100],
            'model__rf__max_depth': [None],

            # Neural Network (MLPClassifier)
            'model__nn__hidden_layer_sizes': [(256, 64)],
            'model__nn__activation': ['relu'],
            'model__nn__alpha': [0.001],
            'model__nn__learning_rate_init': [0.001],

            # KNeighborsClassifier
            'model__knn__n_neighbors': [7],
            'model__knn__weights': ['distance'],

            # VotingClassifier-level
            'model__voting': ['soft'],
            'model__weights': [[6, 4, 5], [7, 4, 5], [6, 3, 5], [6, 3, 4], [7, 3, 5]]

        }


    else:
        raise ValueError(f"Unknown model_name: {args.model}")
    
        # Add balance strategy parameters if balancing is enabled
    if args.balance == 'undersample':
        param_grid.update({
            'balance__sampling_strategy': [0.5, 0.7, 1.0]  # Ratio of minority to majority after undersampling
        })

    elif args.balance == 'oversample':
        param_grid.update({
            'balance__sampling_strategy': ['minority', 0.5, 0.7]  # Control how much to oversample the minority
        })

    elif args.balance == 'smote':
        param_grid.update({
            'balance__sampling_strategy': ['minority', 0.5, 0.7],
            'balance__k_neighbors': [3, 5],  # Number of nearest neighbors in SMOTE
        })

    elif args.balance == 'smote_tomek':
        param_grid.update({
            'balance__sampling_strategy': ['minority', 0.5, 0.7],
            'balance__smote': [SMOTE(k_neighbors=3), SMOTE(k_neighbors=5)],
        })

    elif args.balance == 'hybrid':
        param_grid.update({
            'over__sampling_strategy': [0.5 , 0.3],  # Oversampling ratio
            'over__k_neighbors': [3, 5],                    # SMOTE neighbors
            'under__sampling_strategy': [1.0, 0.7]          # Undersampling ratio
        })

    elif args.balance == 'kmeans':
        param_grid.update({
            'balance__n_clusters': [300, 400, 500]  # Number of clusters for KMeans undersampler
        })

    elif args.balance == 'nearmiss':
        param_grid.update({
            'balance__version': [1, 2],           # NearMiss versions known to perform well
            'balance__n_neighbors': [3, 5]        # Number of neighbors to consider

        })

    elif args.balance == 'cluster_centroids':
        param_grid.update({
            'balance__estimator__n_clusters': [100, 200, 300, 400],  # Number of clusters for ClusterCentroids

        })
    elif args.use_pca:
        param_grid.update({
            'pca__n_components' : [7, 14, 18]
        })

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, verbose= verbose, n_jobs=n_jobs, error_score='raise')
    return grid