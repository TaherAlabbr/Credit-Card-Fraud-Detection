from imblearn.base import BaseSampler
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.cluster import KMeans
import numpy as np
from numbers import Integral

class KMeansUnderSampler(BaseSampler):
    _sampling_type = 'under-sampling'

    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed='left')],
        "random_state": [Interval(Integral, 0, None, closed='left'), None],
        "sampling_strategy": [
            StrOptions({"auto", "majority", "not minority", "all", "minority"}),
            callable,
            dict,
            float
        ]
    }

    def __init__(self, n_clusters=100, random_state=42, sampling_strategy='auto'):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_clusters = n_clusters
        self.random_state = random_state

    def _fit_resample(self, X, y):
        """
        Internal method used by imblearn to perform resampling.
        """
        # Separate minority and majority classes
        X_minority = X[y == 1]
        y_minority = y[y == 1]
        X_majority = X[y == 0]
        y_majority = y[y == 0]

        # Fit KMeans on majority class
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans.fit(X_majority)
        X_majority_sampled = kmeans.cluster_centers_
        y_majority_sampled = np.zeros(self.n_clusters, dtype=y.dtype)

        # Combine
        X_resampled = np.vstack((X_minority, X_majority_sampled))
        y_resampled = np.concatenate((y_minority, y_majority_sampled))

        return X_resampled, y_resampled
