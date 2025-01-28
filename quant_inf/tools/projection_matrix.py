import numpy as np

def get_feature_matrix(features):
    """For a features vector, compute the number of time each feature
    appears in the feature matrix, and position of theses features
    (each feature is replace by a unique int)

    Args:
        features (list[set]): list of features (set) defining the
            vector $\\varphi$

    Returns:
        tuple[np.array,np.array]: number of time each feature
            appears ,position of theses features
    """
    feature_matrix = np.array(
            [[str(sorted(list(feat1^feat2))) for feat1 in features] for feat2 in features]
            )
    _, feature_inverse, feature_counts = np.unique(feature_matrix,return_counts=True,return_inverse=True)
    return feature_counts,feature_inverse.reshape(len(features),-1)

class Projection:
    """Projection onto subspace V defined by a feature vector.
    """
    def __init__(self,features) -> None:
        self.n = len(features)
        self._feature_counts, self._feature_inverse = get_feature_matrix(features)

    def __call__(self,x:np.ndarray):
        """Compute projection of x onto subspace V

        Args:
            x (np.ndarray): matrix

        Returns:
            np.ndarray: projected matrix
        """
        accumulator = np.bincount(self._feature_inverse.ravel(), weights=x.ravel())
        accumulator = accumulator/self._feature_counts
        return accumulator[self._feature_inverse]