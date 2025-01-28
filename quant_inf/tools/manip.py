import numpy as np

def feature_vect(x,features):
    """Compute feature vector $\\varphi(x)$

    Args:
        x (numpy.ndarray): vector x representing the value at each vertex
        /!\\ x[i] represents the value at edge $x_i$, for i = 1...d
        x[0] should by equal to 1.
        features (list): list of features (set) defining the vector $\\varphi$

    Returns:
        numpy.array: $\\varphi(x)$
    """
    assert x[0] == 1
    return np.array([np.prod(x[list(feat)]) for feat in features])

def coeff_to_matrix(coefficients:list[tuple[set,float]],
                    d:int) -> np.ndarray:
    """Create symetric matrix representing coefficients,
    i.e. (1, x)@matrix@(1, x) = sum_i(theta_i*x_i) + sum_ij(theta_ij*x_i*x_j)

    Args:
        coefficients (list[tuple[set,float]]): list of coefficients, as
            (inf,coef), where ind (set) represents a feature (either a
            vertex or an edge) of the model and coef (float) the associated
            coefficient
        d (int): Number of vertex in the models

    Raises:
        ValueError: raised if any feature is not an edge or a vertex
    
    Returns:
        np.ndarray
    """
    F = np.zeros((d+1,d+1))
    for feat,coef in coefficients:
        if len(feat) == 1:
            i, = feat
            F[0,i],F[i,0] = coef/2,coef/2
        elif len(feat) == 2:
            i,j = feat
            F[j,i],F[i,j] = coef/2,coef/2
        else:
            raise ValueError("Only support MRF")
    return F

def test_first_features(features,d):
    """Test whether the first d+1 features corresponds
    to (1,x)

    Args:
        features (_type_): list of features (set) defining
            a vector $\\varphi$
        d (int): number of variables
    """
    features_1 = (
        [set()] 
        + [{i} for i in range(1,d+1)]
    )
    assert features[:d+1]  == features_1