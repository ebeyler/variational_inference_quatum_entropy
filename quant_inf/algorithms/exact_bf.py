import numpy as np
from quant_inf.algorithms.base_class import InferenceAlgorithm
from itertools import product
from scipy.special import entr
from quant_inf.tools.manip import feature_vect,coeff_to_matrix,test_first_features

class ExactBruteForce(InferenceAlgorithm):
    """Brute force inference on the Ising model.

    Attributes:
        d (int): number of variables
        coefficients (list[tuple[set,float]]): list of coefficients,
            as (inf,coef), where ind (set[int]) represents a feature of
            the model (either an edge or a vertex) and coef (float)
            the associated coefficient
        marginals (np.ndarray): marginals [s,x_s]
        log_partition (float): log-partion function
        entropy (float): entropy
        moment_matrix (np.dnarray): moment matrix E_p(phi(x)^Tphi(x))
        features (list): list of features (set) defining the vector $\\varphi$
            If the feature vector is not specified, it will use (1,x_1,...,x_d).
        eps (float): temperature parameter $\\varepsilon$. Defaults to 1..


    Note:
        To access attributes marginals, entropy, moment_matrixand log_partition
        you must first solve the model by calling ExactBruteForce.solve()
    """

    def __init__(self,
                 d:int=None,
                 coefficients:list[tuple[set,float]]=None,
                 features:list[set]=None,
                 eps:float=1.
                 ) -> None:
        super().__init__()
        self.d = d
        self.coefficients = coefficients
        self.eps = eps
        self.features = features
        self.moment_matrix = None
    
    @property
    def coefficients(self):
        return self._coefficients
    
    @coefficients.setter
    def coefficients(self,value):
        self._coefficients = value
        if value is not None:
            self._coeff_matrix = coeff_to_matrix(value,self.d)
    
    @property
    def features(self):
        return self._features
    
    @features.setter
    def features(self,value):
        self._features = value
        if value is not None:
            test_first_features(value,self.d)

    def solve(self):
        """Compute the log-partition function, H(p) and optimal $\\Sigma_p$ by brute force.
        
        Set the value of the following attributes: marginals,
        entropy, moment_matrix & log_partition
        """

        if self.d is None or self.coefficients is None :
            raise ValueError(
                "Attributes d and coefficients must be initialized before calling solve()"
                )
        
        if self.features is None:
            self.features = (
                [set()] 
                + [{i} for i in range(1,d+1)]
            )

        hypercube = np.array([(1,) + x for x in product([-1,1],repeat=self.d)])

        unormalized_proba = np.exp(np.apply_along_axis(lambda x : np.dot(x,self._coeff_matrix@x)/self.eps,axis = 1,arr = hypercube))
        partition_function = np.mean(unormalized_proba)
        log_partition_function = self.eps*np.log(partition_function)
        proba = unormalized_proba/partition_function

        entropy = np.mean(entr(proba))
        
        features_hypercubes = np.vectorize(lambda x : feature_vect(x,self.features), signature='(d)->(n)')(hypercube) # x -> \varphi(x)
        sefl_tensor_prod = np.vectorize(lambda x: x[:,None]*x[None,:],signature="(n)->(n,n)") # x -> xx^T
        
        sigma = np.mean(proba[:,None,None] * sefl_tensor_prod(features_hypercubes),axis =0)

        self.log_partition = log_partition_function
        self.entropy = entropy
        self.moment_matrix = sigma
        mu = sigma[0,1:self.d+1]
        self.marginals = np.concatenate([((mu+1)/2)[:,None],((1-mu)/2)[:,None]],axis = 1)

if __name__ == '__main__':
    from quant_inf.tools import random_coefficients_gaussian

    d = 5
    features_0 = (
        [set()] 
        + [{i} for i in range(1,d+1)]
    )

    complete_graph_features = [{i} for i in range(1,d+1)] + [{i,j} for i in range(1,d+1) for j in range(i+1,d+1)]
    coefficients = random_coefficients_gaussian(complete_graph_features)
    exact_inference = ExactBruteForce(
        d=d,
        coefficients=coefficients,
        features=features_0)
    exact_inference.solve()
    print("True logp:",exact_inference.log_partition)
    print("Entropy:",exact_inference.entropy)
    print("Marginals:",exact_inference.marginals)
