import numpy as np
import cvxpy as cp
from quant_inf.algorithms.base_class import InferenceAlgorithm
from quant_inf.tools.manip import coeff_to_matrix,test_first_features
from quant_inf.tools.cvxpy import V_constraints, pairwise_edges_constraints as pairwise_edges_constraints_CVX

class LogDetRelaxation(InferenceAlgorithm):
    """Implement the log-determinant relaxation [1] for
    approximate inference on the Ising model.

    Attributes:
        d (int): number of variables
        coefficients (list[tuple[set,float]]): list of coefficients,
            as (inf,coef), where ind (set[int]) represents a feature of
            the model (either an edge or a vertex) and coef (float)
            the associated coefficient
        marginals (np.ndarray): marginals [s,x_s]
        log_partition (float): upper bound on the log-partion
        entropy (float): entropy
        moment_matrix (np.dnarray): moment matrix E_p(phi(x)^Tphi(x))
        features (list[set]): list of features (set) defining the vector $\\varphi$
            If the feature vector is not specified, it will simply use (1,x_1,...,x_d),
            orresponding to the relaxation in [1]. Using a larger feature vector will
            lead to more constraints on the optimization domain, but will not change the
            objective function.
        eps (float): temperature parameter $\\varepsilon$. Defaults to 1..  
    
    Note:
        To access attributes marginals, entropy, moment_matrix and log_partition
        you must first solve the model by calling LogDetRelaxation.solve()
    
    Reference:
        [1] Michael Jordan and Martin J Wainwright. “Semidefinite Relaxations
            for Approximate Inference on Graphs with Cycles”. In: Advances in
            Neural Information Processing Systems. Vol. 16. MIT Press, 2003.
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

    def solve(self,pairwise_edges_constraints:bool=True):
        """Compute the log-determinant relaxation of
        the log-partition function, entropy and moment_matrix
        
        Set the value of the following attributes: marginals,
        entropy, moment_matrix & log_partition

        Args:
            pairwise_edges_constraints (bool, optinal): if True, add constraints from Eq.
                (10) of [1]. Defaults to True.
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

        # Defining optimization problem for Log-det relaxation
        n = len(self.features)
        ep = cp.Parameter(nonneg=True)
        ep.value = self.eps
        Sigma = cp.Variable((n,n),symmetric = True)
        constraints = [Sigma >> 0, cp.trace(Sigma) == n] + V_constraints(Sigma,self.features)

        if pairwise_edges_constraints is True:
            constraints += pairwise_edges_constraints_CVX(
                Sigma=Sigma,
                features=self.features,
                d=self.d
                )
        
        block = np.eye(self.d+1); block[0,0] = 0
        objective_log_det = (
            cp.scalar_product(self._coeff_matrix,Sigma[:self.d+1,:self.d+1])
            + (ep/2) * cp.log_det(Sigma[:self.d+1,:self.d+1]+ (1/3)*block)
            + (ep*self.d/2) * np.log(np.pi * np.e / 2) - ep*self.d*np.log(2)
        )

        prob_log_det = cp.Problem(cp.Maximize(objective_log_det),constraints)

        prob_log_det.solve()
        self.solver_status = prob_log_det.status

        self.log_partition = prob_log_det.value
        self.entropy = (
            (1/2)*cp.log_det(Sigma[:self.d+1,:self.d+1] + (1/3)*block)
            + (self.d/2) * np.log(np.pi * np.e / 2)
            - self.d*np.log(2)
            ).value
        self.moment_matrix = Sigma.value
        mu = self.moment_matrix[0,1:self.d+1]
        self.marginals = np.concatenate([((mu+1)/2)[:,None],((1-mu)/2)[:,None]],axis = 1)

if __name__ == '__main__':
    from quant_inf.tools import random_coefficients_gaussian
    from quant_inf.algorithms import ExactBruteForce

    d = 5

    features_0 = (
        [set()] 
        + [{i} for i in range(1,d+1)]
    )

    print("---------- Test with eps=1. ----------")
    complete_graph_features = [{i} for i in range(1,d+1)] + [{i,j} for i in range(1,d+1) for j in range(i+1,d+1)]
    coefficients = random_coefficients_gaussian(complete_graph_features)
    exact_inference = ExactBruteForce(
        d=d,
        coefficients=coefficients,
        features=features_0)
    exact_inference.solve()

    logdet_inference = LogDetRelaxation(
        d=d,
        coefficients=coefficients,
        features=features_0)
    logdet_inference.solve()

    print("True logp     :",exact_inference.log_partition)
    print("Logp          :",logdet_inference.log_partition)
    print("True entropy  :",exact_inference.entropy)
    print("Entropy       :",logdet_inference.entropy)
    print("True marginals:\n",exact_inference.marginals)   
    print("Marginals     :\n",logdet_inference.marginals)

    print("---------- Test with eps=.1 ----------")
    exact_inference.eps=.1
    exact_inference.solve()
    logdet_inference.eps=.1
    logdet_inference.solve()
    print("True logp     :",exact_inference.log_partition)
    print("Logp          :",logdet_inference.log_partition)
    print("True entropy  :",exact_inference.entropy)
    print("Entropy       :",logdet_inference.entropy)
    print("True marginals:\n",exact_inference.marginals)   
    print("Marginals     :\n",logdet_inference.marginals)