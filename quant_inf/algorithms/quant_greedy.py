import numpy as np
from quant_inf.algorithms.base_class import InferenceAlgorithm
from quant_inf.algorithms import QuantumRelaxation
from quant_inf.tools.list import unique, diff_list


class QuantGreedyRelaxation(InferenceAlgorithm):
    """Implement the quantum relaxation for
    approximate inference on the Ising model.

    Attributes:
        d (int): number of variables
        coefficients (list[tuple[set,float]]): list of coefficients,
            as (inf,coef), where ind (set[int]) represents a feature of
            the model (either an edge or a vertex) and coef (float)
            the associated coefficient
        marginals (np.ndarray): marginals [s,x_s]
        log_partition (float): TRW upper bound on the log-partion
        entropy (float): entropy
        moment_matrix (np.dnarray): moment matrix E_p(phi(x)^Tphi(x))
        features (list[set]): features selected by the greedy algorithm
        eps (float): temperature parameter $\\varepsilon$. Defaults to 1..

    Note:
        To access attributes marginals, entropy, moment_matrix and log_partition,
        you must first solve the model by calling QuantGreedyRelaxation.solve()
    """
    
    def __init__(self,
                 d:int=None,
                 coefficients:list[tuple[set,float]]=None,
                 eps:float=1.) -> None:
        super().__init__()
        self.d = d
        self.coefficients = coefficients
        self.eps = eps
        self.entropy = None
        self.moment_matrix = None
        self.features = None

    def solve(self,
              number_extra_features:int,
              tol_search:float=1.e-2,
              **kargs
              ):
        """Compute the quantum relaxation of
        the log-partition function, entropy and moment_matrix,
        with a greedy selection of extra features 

        Set the value of the following attributes: marginals,
        entropy, moment_matrix, features & log_partition

        Args:
            number_extra_features (int): number of extra features to
                add with greedy algorithm
            tol_search (float, optional): tolerance of the solver during
                the greedy procedure
            **kargs: keyword arguments to parametrize Chambolle-Pock
                algorithm (see package.algorithms.QuantumRelaxation)
        """
        self.features = None
        if number_extra_features == 0:
            self.features = [set()] + [{i} for i in range(1,self.d+1)]
            model = QuantumRelaxation(
                coefficients=self.coefficients,
                d=self.d,
                features=self.features,
                eps=self.eps)
            model.solve(**kargs)
            self.log_partition = model.log_partition
            self.entropy = model.entropy
            self.moment_matrix = model.moment_matrix
            self.marginals = model.marginals
        for _ in range(number_extra_features):
            self._select_feature(tol_search=tol_search,**kargs)

    def _select_feature(
            self,
            tol_search:float=1.e-2,
            **kargs):
        """Select one feature with greedy process, added to self.features.
        If self.features = None, it is iniatilized with features 
        1,x_1,...,x_d

        Args:
            tol_search (float, optional): tolerance of the solver during
                the greedy procedure
            **kargs: keyword arguments to parametrize Chambolle-Pock
                algorithm (see package.algorithms.QuantumRelaxation)
        """
        
        if self.features is None:
            self.features = [set()] + [{i} for i in range(1,self.d+1)]
        
        arg_solver = kargs.copy()
        arg_solver["tol"]=tol_search
        
        logp_greedy = +np.inf
        greedy_selected_feature = None
        features_greedy_pool = unique([feat1^feat2 for feat1 in self.features for feat2 in [{i} for i in range(1,self.d+1)]])
        features_greedy_pool = diff_list(features_greedy_pool,self.features)
        for j in range(len(features_greedy_pool)):
            prob_greedy = QuantumRelaxation(
                coefficients=self.coefficients,
                d=self.d,
                features=self.features+[features_greedy_pool[j]],
                eps=self.eps)
            prob_greedy.solve(**arg_solver)
            if  prob_greedy.log_partition < logp_greedy:
                logp_greedy = prob_greedy.log_partition
                selected_model = prob_greedy
                greedy_selected_feature = j
        self.features.append(features_greedy_pool[greedy_selected_feature])

        selected_model.solve(**kargs)
        self.log_partition = selected_model.log_partition
        self.entropy = selected_model.entropy
        self.moment_matrix = selected_model.moment_matrix
        self.marginals = selected_model.marginals


if __name__ == '__main__':
    from quant_inf.tools import random_coefficients_gaussian
    from quant_inf.algorithms import ExactBruteForce

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
    quantum_inference = QuantGreedyRelaxation(
        d=d,
        coefficients=coefficients)
    quantum_inference.solve(3)
    print("True logp     :",exact_inference.log_partition)
    print("Logp          :",quantum_inference.log_partition)
    print("True entropy  :",exact_inference.entropy)
    print("Entropy       :",quantum_inference.entropy)
    print("True marginals:\n",exact_inference.marginals)   
    print("Marginals     :\n",quantum_inference.marginals)