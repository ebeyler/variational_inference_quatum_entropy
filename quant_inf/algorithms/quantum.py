import numpy as np
from quant_inf.algorithms.base_class import InferenceAlgorithm
from quant_inf.tools.manip import coeff_to_matrix,test_first_features
from quant_inf.tools.projection_matrix import Projection
from scipy.special import wrightomega,entr
import warnings

class QuantumRelaxation(InferenceAlgorithm):
    """Implement the quantum relaxation for
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
        features (list): list of features (set) defining the vector $\\varphi$
            If the feature vector is not specified, it will use (1,x_1,...,x_d).
        eps (float): temperature parameter $\\varepsilon$. Defaults to 1..

    Note:
        To access attributes marginals, entropy, moment_matrix and log_partition,
        you must first solve the model by calling QuantumRelaxation.solve()
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
            self._proj_V = Projection(value)
            
    def solve(
            self,
            max_iter:int=10000,
            tol:float=1e-8,
            tau:float=3.,
            sig:float=.3):
        """Compute the quantum relaxation of
        the log-partition function, entropy and moment_matrix,
        using Chambolle-Pock dual algorithm
        
        Set the value of the following attributes: marginals,
        entropy, moment_matrix & log_partition

        Args:
            max_iter (int, optinal): maximal number of iterations for 
                Chambolle-Pock algorithm. Defaults to 10000.
            tol (float, optional): Optimization stoping criterion on duality gap.
                Defaults to .9.
            tau (float, optional): Chambolle-Pock primal step-size.
                Defaults to 3..
            sig (float, optional): Chambolle-Pock dual step-size.
                Defaults to .3.
        
        Note:
            One should have tau*sig < 1
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
            
        n = len(self.features)
        k = len(self._coeff_matrix)
        F = np.zeros((n,n))
        F[:k,:k] = self._coeff_matrix
   
        x = np.eye(n)
        y = np.eye(n)
        x_bar = x

        for i in range(max_iter):
            # prox step on dual variable
            arg_prox = y + sig*x_bar
            proj = self._proj_V(arg_prox)
            y = arg_prox - proj - (sig - (np.trace(proj)/n))*np.eye(n)

            # prox step on primal variable
            arg_prox = x - tau*y
            eigval,eigvect = np.linalg.eigh(n*(F + arg_prox/tau)/self.eps)
            x_new = (eigvect * (self.eps*tau/n) * np.real(wrightomega(eigval - np.log(self.eps*tau/n)))) @ eigvect.T

            x_bar = 2*x_new - x
            x = x_new

            if i%10 == 0 or i == max_iter-1:
                #Computing feasible x
                x_feasible = self._proj_V(x)
                x_feasible = x_feasible + (1 - (np.trace(x_feasible))/n)*np.eye(n)
                eig = np.linalg.eigvalsh(x_feasible)
                min_eig = np.min(eig)
                if min_eig<0:
                    u=(-min_eig)/(1-min_eig)
                else:
                    u=0
                x_feasible = (1-u)*x_feasible + u*np.eye(n)
                eig = (1-u)*eig + u*np.ones(n)
                eig = eig*(eig>0) #some values may remain negative up to machine precision
                entropy = (1/n)*np.sum(entr(eig))
                primal_value = np.sum(F[:k,:k]*x_feasible[:k,:k]) + self.eps*entropy
                
                eig = np.linalg.eigvalsh(n*(F-y)/self.eps)
                dual_value = np.trace(y) + self.eps*np.sum(np.exp(eig))/n - self.eps
                
                if np.abs(dual_value-primal_value) < tol:
                    break
        
        if i == max_iter-1:
            warnings.warn(f'Did not reached convergence. Duality gap: {np.abs(dual_value-primal_value):.2e}')

        self.log_partition = primal_value
        self.entropy = entropy
        self.moment_matrix = x_feasible
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
    quantum_inference = QuantumRelaxation(
        d=d,
        coefficients=coefficients,
        features=features_0)
    quantum_inference.solve()
    print("True logp     :",exact_inference.log_partition)
    print("Logp          :",quantum_inference.log_partition)
    print("True entropy  :",exact_inference.entropy)
    print("Entropy       :",quantum_inference.entropy)
    print("True marginals:\n",exact_inference.marginals)   
    print("Marginals     :\n",quantum_inference.marginals)

    print("---------- Test with eps=2. ----------")
    exact_inference.eps=2
    exact_inference.solve()
    quantum_inference.eps=2
    quantum_inference.solve()
    print("True logp     :",exact_inference.log_partition)
    print("Logp          :",quantum_inference.log_partition)
    print("True entropy  :",exact_inference.entropy)
    print("Entropy       :",quantum_inference.entropy)
    print("True marginals:\n",exact_inference.marginals)   
    print("Marginals     :\n",quantum_inference.marginals)