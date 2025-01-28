import numpy as np
import cvxpy as cp
from quant_inf.algorithms.base_class import InferenceAlgorithm
from quant_inf.tools.manip import coeff_to_matrix,test_first_features
from quant_inf.tools.cvxpy import V_constraints
import warnings

def f(t):
    return t*np.log(t) - t + 1

def grad_f(t):
    return np.log(t)

def oracle(Sigma:np.ndarray,tol_eigenvalues:float = 1e-8)-> tuple[float,np.ndarray]:
    n = Sigma.shape[0]
    eigvalues,eigvectors = np.linalg.eigh(Sigma)
    sig_log_sig = (eigvectors * f(eigvalues)) @ eigvectors.T
    value = np.max(np.diag(sig_log_sig))
    ind_max = np.argmax(np.diag(sig_log_sig))
    grad = 0
    for k in range(n):
        grad += (eigvectors[ind_max, k]**2) *(grad_f(eigvalues[k]))*(eigvectors[:,k,None]@eigvectors[:,k,None].T)
        for l in range(k+1, n):
            if np.abs(eigvalues[l] - eigvalues[k]) < tol_eigenvalues:
                grad += eigvectors[ind_max, k]*eigvectors[ind_max, l]*(grad_f(eigvalues[k]))*(eigvectors[:,k,None]@eigvectors[:,l,None].T+eigvectors[:,l,None]@eigvectors[:,k,None].T)
            else:
                grad += eigvectors[ind_max, k]*eigvectors[ind_max, l]*((f(eigvalues[l]) - f(eigvalues[k]))/(eigvalues[l] - eigvalues[k]))*(eigvectors[:,k,None]@eigvectors[:,l,None].T+eigvectors[:,l,None]@eigvectors[:,k,None].T)
    return value,grad

class QuantDiagRelaxation(InferenceAlgorithm):
    """Implement the quantum relaxation with
    diagonal metric learning for approximate
    inference on the Ising model.

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
        you must first solve the model by calling QuantDiagRelaxation.solve()
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

    def solve(self,
              max_iter:int=200,
              tol_kelley:float=1e-2,
              tol_solver:float=1e-12,
              tol_psd:float=1e-12,
              verbose:bool=False):
        """Compute the quantum relaxation with diagonal metric
        learning of the log-partition function, entropy and
        moment_matrix, using Kelley's method.
        
        Set the value of the following attributes: marginals,
        entropy, moment_matrix & log_partition

        Args:
            max_iter (int, optional): maximal number of kelley
                iterations. Defaults to 200.
            tol_kelley (float, optional): tolerance for stopping criterion
                of kelley method. Defaults to 1e-2.
            tol_solver (float, optional): tolerance for the solver.
                Defaults to 1e-12.
            tol_psd (float, optional): tolerance for the psd constraint.
                Defaults to 1e-12.
            verbose (bool, optional): Defaults to False.
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
        t = cp.Variable(1)
        Sigma = cp.Variable((n,n),symmetric = True)
        ep = cp.Parameter(nonneg=True)
        objective = cp.Maximize(t)
        constraints = [Sigma >> 0, cp.trace(Sigma) == n] + V_constraints(Sigma,self.features)
        
        ep.value = self.eps
        Sigma.value = np.eye(n)
        value,grad = oracle(Sigma.value)
        list_value = [value - np.trace(grad@Sigma.value)]
        list_grad = [grad]

        best_lower_bound = np.trace(self._coeff_matrix)

        for i in range(max_iter):
            constraints.append(cp.trace(self._coeff_matrix@Sigma[:self.d+1,:self.d+1]) - ep*list_value[i] - ep*cp.trace(list_grad[i]@Sigma) >= t)
            
            problem = cp.Problem(objective,constraints)
            
            problem.solve(eps = tol_solver)

            min_eig = np.min(np.linalg.eigvalsh(Sigma.value))
            if min_eig<tol_psd:
                u=(tol_psd-min_eig)/(1-min_eig)
            else:
                u=0

            Sigma.value = (1-u)*Sigma.value + u*np.eye(n)

            value,grad = oracle(Sigma.value)

            best_lower_bound = np.maximum((cp.trace(self._coeff_matrix@Sigma[:self.d+1,:self.d+1]) - ep*value).value, best_lower_bound)

            if verbose: print("Iter", i, "-- Value: ",objective.value,"-- Certificate: ",objective.value-best_lower_bound)
            
            if objective.value-best_lower_bound < tol_kelley:
                if verbose: print('Convergence reached')
                break

            list_value.append(value - np.trace(grad@Sigma.value))
            list_grad.append(grad)
        
        if i == max_iter-1:
            warnings.warn(f'Did not reached convergence. Gap: {np.abs(objective.value-best_lower_bound):.2e}')

        self.log_partition = problem.value
        self.entropy = -value
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

    complete_graph_features = [{i} for i in range(1,d+1)] + [{i,j} for i in range(1,d+1) for j in range(i+1,d+1)]
    coefficients = random_coefficients_gaussian(complete_graph_features)
    exact_inference = ExactBruteForce(
        d=d,
        coefficients=coefficients,
        features=features_0)
    exact_inference.solve()
    quantum_inference = QuantDiagRelaxation(
        d=d,
        coefficients=coefficients,
        features=features_0)
    quantum_inference.solve(verbose=True)
    print("True logp     :",exact_inference.log_partition)
    print("Logp          :",quantum_inference.log_partition)
    print("True entropy  :",exact_inference.entropy)
    print("Entropy       :",quantum_inference.entropy)
    print("True marginals:\n",exact_inference.marginals)   
    print("Marginals     :\n",quantum_inference.marginals)