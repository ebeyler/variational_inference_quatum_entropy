"""Base class of algorithm
"""
import numpy as np

class InferenceAlgorithm:
    def __init__(self):
        self.d = None
        self.coefficients = None
        self.marginals = None #[d,x_d]
        self.log_partition = None
        self.entropy = None
    
    def solve():
        raise NotImplementedError
    
    def l1_error(self,exact_marginals):
        if self.marginals is None:
            raise RuntimeError(
                "You must first run algorithm by calling _.solve()."
                )
        return np.mean(np.abs(self.marginals[:,0] - exact_marginals[:,0]))