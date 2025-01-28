"""Implementation of the TRW algorithm to compute relaxation of the
log-partition function and of marginals.
"""

import numpy as np
import warnings
from scipy.special import logsumexp
from quant_inf.algorithms.base_class import InferenceAlgorithm
from quant_inf.tools import random_coefficients_gaussian

class TRWRelaxation(InferenceAlgorithm):
    """Implement the TRW algorithm for approximate inference
    on the Ising model.

    Attributes:
        d (int): number of variables
        coefficients (list[tuple[set,float]]): list of coefficients,
            as (inf,coef), where ind (set[int]) represents a feature of
            the model (either an edge or a vertex) and coef (float)
            the associated coefficient
        logM (np.ndarray): matrix of log messages [s,t,x_s]
        rho (np.ndarray): matrix of edge appearance probabilities [s,t]
        marginals (np.ndarray): marginals [s,x_s]
        pairwise_marginals (np.ndarray): pairwise_marginals [s,t,x_s,x_t]
        log_partition (float): TRW upper bound on the log-partion
        entropy (float): entropy
        eps (float): temperature parameter $\\varepsilon$. Defaults to 1..

    Note:
        To access attributes logM, rho, marginals, pairwise_marginals, 
        entropy & log_partition, you must first solve the model by
        calling TRW.solve()
    
    Reference:
        [1] M.J. Wainwright, T.S. Jaakkola, and A.S. Willsky. “A New Class of
            Upper Bounds on the Log Partition Function”. In: IEEE Transactions
            on Information Theory 51.7 (July 2005), pp. 2313–2335.
    """
    def __init__(self,
                 d:int=None,
                 coefficients:list[tuple[set,float]]=None,
                 eps:float=1.
                 ) -> None:
        super().__init__()
        self.d = d
        self.coefficients = coefficients
        self.eps = eps
        
        self.logM = None
        self.rho = None
        self.marginals = None
        self.pairwise_marginals = None
        self.log_partition = None
        
        self._x = np.ones((self.d,2))#[s,x_s]
        self._x[:,1] = -1
    
    @property
    def coefficients(self):
        return self._coefficients
    
    @coefficients.setter
    def coefficients(self,value):
        self._coefficients = value
        if value is not None:
            self._graph_edges = [feat for feat,_ in value if len(feat)==2]
            self._theta_1,self._theta_2 = self._create_theta()
            # theta_1: coefficients on vertices [s], theta_2: coefficients on edges [s,t]

    def solve(self,
              damping_factor_message_passing:float=.4,
              max_iter_message_passing:int=10000,
              tol_message_passing:float=1.e-8,
              max_iter_optim_rho= 1000,
              alpha_0 = .5,
              back_track_param1 =.1,
              back_track_param2 = .1,
              min_value_alpha:float=1.e-8,
              tol_optim_rho=1.e-2,
              logs:bool=False) -> None:
        """Compute the TRW relaxation of the log-partition function as
        introduced in [1], with optimization of the matrix of edge
        appearance probabilities by Frank-Wolfe algorithm
        (algorithm 2 of [1]).

        Set the value of the following attributes: logM, rho, marginals,
        pairwise_marginals & log_partition

        Back-tracking linesearch is:
        alpha <- back_track_param1*alpha
        until:
            f((1-alpha)*rho+ alpha*rho_desc) < (f(rho) - back_track_param1*alpha*scalar_prod)
        where scalar_prod = <grad|rho_desc - rho>
        
        Args:
            damping_factor_message_passing (float,optionnal): damping factor for
                message passing updates. Defaults to .4.
            max_iter_message_passing (int, optional): Maximal number of
                iterations for the message passing algorithm (algorithm 1
                of [1]). Defaults to 2000.
            tol_message_passing (float,optional): Stopping criteria for
                algorithm 1. Defaults to 1.e-8.
            max_iter_optim_rho (int, optional): Maximal number of iterations
                for Frank-Wolfe algorithm (algorithm 2). Defaults to 1000.
            alpha_0 (float, optional): Initial value of alpha for backtracking
                line-search (algorithm 2). Defaults to .5.
            back_track_param1 (float, optional): Parameter of the backtracking
                line-search. Defaults to .1.
            back_track_param2 (float, optional): Parameter of the backtracking
                line-search. Defaults to .1.
            min_value_alpha (float, optional): minimal value of alpha in the 
                line-search. If reached, a warning will be raised. Defaults
                to 1.e-8.
            tol_optim_rho (float, optional): Stopping criteria for algorithm 2.
                Defaults to 1.e-2.
            logs (bool, optional): if True, return logs of values along
                iterations. Defaults to False.

        Returns:
            tuple[list,list,list,list]: if logs == True, return logs of values
                (logs_logp,logs_alpha,logs_rho,logs_scalar_prod) along iterations
        """

        rho = self._initialise_rho()

        logM = self._TRW_message_passing(
            rho=rho,
            max_iter=max_iter_message_passing,
            gamma=damping_factor_message_passing,
            tol=tol_message_passing
            )
        logp = self._logp_TRW(logM,rho)

        logs_logp = [logp]
        logs_alpha = [np.nan]
        logs_rho = [rho.copy()]
        logs_scalar_prod = [np.nan]

        for i in range(max_iter_optim_rho):
            # Descent-direction
            I=self._information_vect(logM,rho)
            information_graph=[(set(feat),I[feat[0]-1,feat[1]-1]) for feat in [list(feat) for feat in self._graph_edges]]
            # weighted graph associated with I
            rho_desc=self._mat_rho(
                [(feat,1) for feat,_ in self._kruskal(information_graph)]
                )
            I=np.nan_to_num(I,nan=0) # value doesn't matter because will be multiply by 0
            scalar_prod = np.sum(I*(rho_desc-rho))
            
            #stopping criterion:
            if scalar_prod<tol_optim_rho:
                break

            # Back-tracking line search
            alpha=alpha_0
            logM=self._TRW_message_passing(
                rho=(1-alpha)*rho+ alpha*rho_desc,
                max_iter=max_iter_message_passing,
                gamma=damping_factor_message_passing,
                tol=tol_message_passing,
                logM_init=logM
                )
            logp_alpha=self._logp_TRW(logM=logM,rho=(1-alpha)*rho+ alpha*rho_desc)
            while logp_alpha>(logp-back_track_param1*alpha*scalar_prod):
                alpha=back_track_param2*alpha
                if alpha < min_value_alpha:
                    warnings.warn(
                        "Line-search has failed as alpha reached min_value_alpha. "
                        "It may be caused by non convergence of message passing. "
                        "Optimization of rho is stopped without reaching convergence."
                    )
                    break
                logM=self._TRW_message_passing(
                    rho=(1-alpha)*rho+ alpha*rho_desc,
                    max_iter=max_iter_message_passing,
                    gamma=damping_factor_message_passing,
                    tol=tol_message_passing,
                    logM_init=logM
                    )
                logp_alpha=self._logp_TRW(logM=logM,rho=(1-alpha)*rho+alpha*rho_desc)
            
            if alpha < min_value_alpha:
                """ stopping the optimization of rho if warning on
                alpha has been raised
                """ 
                break
            
            rho=(1-alpha)*rho+alpha*rho_desc
            logp=logp_alpha 

            #Logging values 
            if logs is True: 
                logs_scalar_prod.append(scalar_prod) 
                logs_rho.append(rho.copy())
                logs_alpha.append(alpha)
                logs_logp.append(logp)
        
        if i == (max_iter_optim_rho-1):
            warnings.warn(
                "Optimization of rho didn't reached convergence. The bound may be sub-optimal"
                )

        self.logM=logM
        self.rho=rho
        self.marginals=self._get_marginals(logM,rho)
        self.pairwise_marginals=self._get_pairwise_marginals(logM,rho)
        self.log_partition=logp

        if logs is True:
            return logs_logp,logs_alpha,logs_rho,logs_scalar_prod
        
    def solve_rho_fixed(self,
                        rho:np.ndarray,
                        damping_factor_message_passing:float=.4,
                        max_iter_message_passing:int=10000,
                        tol_message_passing:float=1.e-8) -> None:
        """Compute the TRW relaxation of the log-partition function as
        introduced in [1], with fixed value of the matrix of edge
        appearance probabilities rho.

        Set the value of the following attributes: logM, rho, marginals,
        pairwise_marginals & log_partition
        
        Args:
            damping_factor_message_passing (float,optionnal): damping factor for
                message passing updates. Defaults to .4.
            max_iter_message_passing (int, optional): Maximal number of
                iterations for the message passing algorithm (algorithm 1
                of [1]). Defaults to 10000.
            tol_message_passing (float,optional): Stopping criteria for
                algorithm 1. Defaults to 1.e-8.
        """

        logM = self._TRW_message_passing(
            rho=rho,
            max_iter=max_iter_message_passing,
            gamma=damping_factor_message_passing,
            tol=tol_message_passing
            )
        logp = self._logp_TRW(logM,rho)
        
        self.logM=logM
        self.rho=rho
        self.marginals=self._get_marginals(logM,rho)
        self.pairwise_marginals=self._get_pairwise_marginals(logM,rho)
        self.log_partition=logp
    
    def _create_theta(self):
        """From self.coefficients, returns the vector theta_1
        containing the coefficient on the vertices, and the
        symetric matrix theta_2 containing the coefficients on
        the edges.

        Raises:
            ValueError: raised if any feature is not an edge or a vertex
        
        Returns:
            tuple[np.ndarray,np.darray]: theta_1 [s] , theta_2 [s,t]
        """
        F = np.zeros((self.d+1,self.d+1))
        for feat,coef in self.coefficients:
            if len(feat) == 1:
                i, = feat
                F[0,i],F[i,0] = coef,coef
            elif len(feat) == 2:
                i,j = feat
                F[j,i],F[i,j] = coef,coef
            else:
                raise ValueError("Only support MRF")
        return(F[0,1:],F[1:,1:])
    
    def _initialise_rho(self,n:int=1000)->np.ndarray:
        """Initialise a matrix of edge appearance probabilities
        by drawing random spanning trees of the graph.

        Args:
            n (int, optional): number of spanning trees to draw.
                Defaults to 1000.

        Returns:
            np.ndarray: matrix of edge appearance probabilities
        """
        list_rho = []
        for i in range(n):
            list_rho.append(self._mat_rho(
                [(feat,1) for feat,_ in self._kruskal(random_coefficients_gaussian(self._graph_edges))]
                ))
        return np.mean(np.array(list_rho), axis=0)
    
    def _mat_rho(self,weighted_edges:list[tuple[set,float]]) -> np.ndarray:
        """Construct a symmetric matrix rho representing the edge
        appearance probabilities, for a list of those edge appearance
        probabilities, on a graph with self.d variables

        Args:
            weighted_edges (list[tuple[set,float]]): list of weighted edges,
                as (inf,coef), where ind (set of len 2) represents an edges
                of the graph and coef the associated weight

        Raises:
            ValueError: if coefficents contains elements
            that don't represent weighted edges.
        """
        rho = np.zeros((self.d,self.d))
        for feat,coef in weighted_edges:
            if len(feat) == 2:
                i,j = feat
                rho[j-1,i-1],rho[i-1,j-1] = coef,coef
            else:
                raise ValueError(
                    "weighted_edges should only contain weighted EDGES"
                    )
        return(rho)
    
    @staticmethod
    def _kruskal(weighted_edges:list[tuple[set,float]])->list[tuple[set,float]]:
        """kruskal algorithm to find a maximum spanning tree
        /!\\ this implementation is quite naive and not optimized

        Args:
            weighted_edges (list[tuple[set,float]]): list of weighted edges,
                as (inf,coef), where ind (set of len 2) represents an edges
                of the connected graph and coef the associated weight
        
        Returns:
            list(list[tuple[set,float]]): weighted edges corresponding
                to the maximal weight spanning tree
        """
        sorted_edges = sorted(weighted_edges,
                            key= lambda x: np.nan_to_num(x[1],nan=-np.inf),
                            reverse=True)
        connected_compt = [] # list of the current connected components
        tree = [] # list of current selected edges

        for edge in sorted_edges:
            compt_with_edge = [] # list of connected components that contains a vertex of edge
            compt_without_edge = [] # list of connected components that doesn't contain any vertex of edge
            create_cycle = False
            for compt in connected_compt:
                len_intersect = len(edge[0]&compt)
                if len_intersect == 2:
                    create_cycle = True
                    break
                elif len_intersect == 1:
                    compt_with_edge.append(compt)
                else:
                    compt_without_edge.append(compt)
            
            if create_cycle:
                continue
            else:
                tree.append(edge)
                new_connex_compt=edge[0]
                for compt in compt_with_edge:
                    new_connex_compt = new_connex_compt | compt
                connected_compt = compt_without_edge + [new_connex_compt]
        return(tree)
    
    def _TRW_message_passing(self,rho,max_iter,gamma,tol,logM_init=None):
        """Compute matrix of log messages logM from algo. 1 of [1], cast in the
        log domain.
        Update are damped by factor gamma : logM_t+1 = (1-gamma)*logM_t + gamma*logM_step
        Stopping criteron:  max_ij(|logM_t - logM_step|_ij) < tol

        Args:
            rho (np.ndarray): matrix of edge appearance probabilities [s,t]
            max_iter (int): maximal number of iterations
            gamma (float): Damping parameter.
            tol (float):  Stopping criteron.
            logM_init (np.ndarray): warn start if not None. Defaults to None.

        Returns:
            np.ndarray: matrix of messages M [s,t,x_s]
        """
        if logM_init is not None:
            logM = np.nan_to_num(logM_init, nan=0)
        else:
            logM = np.zeros((self.d,self.d,2))
        for i in range(max_iter):
            with np.errstate(invalid="ignore"):
                arg_exp = (
                        self._x[:,None,:,None]*self._theta_2[:,:,None,None]*self._x[None,:,None,:]/(rho[:,:,None,None]*self.eps)
                        + (self._theta_1[:,None]* self._x[:,:])[None,:,None,:]/self.eps
                        + (np.sum(np.nan_to_num(logM,nan=0)*rho[:,:,None],axis=0,keepdims=True) - logM)[:,:,None,:]
                    ) #[t,s,x_s,x_t]
            logM_step = logsumexp(arg_exp ,axis=-1)
            logM_step = logM_step - logsumexp(logM_step, axis = -1,keepdims=True)
            logM_step = np.swapaxes(logM_step, 0, 1)
            logM_step = logM*(1-gamma) + logM_step*gamma
            
            #stopping criteron
            if np.max(np.abs(np.nan_to_num(logM,nan=0)-np.nan_to_num(logM_step,nan=0))) < tol:
                logM = logM_step
                break
            logM = logM_step
        if i == max_iter-1:
            warnings.warn(
                "Message passing algorithm didn't reached convergence, the solution could be inacurrate."
                )
        return logM

    def _information_vect(self,logM,rho):
        """Matrix of mutual information between variables x_s and x_t,
        computed for matrix of edge appearance probabilities rho and 
        corresponding matrix of log message logM (and parameters of the model)
        /!\\ mutual information [s,t] is only valid if rho_s,t > 0, (otherwize nan)

        Args:
            logM (np.ndarray): matrix of log messages [s,t,x_s]
            rho (np.ndarray): matrix of edge appearance probabilities [s,t]

        Returns:
            np.ndarry: matrix of mutual information [s,t]
        """
        pairwize_marginal = self._get_pairwise_marginals(logM,rho)
        I = (
            pairwize_marginal
            *np.log(
                pairwize_marginal
                /(np.sum(pairwize_marginal, axis= 2,keepdims=True)*np.sum(pairwize_marginal, axis= 3,keepdims=True)))
        )
        I = np.sum(I,axis=(2,3))*self.eps
        return I #[s,t]
    
    def _get_pairwise_marginals(self,logM,rho):
        """Return pairwize marginals,
        computed for matrix of edge appearance probabilities rho and 
        corresponding matrix of log message logM (and parameters of the model)
        /!\\ pairwize marginal [s,t] is only valid if rho_s,t > 0, (otherwize nan)

        Args:
            logM (np.ndarray): matrix of log messages [s,t,x_s]
            rho (np.ndarray): matrix of edge appearance probabilities [s,t]

        Returns:
            np.ndarray: pairwise_marginals [s,t,x_s,x_t]
        """
        with np.errstate(invalid="ignore"): # will have nan for indices for which rho = 0
            pairwise_marginals = np.exp(
                self._x[:,None,:,None]*self._theta_2[:,:,None,None]*self._x[None,:,None,:]/(rho[:,:,None,None]*self.eps)
                +(self._theta_1[:,None]* self._x)[:,None,:,None]/self.eps
                +(self._theta_1[:,None]* self._x)[None,:,None,:]/self.eps
                +np.sum(np.nan_to_num(logM,nan=0)*rho[:,:,None],axis=0)[:,None,:,None]
                +np.sum(np.nan_to_num(logM,nan=0)*rho[:,:,None],axis=0)[None,:,None,:]
                -(logM[:,:,None,:]+np.swapaxes(logM,0,1)[:,:,:,None])
                ) #[s,t,x_s,x_t]
        pairwise_marginals = pairwise_marginals/np.sum(pairwise_marginals, axis = (2,3),keepdims=True)
        return pairwise_marginals #[s,t,x_s,x_t]

    def _get_marginals(self,logM,rho):
        """Return marginals,
        computed for matrix of edge appearance probabilities rho and 
        corresponding matrix of log messages (and parameters of the model)

        Args:
            logM (np.ndarray): matrix of log messages [s,t,x_s]
            rho (np.ndarray): matrix of edge appearance probabilities [s,t]

        Returns:
            np.ndarray: marginals [s,x_s]
        """
        marginals = np.exp(
            self._theta_1[:,None]*self._x/self.eps
            + np.sum(np.nan_to_num(logM,nan=0)*rho[:,:,None],axis=0)
        )
        marginals = marginals/np.sum(marginals, axis = -1,keepdims=True)
        return marginals #[s,x_s]
    
    def _logp_TRW(self,logM,rho):
        """Upper bound on logp from TRW relaxation,
        computed for matrix of edge appearance probabilities rho and 
        corresponding matrix of message M (and parameters of the model)
        /!\\ is valid only if rho_s,t > 0 for all s,t such that theta_s,t > 0

        Args:
            logM (np.ndarray): matrix of log messages [s,t,x_s]
            rho (np.ndarray): matrix of edge appearance probabilities [s,t]

        Returns:
            float:  bound on logp from TRW relaxation
        """
        if np.any(np.logical_and(self._theta_2>0,rho==0)):
            warnings.warn(
                "The upper-bound on the log-partition is not valid as there exist s,t such that rho_s,t = 0 and theta_s,t > 0 "
                )

        marginals = self._get_marginals(logM,rho)
        pairwize_marginals = np.nan_to_num(self._get_pairwise_marginals(logM,rho),nan=1) # value doesn't matter
        scalar_prod = (
            np.sum(self._theta_1[:,None]*self._x*marginals)
            +np.sum(self._x[:,None,:,None]*self._theta_2[:,:,None,None]*self._x[None,:,None,:]*pairwize_marginals)/2
        )
        I=pairwize_marginals*np.log(pairwize_marginals/(marginals[:,None,:,None]*marginals[None,:,None,:]))
        I=np.sum(I,axis=(2,3))
        neg_entropy=np.sum(marginals*np.log(marginals))+(np.sum(rho*I)/2)+self.d*np.log(2)
        self.entropy = -neg_entropy #saving value of entropy
        return scalar_prod-self.eps*neg_entropy

if __name__ == '__main__':
    import time
    from quant_inf.tools import random_coefficients_gaussian
    from quant_inf.algorithms import ExactBruteForce
    import pandas as pd
    import matplotlib.pyplot as plt
    print("---------  Testing TRW algorithm  ---------")
    print("Coefficients are non-zero only on edges of a tree but algorithm is ignorant of that fact.")

    d = 5
    features_0 = (
        [set()] 
        + [{i} for i in range(1,d+1)]
    )

    chain_features =[{i} for i in range(1,d+1)] + [{i,i+1} for i in range(1,d)]

    coefficients = random_coefficients_gaussian(
        graph_features=chain_features,
        scale=.3)
    
    coefficients = coefficients +  [({i,j},0) for i in range(1,d+1) for j in range(i+2,d+1)]

    exact_inference = ExactBruteForce(
        d=d,
        coefficients=coefficients,
        features=features_0)
    exact_inference.solve()

    TRW_inference=TRWRelaxation(
        d=d,
        coefficients=coefficients)
    
    print("---------    Testing rho fixed    ---------")
    rho_fixed = 2*(np.ones((d,d)) - np.eye(d))/d
    start = time.perf_counter()
    TRW_inference.solve_rho_fixed(rho_fixed)
    print(f"Running time: {time.perf_counter()-start}")
    print(f"Estimate of logp: {TRW_inference.log_partition:.5f} ")

    print("---------  Testing rho optimized  ---------")

    start = time.perf_counter()
    logs_logp,logs_alpha,logs_rho,logs_scalar_prod=TRW_inference.solve(logs=True)
    print(f"Running time: {time.perf_counter()-start}")
    
    print("rho at initialization:\n",pd.DataFrame(np.round(logs_rho[0],decimals=6)))
    print("rho at terminaison:\n",pd.DataFrame(np.round(logs_rho[-1],decimals=6)))
    print(f"True value of logp: {exact_inference.log_partition:.5f} -- estimate with initial rho: {logs_logp[0]:.5f} -- estimate with final rho: {logs_logp[-1]:.5f} ")
    print("True entropy  :",exact_inference.entropy)
    print("Entropy       :",TRW_inference.entropy)
    print("True marginals:\n",exact_inference.marginals)   
    print("Marginals     :\n",TRW_inference.marginals)    

    print("---------  Testing with eps = 2.  ---------")
    print("---------  Testing rho optimized  ---------")
    exact_inference = ExactBruteForce(
        d=d,
        coefficients=coefficients,
        features=features_0,
        eps=2.)
    exact_inference.solve()

    TRW_inference=TRWRelaxation(
        d=d,
        coefficients=coefficients,
        eps=2.)
    
    start = time.perf_counter()
    logs_logp,logs_alpha,logs_rho,logs_scalar_prod=TRW_inference.solve(logs=True)
    print(f"Running time: {time.perf_counter()-start}")
    
    print("rho at initialization:\n",pd.DataFrame(np.round(logs_rho[0],decimals=6)))
    print("rho at terminaison:\n",pd.DataFrame(np.round(logs_rho[-1],decimals=6)))
    print(f"True value of logp: {exact_inference.log_partition:.5f} -- estimate with initial rho: {logs_logp[0]:.5f} -- estimate with final rho: {logs_logp[-1]:.5f} ")
    print("True entropy  :",exact_inference.entropy)
    print("Entropy       :",TRW_inference.entropy)
    print("True marginals:\n",exact_inference.marginals)   
    print("Marginals     :\n",TRW_inference.marginals) 

    #Plot
    # n_iter = len(logs_logp)
    # fig, axes = plt.subplots(3,1,figsize=(10,10),sharex=True)
    # axes[0].plot(range(n_iter),logs_logp)
    # axes[0].plot(range(n_iter),[exact_inference.log_partition]*n_iter)
    # axes[1].plot(range(n_iter),logs_alpha)
    # axes[1].set_yscale("log")
    # axes[2].plot(range(n_iter),logs_scalar_prod)
    # plt.show()

