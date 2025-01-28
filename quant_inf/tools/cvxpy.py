def V_constraints(Sigma,features, verbose = False):
    """Generate CVXPY constaints for the moment matrix to be in V.

    These constraints are equality contraints between coefficients.
    To avoid redundancy, coefficient above the diagonal are walked through in
    lexicographical order, and equality constraint is only with the first 
    coefficient following  in lexicographical order and verifing same
    symmetric difference between features.

    Args:
        Sigma (cvxpy.expressions.variable.Variable): moment matrix  
        features (list[set]): list of features (set) defining the vector $\\varphi$
        verbose (bool, optional): Defaults to False.

    Returns:
        list: list of CVXPY linear constraints on Sigma
    """
    n = len(features)
    indices = [(i,j) for i in range(n) for j in range(i,n)]
    constraints = []
    for k in range(len(indices)):
        if verbose : print("Searching for",indices[k])
        i1,j1 = indices[k]
        for l in range(k+1,len(indices)):
            i2,j2 = indices[l]
            if (features[i1] ^ features[j1]) == (features[i2] ^ features[j2]):
                constraints.append(Sigma[i1,j1]==Sigma[i2,j2])
                if verbose : print("Found:",indices[l])
                break
    if verbose : print("Number of constraints:", len(constraints))
    return constraints

def pairwise_edges_constraints(Sigma,features:list[set],d:int):
    """Return CVXPY constraints corresponding to Eq. (10) in [1]

    Args:
        Sigma (_type_):  moment matrix  
        features (list[set]): list of features (set) defining the vector $\\varphi$
        d (int): number of variables

    Returns:
        list: list of CVXPY linear constraints on Sigma
    
    Reference:
        [1] Michael Jordan and Martin J Wainwright. “Semidefinite Relaxations
            for Approximate Inference on Graphs with Cycles”. In: Advances in
            Neural Information Processing Systems. Vol. 16. MIT Press, 2003.
    """
    constraints = []
    for s in range(d):
        for t in range(s,d):
            for a in (-1,1):
                for b in (-1,1):
                    constraints.append(1 + a*Sigma[0,s+1] + b*Sigma[0,t+1] + a*b*Sigma[s+1,t+1] >= 0)
    return constraints
