import numpy as np

def random_coefficients_gaussian(graph_features:list[set],
                                 loc:float = 0.,
                                 scale:float = 1.) -> list[tuple[set,float]]:
    """Generate random normal coefficients for each feature in graph_features

    Args:
        graph_features (list[set]): Vertex and edges for which to generate
            coefficient, represented as sets
        loc (float, optional): Mean of the gaussian. Defaults to 0..
        scale (float, optional): Scale of the gaussian. Defaults to 1..

    Returns:
        list[tuple[set,float]]: list of generated coefficients, as
            (feat,coef), where feat (set) represents a feature and
            coef (float) the associated coefficient
    """
    return [(feat,np.random.normal(loc=loc,scale = scale)) for feat in graph_features]

def ramdom_coefficient_TRW(d:int,
                           strenght:float,
                           graph:str,
                           interaction:str) -> list[tuple[set,float]]:
    """Generates random coefficients as in [1]

    Args:
        d (int): if graph == "complete", number of variable,
            if graph == "grid", side of the gird
        strenght (float): strenght of the interaction
        graph (str): type of graph, either "grid" or "complete"
        interaction (str): type of interaction, either "mixed" or 
            "attractive".

    Returns:
        list[tuple[set,float]]: list of generated coefficients, as
            (feat,coef), where feat (set) represents a feature and
            coef (float) the associated coefficient
    
    Reference:
        [1] M.J. Wainwright, T.S. Jaakkola, and A.S. Willsky. “A New Class of
            Upper Bounds on the Log Partition Function”. In: IEEE Transactions
            on Information Theory 51.7 (July 2005), pp. 2313–2335.
    """
    if graph == "complete":
        edges = [{i,j} for i in range(1,d+1) for j in range(i+1,d+1)]
    elif graph == "grid":
        edges = []
        for i in range(1,d):
            for j in range(1,d):
                edges.append({(i-1)*d + j,(i-1)*d+ (j+1)})
                edges.append({(i-1)*d + j,i*d + j})
        for i in range(1,d):
            edges.append({(i-1)*d + d,i*d + d})
        for j in range(1,d):
            edges.append({(d-1)*d + j,(d-1)*d+ j+1})
    else:
        raise ValueError(
            'graph should be either "complete" or "grid"'
        )
    
    coefficients = [({i},np.random.uniform(-.05,.05)) for i in range(1,d+1)]

    if interaction == "attractive":
        coefficients += [(edge,np.random.uniform(0,strenght)) for edge in edges]
    elif interaction == "mixed":
        coefficients += [(edge,np.random.uniform(-strenght,strenght)) for edge in edges]
    else:
        raise ValueError(
            'interaction should be either "attractive" or "mixed"'
        )    
    return coefficients

def ramdom_coefficient_logdet(d:int,
                           strenght:float,
                           graph:str,
                           interaction:str) -> list[tuple[set,float]]:
    """Generates random coefficients as in [1]

    Args:
        d (int): if graph == "complete", number of variable,
            if graph == "grid", side of the gird
        strenght (float): strenght of the interaction
        graph (str): type of graph, either "grid" or "complete"
        interaction (str): type of interaction, either "mixed" or 
            "attractive" or "repulsive".

    Returns:
        list[tuple[set,float]]: list of generated coefficients, as
            (feat,coef), where feat (set) represents a feature and
            coef (float) the associated coefficient
    
    Reference:
        [1] Michael Jordan and Martin J Wainwright. “Semidefinite Relaxations
            for Approximate Inference on Graphs with Cycles”. In: Advances in
            Neural Information Processing Systems. Vol. 16. MIT Press, 2003.    
    """
    if graph == "complete":
        edges = [{i,j} for i in range(1,d+1) for j in range(i+1,d+1)]
    elif graph == "grid":
        edges = []
        for i in range(1,d):
            for j in range(1,d):
                edges.append({(i-1)*d + j,(i-1)*d+ (j+1)})
                edges.append({(i-1)*d + j,i*d + j})
        for i in range(1,d):
            edges.append({(i-1)*d + d,i*d + d})
        for j in range(1,d):
            edges.append({(d-1)*d + j,(d-1)*d+ j+1})
    else:
        raise ValueError(
            'graph should be either "complete" or "grid"'
        )
    
    coefficients = [({i},np.random.uniform(-.25,.25)) for i in range(1,d+1)]

    if interaction == "attractive":
        coefficients += [(edge,np.random.uniform(0,2*strenght)) for edge in edges]
    elif interaction == "mixed":
        coefficients += [(edge,np.random.uniform(-strenght,strenght)) for edge in edges]
    elif interaction == "repulsive":
        coefficients += [(edge,np.random.uniform(-2*strenght,0)) for edge in edges]
    else:
        raise ValueError(
            'interaction should be either "attractive" or "mixed"' or "repulsive"
        )    
    return coefficients