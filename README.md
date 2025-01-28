# Variational Inference on the Boolean Hypercube with the Quantum Entropy

Code associated with the paper "Variational Inference on the Boolean Hypercube with the Quantum Entropy".

## Installation

To install the package, run:
````
pip install '.[exp]'
````

(If you only want to use the algorithms without running the experiments, you can simply run `pip install .`)

## Organization
The folders are organized as follow:
- `experiments`: code to reproduce the figures in the paper.
- `quant_inf`: implementation of the algorithms and tools.

## How to use the algorithms?
- Vertices are represented as python sets of size 1 and edges as sets of size two.
- A features $x^\alpha$ is represented by a python set, that contain the integers $i$ for which $\alpha_i =1$. For exemple, the feature $x_1x_2$ is represented by the set `{1,2}`, and the constant feature $x^0 = 1$ is represented by the empty set `set()`.
- A feature vector is represented as a list of features, i.e. a list of sets. For exemple, the feature vector $\varphi(x) = (1,x_1,x_2,x_3)$ is represented by the list `[set(),{1},{2},{3}]`.
- A graph is represented as a list of vertices and edges, i.e. a list of sets. For exemple, the graph 1-2-3 is represented by the list `[{1},{2},{3},{1,2},{2,3}]`.
- A parametrized model is represented as a list of tuple of `(feat,coef)`, where `feat` is a feature (vertex or edge) in a graph and `coef` the corresponding coefficient. For exemple, a model on the graph 1-2-3 could be represented by the list `[({1},0),({2},-1.5),({3},.4),({1,2},2),({2,3},-.7)]`.
- To run the quantum inference, you must first initialize a quantum inference model `QuantumRelaxation` with `d`the number of variables, `coefficients` the coefficients of the model and `features` the feature vector $\varphi(x)$, and call `QuantumRelaxation.solve()`. You can then access to the quantum bound on the log-partition function `QuantumRelaxation.log_partition` or the approximated marginals `QuantumRelaxation.marginals`.  For example:
````python
from quant_inf.algorithms import QuantumRelaxation
d = 3
coefficients = [({1},0),({2},-1.5),({3},.4),({1,2},2),({2,3},-.7)]
feature_vect = [set(),{1},{2},{3}]

quantum_inference = QuantumRelaxation(
    d=d,
    coefficients=coefficients,
    features=feature_vect
    )

quantum_inference.solve()
print(quantum_inference.log_partition)
````

- the inference models implemented are:
    - `ExactBruteForce`: compute the log-partition function, the moment matrix $\Sigma_p$ and marginals by brute force.
    - `QuantumRelaxation`: quantum relaxation of the log-partition function 
    - `QuantDiagRelaxation`: quantum relaxation with diagonal metric learning
    - `QuantGreedyRelaxation`:  quantum relaxation with greedy selection of features (specify the number of extra features to select with `QuantGreedyRelaxation.solve(number_extra_features:int)`)
    - `LogDetRelaxation`: log-determinant relaxation from [1]
    - `TRWRelaxation`: tree-reweighted message passing (TRW) relaxation from [2]

- note that all features vectors must start with features `set(),{1},...,{d}` in order to represent the function $f(x)$ as a quadratic form in $\varphi(x)$.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## References 

[1] Michael Jordan and Martin J Wainwright. “Semidefinite Relaxations for Approximate Inference on Graphs with Cycles”. In: Advances in Neural Information Processing Systems. Vol. 16. MIT Press, 2003.

[2] M.J. Wainwright, T.S. Jaakkola, and A.S. Willsky. “A New Class of Upper Bounds on the Log Partition Function”. In: IEEE Transactions on Information Theory 51.7 (July 2005), pp. 2313–2335.