## PYthon CUda Genetic Algorithm

PyCuGa provides a simple and easy package for performing island-based genetic algorithm using Python and Cuda. 

# Variables

| Method  |  |
| ------------- |:-------------:|
| Selection     | Elitism |
| Crossover     | Single, Double |
| Mutation      | Number     |


```
pip install pycuga
```

```python
p1 = PyCUGA( mutationThreshold = 0.1, isTime = False, time = 0, constArr = "", chromosomeSize = 18432, evaluationString = "")
p1. launchKernel(islandSize = 32, blockSize = 128, chromosomeNo = 18432, migrationRounds = 20,rounds = 100)

```
