## 📦pycuga

Pycuga (PYthon CUda Genetic Algorithm) provides a package for performing island-based genetic algorithm on Python and CUDA.
- Methods for migartion, selection and mutation are implemented already. Users only need to pick the method during initialisation.
- User needs to define evaluation using Cuda code.

### Motivation
- When I worked on my previous project on [Solving Maximum Satisfiability Problem using CUDA](https://github.com/issacto/cuda-maxsat), I realised a lot of code could be reused, which save a lot of development time for solving other optimisation problems using genetic algorithm and CUDA. 


### Parameters

| Methods currently supported |  | 
| ------------- |:-------------:|
| Selection     | Elitism, Roulette Wheel | 
| Crossover     | One (point), Two (points), Uniform|
| Mutation      | Number     |


```
pip install pycuga
```

```python
p1 = PyCUGA(constArr, chromosomeSize, stringPlaceholder, mutationNumber, selectionMode, crossoverMode)
p1. launchKernel(islandSize, blockSize , chromosomeNo, isTime, timeAllowed, migrationRounds, rounds)
```

- constArr (numpy array): defines the details of the problem
- chromosomeSize (int): number of variables in a chromosome 
- stringPlaceholder (string): defines the evalutaion method and other constants
- mutationNumber (int): /
- selectionMode (string): "elitism" or "roulettewheel"
- crossoverMode (string): "one" or "two" or "uniform"

- islandSize (int): size of an island
- blockSize (int): Cuda block size
- chromosomeNo (int): number of chromosomes
- migrationRounds (int): number of rounds per migration
- isTime (bool): stopped by time or number of rounds
- timeAllowed (int): seconds allowed
- rounds (int): total number of rounds

Examples [here](https://github.com/issacto/PyCuGa/tree/main/samples).

Colab [example](https://colab.research.google.com/drive/1D49YgmBiyiJ8zgygs4XXp123ySmc1yNf?usp=sharing).

### Limitations
* Use multiples of 32 (for chromosome parameters) to avoid bugs and increase efficiency.
* Island migration is limited to 1 item currently
* Lack unit testing


### ❗Disclaimer
This is a mini project which I've put quite a lot of time and effort into, but I can't take responsibility for any bugs nor errors.
