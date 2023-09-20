## 📦pycuga

Pycuga (PYthon CUda Genetic Algorithm) provides a simple and easy package for performing island-based genetic algorithm using Python and Cuda.

### Motivation
- When I worked on my previous project on [Solving Maximum Satisfiability Problem using CUDA](https://github.com/issacto/cuda-maxsat), I realised a lot of code could be reused, which save a lot of development time for solving other optimisation problems using genetic algorithm and CUDA. 


### Parameters

| Methods currently supported |  | 
| ------------- |:-------------:|
| Selection     | Elitism, Roulette Wheel | 
| Crossover     | Single, Double, Uniform|
| Mutation      | Number     |



```
pip install pycuga
```

```python
p1 = PyCUGA( isTime, time , constArr, chromosomeSize, stringPlaceholder,mutationNumber , selectionMode, crossoverMode)
p1. launchKernel(islandSize , blockSize , chromosomeNo, migrationRounds,rounds)
```

Examples [here](https://github.com/issacto/PyCuGa/tree/main/samples).


### Limitations
* Use multiples of 32 (for chromosome parameters) to avoid bugs and increase efficiency.
* Island migration is limited to 1 item currently
* Lack unit testing


### ❗Disclaimer
This is a mini project which I've put quite a lot of time and effort into, but I can't take responsibility for any bugs nor errors.
