# 📦PYthon CUda Genetic Algorithm - PYCUGA

PyCuGa provides a simple and easy package for performing island-based genetic algorithm using Python and Cuda.


## Variables

| Methods currently supported |  |
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

## Limitations
* Use multiples of 32 (for chromosome parameters)to avoid bugs and increase efficiency.
* Island migration is limited to 1 item currently
* Lack unit testing


## ❗Disclaimer
This is a mini project which I've put quite a lot of time and effort into, but I can't take responsibility for any bugs nor errors.
