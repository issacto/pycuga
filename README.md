## Python Cuda Genetic Algorithm Package


<!--  
python setup.py bdist_wheel  
 python -m twine upload dist/*
-->


```python
p1 = PyCUGA( mutationThreshold = 0.1, isTime = False, time = 0, constArr = "", chromosomeSize = 18432, evaluationString = "")
p1. launchKernel(islandSize = 32, blockSize = 128, chromosomeNo = 18432, migrationRounds = 20,rounds = 100)

```