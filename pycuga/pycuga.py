import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.driver as drv
import time
import os
import sys
sys.path.append("./algos")
import algos.tools as tools
import math
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom




class PyCUGA:
    def __init__(self, crossoverMode, mutationMode, selectionMode, evaluationMode, migrationNo, isTime, time, rounds, constArr, chromosomeSize):
        self.crossoverMode = crossoverMode
        self.mutationMode = mutationMode
        self.selectionMode = selectionMode
        self.evaluationMode = evaluationMode
        self.migrationNo = migrationNo
        self.isTime = isTime
        self.time = time
        self.rounds = rounds
        self.constArr=constArr
        self.dev = drv.Device(0)
        self.ctx = self.dev.make_context()
        self.ulonglongRequired = math.ceil(chromosomeSize/64)
        self.chromosomeSize=chromosomeSize
        # declare global CUDA functions
        cudaCode = tools.read_files_as_strings()
        mod = SourceModule(cudaCode)
        self.crossover = mod.get_function("vector_add_kernel")
        self.mutation = mod.get_function("vector_add_kernel")
        self.selection = mod.get_function("vector_add_kernel")
        self.evaluation = mod.get_function("vector_add_kernel")

    def launchKernel(self, threadSize, gridSize):
        chromosomes= np.array(np.random.randint(18446744073709551615, size=(threadSize*self.ulonglongRequired)),dtype=np.ulonglong)
        chromosomesResults= np.array(np.random.randint(0, size=threadSize),dtype=np.ulonglong)
        chromosomes_gpu = drv.to_device(chromosomes.nbytes)
        chromosomesResults_gpu = drv.mem_alloc(chromosomesResults.nbytes)
        rng = curandom.XORWOWRandomNumberGenerator()
        roundCount = 0
        while (self.isTime) or (not self.isTime and roundCount<self.rounds):
            # set up
            random_crossover_index_cpu = np.random.randint(0, 123, size=array_size, dtype=np.uint32)
            random_crossover_index_gpu = gpuarray.to_gpu(random_crossover_index_cpu)
            random_crossover_length_cpu = np.random.randint(0, 123, size=array_size, dtype=np.uint32)
            random_crossover_length_gpu = gpuarray.to_gpu(random_crossover_length_cpu)
            random_mutation_prob_cpu = np.random.randint(0, 123, size=array_size, dtype=np.uint32)
            random_mutation_prob_gpu = gpuarray.to_gpu(random_mutation_prob_cpu)
            random_mutation_index_cpu = np.random.randint(0, 123, size=array_size, dtype=np.uint32)
            random_mutation_index_gpu = gpuarray.to_gpu(random_mutation_index_cpu)
            



            roundCount +=1
        
        print("DONE")

        
