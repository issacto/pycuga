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
    def __init__(self, mutationThreshold, isTime, time, rounds, constArr, chromosomeSize, evaluationString):
        self.isTime = isTime
        self.time = time
        self.rounds = rounds
        self.constArr=constArr
        self.dev = drv.Device(0)
        self.ctx = self.dev.make_context()
        self.ulonglongRequired = math.ceil(chromosomeSize/64)
        self.chromosomeSize=chromosomeSize
        self.mutationThreshold = mutationThreshold
        # declare global CUDA functions
        cudaCode = tools.read_files_as_strings()

        mod = SourceModule((cudaCode+evaluationString).replace("ULONGLONGREQUIREDVALUE", str(self.ulonglongRequired)))

        self.crossover = mod.get_function("crossover")
        self.mutation = mod.get_function("mutation")
        self.selection = mod.get_function("selection")
        self.evaluation = mod.get_function("evaluation")
        self.internalReOrder = mod.get_function("internalReOrder")
        self.migration = mod.get_function("migration")

    def launchKernel(self, islandSize, blockSize, chromosomeNo, migrationRounds):
        parentsGridSize = (chromosomeNo*self.ulonglongRequired+blockSize-1)//blockSize
        islandGridSize = (chromosomeNo*self.ulonglongRequired/islandSize+blockSize-1)//blockSize
        maxChromosomeSize = chromosomeNo*self.ulonglongRequired
        maxIslandSize = chromosomeNo*self.ulonglongRequired/islandSize
        bestChromsomeSize = chromosomeNo/islandSize
        #################################
        # Declare Arrays #
        #################################
        chromosomes= np.array(np.random.randint(18446744073709551615, size=maxChromosomeSize),dtype=np.ulonglong)
        chromosomes_gpu = cuda.mem_alloc(chromosomes.nbytes)
        chromosomesResults= np.array(np.random.randint(0, size=chromosomeNo),dtype=np.int)
        chromosomesResults_gpu = cuda.mem_alloc(chromosomesResults.nbytes)
        islandBestChromosomes= np.array(np.random.randint(0, size=bestChromsomeSize),dtype=np.ulonglong)
        islandBestChromosomes_gpu = cuda.mem_alloc(islandBestChromosomes.nbytes)

        roundCount = 0
        while (self.isTime) or (not self.isTime and roundCount<self.rounds):
            ##################################
            # Migration #
            ##################################
            if(roundCount%migrationRounds==0 and roundCount!=0):
                self.internalReOrder(chromosomes_gpu, np.int32(self.ulonglongRequired), chromosomesResults_gpu, np.int32(islandSize) , np.int32(maxIslandSize), block=(blockSize, 1, 1), grid=(islandGridSize, 1))       
                self.migration(chromosomes_gpu,np.int32(self.ulonglongRequired), np.int32(islandSize) , np.int32(maxIslandSize),  block=(blockSize, 1, 1), grid=(islandGridSize, 1))
            
            ##################################
            # Randomisation #
            ##################################
            # crossover
            random_crossover_index_cpu = np.random.randint(0, self.chromosomeSize, size=chromosomeNo, dtype=np.uint32)
            random_crossover_index_gpu = gpuarray.to_gpu(random_crossover_index_cpu)
            random_crossover_length_cpu = np.random.randint(0, self.chromosomeSize, size=chromosomeNo, dtype=np.uint32)
            random_crossover_length_gpu = gpuarray.to_gpu(random_crossover_length_cpu)

            # mutation
            random_mutation_prob_cpu = np.random.uniform(0, 1, chromosomeNo).astype(np.float32)
            random_mutation_prob_gpu = gpuarray.to_gpu(random_mutation_prob_cpu)

            random_mutation_index_cpu = np.random.randint(0, self.chromosomeSize, size=chromosomeNo, dtype=np.uint32)
            random_mutation_index_gpu = gpuarray.to_gpu(random_mutation_index_cpu)

            ##################################
            ##################################

            ##################################
            # genetic algorithm
            ##################################
            # np.int32(n)
            self.evaluation(chromosomes_gpu,  np.int32(self.ulonglongRequired), chromosomesResults_gpu, np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))       
            self.selection(chromosomes_gpu, np.int32(self.ulonglongRequired), chromosomesResults_gpu, islandBestChromosomes_gpu,  np.int32(islandSize), np.int32(maxIslandSize), block=(blockSize, 1, 1), grid=(islandGridSize, 1))
            self.crossover(chromosomes_gpu, np.int32(self.ulonglongRequired), islandBestChromosomes_gpu, random_crossover_index_gpu, random_crossover_length_gpu, np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
            self.mutation(chromosomes_gpu,np.int32(self.ulonglongRequired), random_mutation_prob_gpu, random_mutation_index_gpu , np.int32(self.mutationThreshold), np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
            
            ##################################
            ##################################
            roundCount +=1
        
        print("DONE")

        
