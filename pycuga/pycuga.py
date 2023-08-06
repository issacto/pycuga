import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.driver as drv
import time
import os
import sys
# sys.path.append("./algos")
# import algos.tools as tools
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
        # cudaCode = tools.read_files_as_strings()
        cudaCode = cudaCodeConst

        mod = SourceModule((cudaCode+evaluationString).replace("ULONGLONGREQUIREDVALUE", str(self.ulonglongRequired)))

        self.crossover = mod.get_function("crossover")
        self.mutation = mod.get_function("mutation")
        self.selection = mod.get_function("selection")
        self.evaluation = mod.get_function("evaluation")
        self.internalReOrder = mod.get_function("internalReOrder")
        self.migration = mod.get_function("migration")

    def launchKernel(self, islandSize, blockSize, chromosomeNo, migrationRounds):
        parentsGridSize = int((chromosomeNo+blockSize-1)//blockSize)
        islandGridSize = int((chromosomeNo/islandSize+blockSize-1)//blockSize)
        maxChromosomeSize = chromosomeNo
        maxIslandSize = int(chromosomeNo//islandSize)
        bestChromsomeSize = int(chromosomeNo/islandSize)
        #################################
        # Declare Arrays #
        #################################
        chromosomes = np.random.randint(0, np.iinfo(np.uint64).max, size=maxChromosomeSize, dtype=np.uint64)
        chromosomes_gpu = cuda.mem_alloc(maxChromosomeSize*(chromosomes.dtype.itemsize))
        drv.memcpy_htod(chromosomes_gpu, chromosomes)

        chromosomesResults= np.array(np.random.randint(20, size=chromosomeNo),dtype=np.uint32)
        chromosomesResults_gpu = cuda.mem_alloc(chromosomesResults.nbytes)
        drv.memcpy_htod(chromosomesResults_gpu, chromosomesResults)

        islandBestChromosomes= np.random.randint(0, np.iinfo(np.uint64).max, size=maxIslandSize, dtype=np.uint64)
        islandBestChromosomes_gpu = cuda.mem_alloc(maxIslandSize*(islandBestChromosomes.dtype.itemsize))
        drv.memcpy_htod(islandBestChromosomes_gpu, islandBestChromosomes)


        roundCount = 0
        while (self.isTime) or (not self.isTime and roundCount<self.rounds):
            ##################################
            # Migration #
            ##################################
            if(roundCount%migrationRounds==0 and roundCount!=0):
                print("MIGRATION")
                self.internalReOrder(chromosomes_gpu, np.int32(self.ulonglongRequired), chromosomesResults_gpu, np.int32(islandSize) , np.int32(maxIslandSize), block=(blockSize, 1, 1), grid=(islandGridSize, 1))
                drv.memcpy_dtoh(chromosomes, chromosomes_gpu)
                print("chromosomes!!")   
                print(chromosomes[0:100])        
                self.migration(chromosomes_gpu,np.int32(self.ulonglongRequired), np.int32(islandSize), np.int32(maxChromosomeSize*self.ulonglongRequired) , np.int32(maxIslandSize),  block=(blockSize, 1, 1), grid=(islandGridSize, 1))
                drv.memcpy_dtoh(chromosomes, chromosomes_gpu)
                print("chromosomes!!")   
                print(chromosomes[0:100]) 
            
            ##################################
            # Randomisation #
            ##################################
            # crossover
            random_crossover_index_cpu = np.random.randint(0, self.chromosomeSize, size=chromosomeNo, dtype=np.uint32)
            random_crossover_index_gpu = gpuarray.to_gpu(random_crossover_index_cpu)

            random_crossover_length_cpu = np.random.randint(0, int(self.chromosomeSize/2), size=chromosomeNo, dtype=np.uint32)
            random_crossover_length_gpu = gpuarray.to_gpu(random_crossover_length_cpu)
            
            # drv.memcpy_dtoh(random_crossover_length_cpu, random_crossover_length_gpu)
            # print("random_crossover_length_cpu")   
            # print(random_crossover_length_cpu[0:40])

            # mutation
            random_mutation_prob_cpu = np.random.uniform(0, 1, chromosomeNo).astype(np.float32)
            random_mutation_prob_gpu = cuda.mem_alloc(chromosomeNo*(random_mutation_prob_cpu.dtype.itemsize))
            drv.memcpy_htod(random_mutation_prob_gpu, random_mutation_prob_cpu)

            drv.memcpy_dtoh(random_mutation_prob_cpu, random_mutation_prob_gpu)
            print("random_mutation_prob_cpu")   
            print(random_mutation_prob_cpu[0:40])

            random_mutation_index_cpu = np.random.randint(0, self.chromosomeSize, size=chromosomeNo, dtype=np.uint32)
            random_mutation_index_gpu =  cuda.mem_alloc(chromosomeNo*(random_mutation_index_cpu.dtype.itemsize))
            drv.memcpy_htod(random_mutation_index_gpu, random_mutation_index_cpu)

            drv.memcpy_dtoh(random_mutation_index_cpu, random_mutation_index_gpu)
            print("random_mutation_index_cpu")   
            print(random_mutation_index_cpu[0:40])
            ##################################
            ##################################

            ##################################
            # genetic algorithm
            ##################################
            # np.int32(n)
            self.evaluation(chromosomes_gpu,  np.int32(self.ulonglongRequired), chromosomesResults_gpu, np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))    
            self.selection(chromosomes_gpu, np.int32(self.ulonglongRequired), chromosomesResults_gpu, islandBestChromosomes_gpu,  np.int32(islandSize), np.int32(maxIslandSize), block=(blockSize, 1, 1), grid=(islandGridSize, 1))
            drv.memcpy_dtoh(islandBestChromosomes, islandBestChromosomes_gpu)
            print("islandBestChromosomes_gpu1")   
            print(islandBestChromosomes[0:40])   
            drv.memcpy_dtoh(chromosomes, chromosomes_gpu)
            print("chromosomes!!")   
            print(chromosomes[0:40])   
            self.crossover(chromosomes_gpu, np.int32(self.ulonglongRequired), islandBestChromosomes_gpu, random_crossover_index_gpu, random_crossover_length_gpu, np.int32(islandSize) ,np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
            drv.memcpy_dtoh(islandBestChromosomes, islandBestChromosomes_gpu)
            print("islandBestChromosomes_gpu2")   
            print(islandBestChromosomes[0:40])   
            drv.memcpy_dtoh(chromosomes, chromosomes_gpu)
            print("chromosomes!!")   
            print(chromosomes[0:40])  
            self.mutation(chromosomes_gpu,np.int32(self.ulonglongRequired), random_mutation_prob_gpu, random_mutation_index_gpu , np.int32(self.mutationThreshold), np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
            drv.memcpy_dtoh(islandBestChromosomes, islandBestChromosomes_gpu)
            print("islandBestChromosomes_gpu3")   
            print(islandBestChromosomes[0:40])   
            drv.memcpy_dtoh(chromosomes, chromosomes_gpu)
            print("chromosomes!!")   
            print(chromosomes[0:40])  
            
            ##################################
            ##################################
            roundCount +=1

        drv.memcpy_dtoh(chromosomes, chromosomes_gpu)
        print(chromosomes)
        drv.memcpy_dtoh(chromosomesResults, chromosomesResults_gpu)
        print(chromosomesResults)

        print("DONE")

        
