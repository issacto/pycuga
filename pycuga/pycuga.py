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
    def __init__(self, mutationThreshold, isTime, time, constArr, chromosomeSize, evaluationString):
        self.isTime = isTime
        self.time = time
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

    def launchKernel(self, islandSize, blockSize, chromosomeNo, migrationRounds, rounds):
        parentsGridSize = int((chromosomeNo+blockSize-1)//blockSize)
        islandGridSize = int((chromosomeNo/islandSize+blockSize-1)//blockSize)
        maxChromosomeSize = chromosomeNo*self.ulonglongRequired
        maxIslandSize = int(chromosomeNo//islandSize)
        bestChromsomeSize = int(chromosomeNo/islandSize)
         #################################
        # Print Variables #
        #################################
        print("Block Size: " ,blockSize )
        print("Parent Grid Size: ", parentsGridSize)
        print("Island Grid Size: " ,islandGridSize )


        #################################
        # Declare Arrays #
        #################################
        # chromosomes = np.random.randint(0, np.iinfo(np.uint64).max, size=maxChromosomeSize, dtype=np.uint64)
        # chromosomes_gpu = cuda.mem_alloc(maxChromosomeSize*(chromosomes.dtype.itemsize))
        # drv.memcpy_htod(chromosomes_gpu, chromosomes)

        chromosomes = np.random.randint(0, np.iinfo(np.uint64).max, size=maxChromosomeSize, dtype=np.uint64)
        chromosomes_gpu = gpuarray.to_gpu(chromosomes)

        chromosomesResults= np.random.randint(0, 20, size=chromosomeNo).astype(np.int32)
        chromosomesResults_gpu = gpuarray.to_gpu(chromosomesResults)
        print("chromosomesResults_gpu")
        print(chromosomesResults_gpu)
        print("<<<results Evaluation Zero Evaluation>>> : ", (gpuarray.max(chromosomesResults_gpu)).get())

        islandBestChromosomes= np.random.randint(0, np.iinfo(np.uint64).max, size=maxIslandSize, dtype=np.uint64)
        islandBestChromosomes_gpu = gpuarray.to_gpu(islandBestChromosomes)


        roundCount = 0
        maxVal = 0
        maxChromosome =""
        while (self.isTime) or (not self.isTime and roundCount<rounds):
            print("Round - ", roundCount)
            ##################################
            # Migration #
            ##################################
            if(roundCount%migrationRounds==0 and roundCount!=0):
                print("MIGRATION")
                self.internalReOrder(chromosomes_gpu, np.int32(self.ulonglongRequired), chromosomesResults_gpu, np.int32(islandSize) , np.int32(maxIslandSize), block=(blockSize, 1, 1), grid=(islandGridSize, 1))
                self.evaluation(chromosomes_gpu,  np.int32(self.ulonglongRequired), chromosomesResults_gpu, np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
                print("<<<results>>> : ", (gpuarray.max(chromosomesResults_gpu)).get())
                self.migration(chromosomes_gpu,np.int32(self.ulonglongRequired), np.int32(islandSize), np.int32(maxChromosomeSize) , np.int32(maxIslandSize),  block=(blockSize, 1, 1), grid=(islandGridSize, 1))


            ##################################
            # Randomisation #
            ##################################
            # crossover
            random_crossover_index_cpu = np.random.randint(0, self.chromosomeSize, size=chromosomeNo)
            random_crossover_index_gpu = gpuarray.to_gpu(random_crossover_index_cpu)

            random_crossover_length_cpu = np.random.randint(0, int(self.chromosomeSize/2), size=chromosomeNo)
            random_crossover_length_gpu = gpuarray.to_gpu(random_crossover_length_cpu)

            ##################################
            ##################################
            ##################################
            ####### Genetic algorithm ########
            ##################################
            ##################################
            ##################################
            
            print("evaluation")
            self.evaluation(chromosomes_gpu,  np.int32(self.ulonglongRequired), chromosomesResults_gpu, np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
            # print("<<<results Evaluation First Evaluation>>> : ", (gpuarray.max(chromosomesResults_gpu)).get())
            print("selection")
            self.selection(chromosomes_gpu, np.int32(self.ulonglongRequired), chromosomesResults_gpu, islandBestChromosomes_gpu,  np.int32(islandSize), np.int32(maxIslandSize), block=(blockSize, 1, 1), grid=(islandGridSize, 1))
            print("crossover")
            self.crossover(chromosomes_gpu, np.int32(self.ulonglongRequired), islandBestChromosomes_gpu, random_crossover_index_gpu, random_crossover_length_gpu, np.int32(islandSize) ,np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
            random_mutation_index_cpu = np.random.randint(0, self.chromosomeSize, size=chromosomeNo).astype(np.int32)
            random_mutation_index_gpu = gpuarray.to_gpu(random_mutation_index_cpu)
            print("mutation")
            self.mutation(chromosomes_gpu, np.int32(self.ulonglongRequired), random_mutation_index_gpu, np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
            

            ##################################
            ##################################
            # DEBUGGING
            ##################################
            # print("IMPORTANT INFORMATION")
            # print("self.ulonglongRequired",self.ulonglongRequired)
            # print("maxChromosomeSize",maxChromosomeSize)
            # print("blockSize",blockSize)
            # print("parentsGridSize",parentsGridSize)
            # print("chromosomes_gpu",chromosomes_gpu.size)
            # print("self.chromosomeSize",self.chromosomeSize)
            # print("random_mutation_index_gpu",random_mutation_index_gpu.size)
            # print("random_mutation_index_gpu max",np.max(random_mutation_index_gpu.get()))
            # print("random_mutation_index_gpu minimum",np.min(random_mutation_index_gpu.get()))
            ##################################
            ##################################
            
            
            ##################################
            # print result
            ##################################
            self.evaluation(chromosomes_gpu, np.int32(self.ulonglongRequired), chromosomesResults_gpu, np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))            

            chromosomes = chromosomes_gpu.get()
            chromosomesResults = chromosomesResults_gpu.get()
            # print(chromosomesResults)
            # print("<<<results>>> : ", (gpuarray.max(chromosomesResults_gpu)).get())

            ##################################
            ##################################
            roundCount +=1
            if((gpuarray.max(chromosomesResults_gpu)).get()>maxVal):
                maxChromosome=""
                maxVal=(gpuarray.max(chromosomesResults_gpu)).get()
                resultIndex=(chromosomesResults_gpu.get()).argmax()
                for i in range(self.ulonglongRequired):
                    maxChromosome+=str(chromosomes[resultIndex*self.ulonglongRequired+i])
                    maxChromosome+=" "


        ##################################
        # print result
        ##################################
        print("maxVal")
        print(maxVal)
        print("maxChromosome")
        print(maxChromosome)

        print("<-----Completed---->")


