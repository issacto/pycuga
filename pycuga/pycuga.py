import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.driver as drv
import time
import os
import sys
import math
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
import os
from pycuga.algos.tools import read_files_as_strings_manual




class PyCUGA:
    def __init__(self, isTime, time, constArr, chromosomeSize, stringPlaceholder, mutationNumber, selectionMode, crossoverMode):
        self.isTime = isTime
        self.time = time
        self.dev = drv.Device(0)
        self.ctx = self.dev.make_context()
        self.ulonglongRequired = math.ceil(chromosomeSize/64)
        self.chromosomeSize=chromosomeSize
        self.mutationNumber = mutationNumber
        self.crossoverMode= crossoverMode
        self.selectionMode = selectionMode
        # num_devices = cuda.Device.count()

        # declare global CUDA functions
        
        cudaCode = read_files_as_strings_manual()
        mod = SourceModule((cudaCode+stringPlaceholder).replace("TO-BE-REPLACED-ulonglongRequired", str(self.ulonglongRequired)))
        self.crossover_one = mod.get_function("crossover_one")
        self.crossover_two = mod.get_function("crossover_two")
        self.crossover_uniform = mod.get_function("crossover_uniform")
        self.mutation = mod.get_function("mutation")
        self.selection_elitism = mod.get_function("selection_elitism")
        self.selection_roulettewheel = mod.get_function("selection_roulettewheel")
        self.evaluation = mod.get_function("evaluation")
        self.internalReOrder = mod.get_function("internalReOrder")
        self.migration = mod.get_function("migration")
        # try
        self.constantArray = gpuarray.to_gpu(constArr)
        self.constantArraySize = self.constantArray.size

    def launchKernel(self, islandSize, blockSize, chromosomeNo, migrationRounds, rounds, isDebug = False):
        parentsGridSize = int((chromosomeNo+blockSize-1)//blockSize)
        islandGridSize = int((chromosomeNo/islandSize+blockSize-1)//blockSize)
        maxChromosomeThread = chromosomeNo*self.ulonglongRequired
        maxIslandThread = int(chromosomeNo//islandSize)
        #################################
        # Print Variables #
        #################################
        if(isDebug):
            print("Block Size: " ,blockSize )
            print("Parent Grid Size: ", parentsGridSize)
            print("Island Grid Size: " ,islandGridSize )


        #################################
        # Declare Arrays and Constants #
        #################################
        chromosomes = np.random.randint(0, np.iinfo(np.uint64).max, size=maxChromosomeThread, dtype=np.uint64)
        chromosomes_gpu = gpuarray.to_gpu(chromosomes)

        chromosomesResults= np.random.randint(0, 20, size=chromosomeNo).astype(np.int32)
        chromosomesResults_gpu = gpuarray.to_gpu(chromosomesResults)

        islandBestChromosomes= np.random.randint(0, np.iinfo(np.uint64).max, size=maxIslandThread, dtype=np.uint64)
        islandBestChromosomes_gpu = gpuarray.to_gpu(islandBestChromosomes)


        roundCount = 0
        maxVal = 0
        maxChromosome =""

        while (self.isTime) or (not self.isTime and roundCount<rounds):
            print("Round - ", roundCount, maxVal, maxChromosome)
            ##################################
            # Migration #
            ##################################
            if(roundCount%migrationRounds==0 and roundCount!=0):
                if(isDebug):
                    print("MIGRATION")
                self.internalReOrder(chromosomes_gpu, np.int32(self.ulonglongRequired), chromosomesResults_gpu, np.int32(islandSize) , np.int32(maxIslandThread), block=(blockSize, 1, 1), grid=(islandGridSize, 1))
                self.evaluation(chromosomes_gpu,  np.int32(self.ulonglongRequired), chromosomesResults_gpu, self.constantArray, np.int32(self.constantArraySize), np.int32(maxChromosomeThread), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
                self.migration(chromosomes_gpu,np.int32(self.ulonglongRequired), np.int32(islandSize), np.int32(maxChromosomeThread) , np.int32(maxIslandThread),  block=(blockSize, 1, 1), grid=(islandGridSize, 1))


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
            ####### Genetic algorithm ########
            ##################################
            ##################################
            
            if(isDebug):
                print("evaluation")
            self.evaluation(chromosomes_gpu,  np.int32(self.ulonglongRequired), chromosomesResults_gpu, self.constantArray, np.int32(self.constantArraySize), np.int32(maxChromosomeThread), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
            

            ##################################
            # Selection #
            ##################################

            if(isDebug):
                print("selection")
            if(self.selectionMode=="elitism"):
                self.selection_elitism(chromosomes_gpu, np.int32(self.ulonglongRequired), chromosomesResults_gpu, islandBestChromosomes_gpu,  np.int32(islandSize), np.int32(maxIslandThread), block=(blockSize, 1, 1), grid=(islandGridSize, 1))
            elif(self.selectionMode=="roulettewheel"):
                random_selection_probs_cpu = np.random.rand(maxIslandThread).astype(np.float32)
                random_selection_probs_gpu = gpuarray.to_gpu(random_selection_probs_cpu)
                self.selection_roulettewheel(chromosomes_gpu, np.int32(self.ulonglongRequired), chromosomesResults_gpu, islandBestChromosomes_gpu, random_selection_probs_gpu, np.int32(islandSize), np.int32(maxIslandThread), block=(blockSize, 1, 1), grid=(islandGridSize, 1))
            else:
                print("NO SELECTION")

            ##################################
            # Crossover #
            ##################################
            if(isDebug):
                print("crossover")
            
            if(self.crossoverMode=="one"):
                self.crossover_one(chromosomes_gpu, np.int32(self.ulonglongRequired), islandBestChromosomes_gpu, random_crossover_index_gpu, np.int32(islandSize) ,np.int32(maxChromosomeThread), block=(blockSize, 1, 1), grid=(parentsGridSize, 1)) 
            elif(self.crossoverMode=="two"):
                self.crossover_two(chromosomes_gpu, np.int32(self.ulonglongRequired), islandBestChromosomes_gpu, random_crossover_index_gpu, random_crossover_length_gpu,  np.int32(islandSize) ,np.int32(maxChromosomeThread), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
            elif(self.crossoverMode=="uniform"):
                self.crossover_uniform(chromosomes_gpu, np.int32(self.ulonglongRequired), islandBestChromosomes_gpu, random_crossover_index_gpu, np.int32(islandSize) ,np.int32(maxChromosomeThread), block=(blockSize, 1, 1), grid=(parentsGridSize, 1)) 
            else:
                print("NO CROSSOVER")
               
            

            ##################################
            # Mutation #
            ##################################
            if(isDebug):
                print("mutation")
            
            random_mutation_index_cpu = np.random.randint(0, self.chromosomeSize, size=chromosomeNo).astype(np.int32)
            random_mutation_index_gpu = gpuarray.to_gpu(random_mutation_index_cpu)
            self.mutation(chromosomes_gpu, np.int32(self.ulonglongRequired), random_mutation_index_gpu, np.int32(self.mutationNumber) ,np.int32(chromosomeNo), np.int32(maxChromosomeThread), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
            

            
            ##################################
            # Evaluation #
            ##################################
            if(isDebug):
                print("evaluation")
            
            self.evaluation(chromosomes_gpu, np.int32(self.ulonglongRequired), chromosomesResults_gpu, self.constantArray, np.int32(self.constantArraySize), np.int32(maxChromosomeThread), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))            

            chromosomes = chromosomes_gpu.get()
            chromosomesResults = chromosomesResults_gpu.get()

            ##################################
            # DEBUGGING #
            ##################################
            # print("IMPORTANT INFORMATION")
            # print("self.ulonglongRequired",self.ulonglongRequired)
            # print("maxChromosomeThread",maxChromosomeThread)
            # print("blockSize",blockSize)
            # print("parentsGridSize",parentsGridSize)
            # print("chromosomes_gpuSize",chromosomes_gpu.size)
            # print("chromosomes_gpu",chromosomes_gpu)
            # print("self.chromosomeSize",self.chromosomeSize)
            # print("random_mutation_index_gpu",random_mutation_index_gpu)
            # print("random_mutation_index_gpu max",np.max(random_mutation_index_gpu.get()))
            # print("random_mutation_index_gpu minimum",np.min(random_mutation_index_gpu.get()))
            # print("chromosomes_gpu[0:50]",chromosomes_gpu[0:50])
            # print("chromosomesResults[0:50]",chromosomesResults[0:50])
            ##################################
            ##################################

            ##################################
            # Print Maximum #
            ##################################
            roundCount +=1
            if((gpuarray.max(chromosomesResults_gpu)).get()>maxVal):
                maxChromosome=""
                maxVal=(gpuarray.max(chromosomesResults_gpu)).get()
                resultIndex=(chromosomesResults_gpu.get()).argmax()
                for i in range(self.ulonglongRequired):
                    maxChromosome+=str(chromosomes[resultIndex*self.ulonglongRequired+i])
                    maxChromosome+=" "


        self.ctx.pop()

        ##################################
        # Print result
        ##################################
        print("Maximum value")
        print(maxVal)
        print("Maximum chromosome")
        print(maxChromosome)
        print("<-----Completed---->")


