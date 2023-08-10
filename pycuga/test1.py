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
            ##################################
            # Migration #
            ##################################
            if(roundCount%migrationRounds==0 and roundCount!=0):
                print("MIGRATION")
                self.internalReOrder(chromosomes_gpu, np.int32(self.ulonglongRequired), chromosomesResults_gpu, np.int32(islandSize) , np.int32(maxIslandSize), block=(blockSize, 1, 1), grid=(islandGridSize, 1))
                self.evaluation(chromosomes_gpu,  np.int32(self.ulonglongRequired), chromosomesResults_gpu, np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
                chromosomes = chromosomes_gpu.get()
                print(chromosomes)
                chromosomes = chromosomes_gpu.get()
                print(chromosomesResults)
                print("<<<results>>> : ", (gpuarray.max(chromosomesResults_gpu)).get())
                self.migration(chromosomes_gpu,np.int32(self.ulonglongRequired), np.int32(islandSize), np.int32(maxChromosomeSize) , np.int32(maxIslandSize),  block=(blockSize, 1, 1), grid=(islandGridSize, 1))

                # drv.memcpy_dtoh(chromosomes, chromosomes_gpu)
                # print("chromosomes!!")
                # print(chromosomes[0:100])

            ##################################
            # Randomisation #
            ##################################
            # crossover
            random_crossover_index_cpu = np.random.randint(0, self.chromosomeSize, size=chromosomeNo)
            random_crossover_index_gpu = gpuarray.to_gpu(random_crossover_index_cpu)

            random_crossover_length_cpu = np.random.randint(0, int(self.chromosomeSize/2), size=chromosomeNo)
            random_crossover_length_gpu = gpuarray.to_gpu(random_crossover_length_cpu)


            # mutation
            # random_mutation_prob_cpu = np.random.uniform(0, 1, chromosomeNo).astype(np.float32)
            # random_mutation_prob_gpu = cuda.mem_alloc(chromosomeNo*(random_mutation_prob_cpu.dtype.itemsize))
            # drv.memcpy_htod(random_mutation_prob_gpu, random_mutation_prob_cpu)

            # drv.memcpy_dtoh(random_mutation_prob_cpu, random_mutation_prob_gpu)
            # print("random_mutation_prob_cpu")
            # print(random_mutation_prob_cpu[0:40])

            

            # print("random_mutation_index_cpu!")
            # print(random_mutation_index_cpu)
            # random_mutation_index_cpu = random_mutation_index_gpu.get()
            # print("random_mutation_index_cpu")
            # print(random_mutation_index_cpu[0:200])
            # print(random_mutation_index_cpu.size)
            ##################################
            ##################################

            ##################################
            # genetic algorithm
            ##################################
            # np.int32(n)
            print("NOT MY FAULT")
            print("<<<results Evaluation Zero-First Evaluation>>> : ", (gpuarray.max(chromosomesResults_gpu)).get())
            print(chromosomesResults_gpu)
            self.evaluation(chromosomes_gpu,  np.int32(self.ulonglongRequired), chromosomesResults_gpu, np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
            print(chromosomesResults_gpu)
            print("<<<results Evaluation First Evaluation>>> : ", (gpuarray.max(chromosomesResults_gpu)).get())
            print("evaluation")
            # print('chromosomesResults1')
            # print(chromosomesResults)
            self.selection(chromosomes_gpu, np.int32(self.ulonglongRequired), chromosomesResults_gpu, islandBestChromosomes_gpu,  np.int32(islandSize), np.int32(maxIslandSize), block=(blockSize, 1, 1), grid=(islandGridSize, 1))
            # print(chromosomes_gpu)
            print("selection")
            # print("selection")
            # print(islandBestChromosomes[0:80])
            # chromosomes = chromosomes_gpu.get()
            # print("chromosomes!!")
            # print(chromosomes[0:40])
            self.crossover(chromosomes_gpu, np.int32(self.ulonglongRequired), islandBestChromosomes_gpu, random_crossover_index_gpu, random_crossover_length_gpu, np.int32(islandSize) ,np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
            # print(chromosomes_gpu)
            print("crossover")
            # print("chromosomes!!")
            # print(chromosomes[0:40])
            # chromosomes = chromosomes_gpu.get()
            # print("MUTATION!!")
            # print(chromosomes.size)
            # print(maxChromosomeSize)
            # tmpBlockSize=32
            # tmpParentsGridSize=int((chromosomeNo+tmpBlockSize-1)//tmpBlockSize)
            print("IMPORTANT INFORMATION")
            print("self.ulonglongRequired",self.ulonglongRequired)
            print("maxChromosomeSize",maxChromosomeSize)
            print("blockSize",blockSize)
            print("parentsGridSize",parentsGridSize)
            print("chromosomes_gpu",chromosomes_gpu.size)
            
            random_mutation_index_cpu = np.random.randint(0, self.chromosomeSize, size=chromosomeNo).astype(np.int32)
            random_mutation_index_gpu = gpuarray.to_gpu(random_mutation_index_cpu)
            # print("self.chromosomeSize",self.chromosomeSize)
            # print("random_mutation_index_gpu",random_mutation_index_gpu.size)
            # print("random_mutation_index_gpu max",np.max(random_mutation_index_gpu.get()))
            # print("random_mutation_index_gpu minimum",np.min(random_mutation_index_gpu.get()))
            self.mutation(chromosomes_gpu, np.int32(self.ulonglongRequired), random_mutation_index_gpu, np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
            # print("<<ERRROR>>")
            # print(self.ulonglongRequired)
            # print(blockSize)
            # print(parentsGridSize)
            # print(maxChromosomeSize)
            print(chromosomes_gpu)
            print("mutation")
            # chromosomes = chromosomes_gpu.get()
            # roundChromosome=0
            # for chromosome in chromosomes:
            #     roundChromosome+=1
            #     if(roundChromosome>2000 and roundChromosome<3000):
            #         print("roundChromosome ", roundChromosome)
            #         print(chromosome)
            # print(chromosomes_gpu.nbytes)
            # print("mutation", roundCount)
            # chromosomes = chromosomes_gpu.get()
            # print("chromosomes!!")
            # print(chromosomes[0:200])


            ##################################
            # print result
            ##################################
            self.evaluation(chromosomes_gpu, np.int32(self.ulonglongRequired), chromosomesResults_gpu, np.int32(maxChromosomeSize), block=(blockSize, 1, 1), grid=(parentsGridSize, 1))
            print("<<<results Second Evaluation>>> : ", (gpuarray.max(chromosomesResults_gpu)).get())
            # drv.memcpy_dtoh(islandBestChromosomes, islandBestChromosomes_gpu)
            # print("islandBestChromosomes_gpu3")
            # print(islandBestChromosomes[0:40])
            # drv.memcpy_dtoh(chromosomes, chromosomes_gpu)
            # print("chromosomes!!")
            # print(chromosomes[0:40])
            chromosomes = chromosomes_gpu.get()
            chromosomesResults = chromosomesResults_gpu.get()
            print(chromosomesResults)
            print("<<<results>>> : ", (gpuarray.max(chromosomesResults_gpu)).get())

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
        # drv.memcpy_dtoh(chromosomes, chromosomes_gpu)
        # print(chromosomes)
        # drv.memcpy_dtoh(chromosomesResults, chromosomesResults_gpu)
        # print(chromosomesResults)

        print("DONE")


