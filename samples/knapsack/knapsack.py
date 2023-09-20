
stringPlaceholder="""

__constant__ int weightMax=1000; 

__global__ void evaluation(unsigned long long int *parents, int ulonglongRequired, int *chromosomesResults, int *constantArray, int constantArraySize, int max)
{
    int id = (blockIdx.x * blockDim.x + threadIdx.x)*ulonglongRequired;
    int tmpSize = constantArraySize/2;
    if(max>id){
        int tmpVar = 0;
        int tmpCost =0;
        bool isContinue = true; 
        for(int i =0;i<tmpSize;i++){
            int tmpParentIndex = i/64+id;
            int tmpParentDigit = i%64;
            if(tmpVar!=-1){
              if((parents[tmpParentIndex] >> tmpParentDigit) & 1){
                  tmpVar+=constantArray[i];
                  tmpCost+=constantArray[tmpSize+i];
                  if(tmpCost>weightMax){
                      tmpVar=-1;
                  }
              }
            }
        }
        chromosomesResults[(blockIdx.x * blockDim.x + threadIdx.x)]=tmpVar;
    }
}
"""

knapsackList = [18, 16, 5, 48, 41, 41, 17, 6, 21, 28, 9, 11, 10, 44, 22, 39, 39, 9, 28, 9, 42, 31, 15, 10, 9, 34, 11, 27, 1, 42, 20, 26, 18, 32, 3, 11, 13, 46, 23, 46, 41, 3, 22, 14, 38, 20, 46, 24, 27, 8, 32, 31, 10, 27, 49, 3, 40, 1, 39, 7, 12, 41, 38, 30, 28, 3, 2, 37, 31, 7, 46, 16, 27, 8, 0, 30, 50, 30, 38, 13, 9, 25, 4, 34, 31, 4, 31, 13, 38, 12, 0, 23, 19, 17, 45, 12, 45, 7, 13, 37, 13, 27, 23, 45, 29, 39, 9, 2, 4, 17, 11, 17, 13, 43, 31, 35, 0, 6, 17, 38, 15, 47, 30, 50, 4, 31, 21, 3, 44, 32, 17, 42, 27, 23, 46, 4, 24, 32, 15, 5, 37, 5, 37, 38, 11, 33, 7, 5, 24, 21, 20, 18, 48, 12, 4, 44, 8, 33, 6, 33, 19, 37, 49, 13, 36, 27, 29, 38, 28, 33, 46, 15, 49, 46, 30, 3, 9, 29, 10, 21, 10, 9, 38, 25, 8, 13, 0, 2, 12, 5, 50, 36, 23, 50, 3, 14, 7, 9, 32, 38, 20, 34, 27, 26, 25, 4, 12, 46, 24, 15, 7, 48, 19, 42, 6, 48, 23, 26, 41, 48, 2, 22, 39, 30, 14, 24, 27, 41, 29, 35, 5, 13, 21, 20, 26, 1, 44, 17, 11, 5, 49, 11, 0, 49, 16, 27, 26, 4, 11, 48, 7, 25, 46, 28, 20, 16]
problemSet=np.array(knapsackList, dtype=np.int32)



# p1 = pycuga.PyCUGA( mutationThreshold = 0.1, isTime = False, time = 0, constArr = problemSet, chromosomeSize = 18432, stringPlaceholder=stringPlaceholder)
p1 = PyCUGA( isTime = False, time = 0, constArr = problemSet, chromosomeSize = 128, stringPlaceholder=stringPlaceholder, mutationNumber = 10,crossoverMode="crossover_double")
p1. launchKernel(islandSize = 32, blockSize = 128, chromosomeNo = 1024, migrationRounds = 50,rounds = 10000)