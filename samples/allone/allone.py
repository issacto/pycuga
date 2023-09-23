from pycuga.pycuga import PyCUGA
import numpy as np

stringPlaceholder="""

 __global__ void evaluation(unsigned long long int *parents, int ulonglongRequired, int *chromosomesResults, int* constantArray, int constantArraySize, int max)
{
    int id = (blockIdx.x * blockDim.x + threadIdx.x)*ulonglongRequired;
    if(max>id){
        int tmpVar = 0;
        for(int i = 0 ;i<ulonglongRequired;i++){
            for(int ii =0; ii< 64;ii++){
                if ((parents[id+i] >> ii) & 1)
                {
                    // if chromsome ith index is 0
                    tmpVar= tmpVar+1;
                }
            }
        }
        chromosomesResults[(blockIdx.x * blockDim.x + threadIdx.x)]=tmpVar;
    }
}

"""

satList=[]

problemSet=np.array(satList, dtype=np.int32)



p1 = PyCUGA( constArr = problemSet, chromosomeSize = 128, stringPlaceholder=stringPlaceholder,mutationNumber = 3, selectionMode="elitism", crossoverMode="two")
p1. launchKernel(islandSize = 32, blockSize = 128, chromosomeNo = 1024, migrationRounds = 50, isTime = False, timeAllowed = 30, rounds = 5000)