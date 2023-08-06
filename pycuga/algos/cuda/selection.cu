__global__ void selection(unsigned long long int *parents, int ulonglongRequired,  int *parentVals, unsigned long long int *blockBestParent, int islandSize, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        int bId = id * islandSize;
        int tmpLargestVal = 0;
        int tmpLargestPar = 0;
        // iterate over the threads in an island
        for (int i = 0; i < islandSize; i++)
        {
            if (parentVals[bId + i] > tmpLargestVal)
            {
                tmpLargestPar = (bId + i)*ulonglongRequired;
                tmpLargestVal = parentVals[bId + i];
            }
        }
        // select the chromosome with the highest fitness value at the corresponding blockBestParent array
        for(int i = 0 ;i<ulonglongRequired;i++){
            blockBestParent[id*ulonglongRequired+i] = parents[tmpLargestPar+i];
        }
    }
}

