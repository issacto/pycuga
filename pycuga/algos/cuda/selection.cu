__global__ void selection_elitism(unsigned long long int *parents, int ulonglongRequired, unsigned int *parentVals, unsigned long long int *blockBestParent, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        int bId = id * threadsInBlockIsland;
        unsigned int tmpLargestIndex = 0;
        unsigned long long int tmpLargestPar = 0;
        // iterate over the threads in an island
        for (int i = 0; i < threadsInBlockIsland; i++)
        {
            if (parentVals[bId + i] > tmpLargestVal)
            {
                tmpLargestPar = parents[bId + i];
                tmpLargestIndex = parentVals[bId + i];
            }
        }
        // select the chromosome with the highest fitness value at the corresponding blockBestParent array
        for(int i = 0 ;i<ulonglongRequired;i++){
            blockBestParent[id*ulonglongRequired+i] = blockBestParent[ulonglongRequired*ulonglongRequired+i];
        }
    }
}

