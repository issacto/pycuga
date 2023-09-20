__global__ void selection_elitism(unsigned long long int *parents, int ulonglongRequired,  int *parentVals, unsigned long long int *blockBestParent, int islandSize, int max)
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

__global__ void selection_roulettewheel(unsigned long long int *parents, int ulonglongRequired,  int *parentVals, unsigned long long int *blockBestParent, float *wheelProbs, int islandSize, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        int bId = id * islandSize;
        unsigned int tmpLowestVal = 100000000;
        unsigned int totalVal = 0;
        // find the lowest and total fitness value
        for (int i = 0; i < islandSize; i++)
        {
            if (parentVals[bId + i] < tmpLowestVal)
            {
                tmpLowestVal = parentVals[bId + i];
            }
            totalVal += parentVals[bId + i];
        }
        unsigned int base = totalVal - islandSize * tmpLowestVal;
        // store the cumulative proabability
        float tmpProb = 0;
        for (int i = 0; i < islandSize; i++)
        {
            tmpProb += (parentVals[bId + i] - tmpLowestVal) / base;
            if (tmpProb > wheelProbs[id])
            {
                // select the chromosome when the probability is higher than the randomly generated probability
                blockBestParent[id] = parents[bId + i];
            }
        }
    }
}
