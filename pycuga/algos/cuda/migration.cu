
__global__ void internalReOrder(unsigned long long int *parents, int ulonglongRequired, unsigned int *parentVals, int islandSize, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        int bId = id * islandSize;
        int lowestIndex, highestIndex, highestVal = 0;
        int lowestVal = 2147483647;
        ;
        for (int i = 0; i < islandSize; i++)
        {
            // store the chromsomes with the lowest and highest fitness values
            if (i == 0)
            {
                lowestVal = parentVals[bId + i];
                highestVal = parentVals[bId + i];
                lowestIndex = bId + i;
                highestIndex = bId + i;
            }
            else
            {
                if (parentVals[bId + i] < lowestVal)
                {
                    lowestVal = parentVals[bId + i];
                    lowestIndex = bId + i;
                }
                else if (parentVals[bId + i] > highestVal)
                {
                    highestVal = parentVals[bId + i];
                    highestIndex = bId + i;
                }
            }
        }
        unsigned long long int tmpLowest[ULONGLONGREQUIREDVALUE];
        unsigned long long int tmpHighest[ULONGLONGREQUIREDVALUE];
        //TODO
        for(int i =0; i < ulonglongRequired ; i++){
            tmpLowest[i]=parents[lowestIndex*ulonglongRequired+i];
        }
        for(int i =0; i < ulonglongRequired ; i++){
            tmpHighest[i]=parents[highestIndex*ulonglongRequired+i];
        }
        // swap the position of the first position with that of the chromosome with lowest fitness values
        for(int i =0; i < ulonglongRequired ; i++){
            parents[lowestIndex*ulonglongRequired+i] = parents[bId*ulonglongRequired+i];
        }
        for(int i =0; i < ulonglongRequired ; i++){
            parents[bId*ulonglongRequired+i]=tmpLowest[i];
        }
        // swap the position of the last position with that of the chromosome with highest fitness values
        for(int i =0; i < ulonglongRequired ; i++){
            parents[highestIndex*ulonglongRequired+i] = parents[(bId + islandSize-1)*ulonglongRequired+i];
        }
        for(int i =0; i < ulonglongRequired ; i++){
            parents[(bId + islandSize-1)*ulonglongRequired+i]=tmpHighest[i];
        }
    }
}

__global__ void migration(unsigned long long int *parents, int ulonglongRequired, int islandSize , int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (max > id)
    {
        // Migration - the last chromosome replaces the first chromosome of the next block
        int index = ((id + 1) * islandSize - 1)*ulonglongRequired;
        if (index >= max)  index = index - max;
        int replaceIndex = ((id + 1) * islandSize)*ulonglongRequired;
        if (replaceIndex >= max)  replaceIndex = replaceIndex - max;
        for(int i =0; i < ulonglongRequired ; i++){
            parents[replaceIndex+i]=parents[index+i];
        }
    }
}
