 #include <cuda_runtime.h> 
 __global__ void evaluation(unsigned long long int *parents, int ulonglongRequired, int *chromosomesResults, int max)
{
    int id = (blockIdx.x * blockDim.x + threadIdx.x)*ulonglongRequired;
    if(max>id){
        int tmpVar = 0;
        for(int i = 0 ;i<ulonglongRequired;i++){
            for(int ii =0; ii< 64;ii++){
                if ((parents[id+i] >> ii) & 1)
                {
                    // if chromsome ith index is 0
                    tmpVar+=1;
                }
            }
        }
        chromosomesResults[(blockIdx.x * blockDim.x + threadIdx.x)]=tmpVar;
    }
}

;__global__ void mutation(unsigned long long int *parents, int ulonglongRequired, int *mutateIndex, int max)
{
    int id = (blockIdx.x * blockDim.x + threadIdx.x)*ulonglongRequired;
    if (max > id)
    {

        int mutateIndexId = mutateIndex[blockIdx.x * blockDim.x + threadIdx.x] / 64 +id;
        int mutateDigit =  mutateIndex[blockIdx.x * blockDim.x + threadIdx.x] % 64;
        // if(mutateIndexId<max  && mutateDigit<64 && mutateDigit>=0){
            int tmpVar = parents[mutateIndexId];
            if (!((tmpVar >> mutateDigit) & 1))
            {
                tmpVar |= (1ULL << mutateDigit);
            }
            else
            {
                // if chromsome idth index is 1
                tmpVar &= ~(1ULL << mutateDigit);
            }
            parents[mutateIndexId]=  tmpVar;
        // }

    }
}

__global__ void internalReOrder(unsigned long long int *parents, int ulonglongRequired, unsigned int *parentVals, int islandSize, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(max>id){
        int bId = id * islandSize;
        int lowestIndex, highestIndex, highestVal = 0;
        int lowestVal = 2147483647;
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

__global__ void migration(unsigned long long int *parents, int ulonglongRequired, int islandSize , int parentsSize, int max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(max>id){
        // Migration - the last chromosome replaces the first chromosome of the next block
        int index = ((id + 1) * islandSize - 1)*ulonglongRequired;
        if (index >= parentsSize)  index = index - parentsSize;
        int replaceIndex = ((id + 1) * islandSize)*ulonglongRequired;
        if (replaceIndex >= parentsSize)  replaceIndex = replaceIndex - parentsSize;
        for(int i =0; i < ulonglongRequired ; i++){
            parents[replaceIndex+i]=parents[index+i];
        }
    }
}
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


__global__ void crossover(unsigned long long int *parents, int ulonglongRequired, unsigned long long int *blockBestParents, int *splitIndex, int *length, int islandSize, int max)
{
    int id = (blockIdx.x * blockDim.x + threadIdx.x)*ulonglongRequired;
    int startingPosition = splitIndex[(blockIdx.x * blockDim.x + threadIdx.x)];
    if (startingPosition < 0)   startingPosition = 0;
    int startingBlock = 0;
    if (startingPosition != 0) startingBlock = startingPosition/64;
    int startingIndex = startingPosition-64*startingBlock;
    
   
    if (max > id)
    {
        int bId = (blockIdx.x * blockDim.x + threadIdx.x)/islandSize;
        for(int i = startingBlock; i< ulonglongRequired;i++ ){
            for (int ii = startingIndex; ii < 64; ii++)
            {
                if ((blockBestParents[bId+i] >> ii) & 1)
                {
                    // if selected chromsome ith index is 1
                    if (!((parents[id+i] >> ii) & 1))
                    {
                        // if chromsome ith index is 0
                        parents[id+i] |= (1ULL << ii);
                    }
                }
                else
                {
                    // if selected chromsome ith index is 0
                    if ((parents[id+i] >> ii) & 1)
                    {
                        // if chromsome ith index is 1
                        parents[id+i] &= ~(1ULL << ii);
                    }
                }
            }
        }
        startingIndex =0;
    }
}

__global__ void crossover_uniform(unsigned long long int *parents, int ulonglongRequired, unsigned long long int *blockBestParents, int *splitIndex, int *length, int max)
{
    int id =(blockIdx.x * blockDim.x + threadIdx.x)*ulonglongRequired;
    if (max > id)
    {
        int bId = blockIdx.x;
        for(int i = id; i< ulonglongRequired;i++ ){
            for (int ii = 0; ii < 64; ii += 2)
            {
                if ((blockBestParents[bId+i] >> ii) & 1)
                {
                    // if selected chromsome ith index is 1
                    if (!((parents[id+i] >> ii) & 1))
                    {
                        // if chromsome ith index is 0
                        parents[id+i] |= (1ULL << ii);
                    }
                }
                else
                {
                    // if selected chromsome ith index is 0
                    if ((parents[id+i] >> ii) & 1)
                    {
                        // if chromsome ith index is 1
                        parents[id+i] &= ~(1ULL << ii);
                    }
                }
            }
        }
    }
}
