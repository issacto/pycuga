
__global__ void crossover_single(unsigned long long int *parents, int ulonglongRequired, unsigned long long int *blockBestParents, int *splitIndex, int islandSize, int max)
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

__global__ void crossover_double(unsigned long long int *parents, int ulonglongRequired, unsigned long long int *blockBestParents, int *splitIndex, int *length, int islandSize, int max)
{
    int id = (blockIdx.x * blockDim.x + threadIdx.x)*ulonglongRequired;
    int startingPosition = splitIndex[(blockIdx.x * blockDim.x + threadIdx.x)];
    if (startingPosition < 0)   startingPosition = 0;
    int startingBlock = 0;
    if (startingPosition != 0) startingBlock = startingPosition/64;
    int startingIndex = startingPosition%64;
    int endingBlock =(startingPosition+length[(blockIdx.x * blockDim.x + threadIdx.x)])/64+1;
    int endingIndex = (startingPosition+length[(blockIdx.x * blockDim.x + threadIdx.x)])%64+1;
    if((startingPosition+length[(blockIdx.x * blockDim.x + threadIdx.x)])>ulonglongRequired*64){
        endingBlock=ulonglongRequired;
        endingIndex = 64;
    }
   
    if (max > id)
    {
        int bId = (blockIdx.x * blockDim.x + threadIdx.x)/islandSize;
        for(int i = startingBlock; i< endingBlock;i++ ){
            for (int ii = startingIndex; ii < endingIndex; ii++)
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
