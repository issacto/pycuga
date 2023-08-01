
__global__ void crossover_fixed(unsigned long long int *parents, int ulonglongRequired, unsigned long long int *blockBestParents, int *splitIndex, int *length, int max)
{
    int id = (blockIdx.x * blockDim.x + threadIdx.x)*ulonglongRequired;
    int startingPosition = splitIndex[id] - length[id];
    if (startingPosition < 0)   startingPosition = 0;
    int startingBlock = 0;
    if (startingPosition != 0) startingBlock = startingPosition/64;
    int startingIndex = startingPosition-64*startingBlock;
    
   
    if (max > id)
    {
        int bId = blockIdx.x * ulonglongRequired;
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
