// one-bit and two-bit flip mutation
__global__ void mutation(unsigned long long int *parents, int ulonglongRequired, float *mutateProb, int *mutateIndex, bool mode, bool isMutationKeep, int mutationThreshold, int max)
{
    int id = (blockIdx.x * blockDim.x + threadIdx.x);
    int bid = id*ulonglongRequired;
    int mutateIndexId = mutateIndex[(blockIdx.x * blockDim.x + threadIdx.x)]/64+bid;
    int mutateDigit =mutateIndex[(blockIdx.x * blockDim.x + threadIdx.x)]%64;
    if (max > id)
    {
         if (mutateProb[(blockIdx.x * blockDim.x + threadIdx.x)] > mutationThreshold)
        {
            if (!((parents[mutateIndexId] >> mutateDigit) & 1))
            {
                // if chromsome idth index is 0
                parents[mutateIndexId] |= (1ULL << mutateDigit);
            }
            else
            {
                // if chromsome idth index is 1
                parents[mutateIndexId] &= ~(1ULL << mutateDigit);
            }
            parents[mutateIndexId]=11111;
        }
    }
}
