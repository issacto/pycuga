// __global__ void mutation(unsigned long long int *parents, int ulonglongRequired, int *mutateIndex, int max)
// {
//     int id = (blockIdx.x * blockDim.x + threadIdx.x)*ulonglongRequired;
//     if (max > id)
//     {

//         int mutateIndexId = mutateIndex[blockIdx.x * blockDim.x + threadIdx.x] / 64 +id;
//         int mutateDigit =  mutateIndex[blockIdx.x * blockDim.x + threadIdx.x] % 64;
//         // if(mutateIndexId<max  && mutateDigit<64 && mutateDigit>=0){
//             int tmpVar = parents[mutateIndexId];
//             if (!((tmpVar >> mutateDigit) & 1))
//             {
//                 tmpVar |= (1ULL << mutateDigit);
//             }
//             else
//             {
//                 // if chromsome idth index is 1
//                 tmpVar &= ~(1ULL << mutateDigit);
//             }
//             parents[mutateIndexId]=  tmpVar;
//         // }

//     }
// }


__global__ void mutation(unsigned long long int *parents, int ulonglongRequired, int *mutateIndex, int mutateVal, int chromosomeNo, int max)
{
    int id = (blockIdx.x * blockDim.x + threadIdx.x)*ulonglongRequired;
    if (max > id)
    {
        for(int i =0;i<mutateVal;i++){
            int index =(blockIdx.x * blockDim.x + threadIdx.x+i)%chromosomeNo;
            int mutateIndexId = mutateIndex[index] / 64 +id;
            int mutateDigit =  mutateIndex[index] % 64; 
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
        }

    }
}
