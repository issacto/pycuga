__global__ void evaluation(unsigned long long int *parents, int ulonglongRequired, unsigned int *chromosomesResults, int max)
{
    int id = (blockIdx.x * blockDim.x + threadIdx.x)*ulonglongRequired;
    chromosomesResults[(blockIdx.x * blockDim.x + threadIdx.x)]=1;
    if((id%32)==0){
        chromosomesResults[(blockIdx.x * blockDim.x + threadIdx.x)]=2;
    }
}