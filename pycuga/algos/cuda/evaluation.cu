// __global__ void evaluation(unsigned long long int *parents, int ulonglongRequired, int *chromosomesResults, int max)
// {
//     int id = (blockIdx.x * blockDim.x + threadIdx.x)*ulonglongRequired;
//     if(max>id){
//         int tmpVar = 0;
//         for(int i = 0 ;i<ulonglongRequired;i++){
//             for(int ii =0; ii< 64;ii++){
//                 if ((parents[id+i] >> ii) & 1)
//                 {
//                     // if chromsome ith index is 0
//                     tmpVar= tmpVar+1;
//                 }
//             }
//         }
//         chromosomesResults[(blockIdx.x * blockDim.x + threadIdx.x)]=tmpVar;
//     }
// }

// chromosomesResults[(blockIdx.x * blockDim.x + threadIdx.x)]=1;

