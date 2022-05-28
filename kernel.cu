#include <stdio.h>

__global__ void histo_kernel(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins)
{
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    extern __shared__ int histo_private[];
    for (int i =threadIdx;i < num_bins;i+= blockDim.x)
{
    histo_private[threadIdx.x] = 0;
}
    __syncthreads();
    // compute block's histogram
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (i < num_elements)
    {
        atomicAdd(&(histo_private[input[i]]), 1);
        i += stride;
    }
    // store to global histogram
    __syncthreads();
    while (threadIdx.x < num_bins)
{
    atomicAdd(&(histo[threadIdx.x]), histo_private[threadIdx.x]);
    threadIdx.x += blockDim.x;
}
    /*************************************************************************/
}
    void histogram(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins) 
{
    /*************************************************************************/
    //INSERT CODE HERE
    dim3 dim_grid(16, 1, 1);
    dim3 dim_block(512, 1, 1);
    histo_kernel<<<dim_grid, dim_block, num_bins*sizeof(unsigned int)>>>(input, bins, num_elements, num_bins);
    /*************************************************************************/
}
