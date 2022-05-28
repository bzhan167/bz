#include <stdio.h>

__global__ void histo_kernel(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins)
{
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    extern __shared__ int histo_private[];
    for (int i =threadIdx;x < num_bins;i+= blockDim.x){
    histo_private[threadIdx.x] = 0;
}
    __syncthreads();
    // compute block's histogram
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (i < bins)
    {
        int a = histo_private[input[i]];
        atomicAdd(&(histo_private[input[i]]), 1);
        i += stride;
    }
    // store to global histogram
    __syncthreads();
    while (threadIdx.x < histo_bins){
    atomicAdd(&(histo[threadIdx.x]), histo_private[threadIdx.x]);
    /*************************************************************************/
}
void histogram(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins)
{
    /*************************************************************************/
    // INSERT CODE HERE
    int BLOCK_SIZE = (int)num_bins;
    BLOCK_SIZE = 512;
    dim3 dim_grid, dim_block;
    dim_block.x = BLOCK_SIZE;
    dim_block.y = dim_block.z = 1;
    dim_grid.x = 1 + (num_elements - 1) / BLOCK_SIZE;
    dim_grid.y = dim_grid.z = 1;
    // create an array of uint8_t to be converted into an array of int
    uint8_t *bins_unpacked;
    cudaMalloc((void **)&bins_unpacked, 4 * num_bins * sizeof(uint8_t));
    // unpack the input uint8_t array
    unpack<<<dim_grid, dim_block>>>(bins, bins_unpacked, num_bins);
    // need an int version of bins_d
    int *bins_int_d;
    cudaMalloc((void **)&bins_int_d, num_bins * sizeof(int));
    // convert the uint8_t array to an int array
    convert<<<dim_grid, dim_block>>>(bins_unpacked, bins_int_d, num_bins);
    // run kernel and enforce saturation requirements
    int histo_private_size = num_bins;
    histo_kernel<<<dim_grid, dim_block, histo_private_size>>>(input, num_elements, bins_int_d, num_bins);
    enforce_saturation<<<dim_grid, dim_block>>>(bins_int_d, num_bins);
    // convert the int array back to uint8_t
    convert_back<<<dim_grid, dim_block>>>(bins_int_d, bins, num_bins);
    /*************************************************************************/
}
