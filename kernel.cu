#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    __shared__ float ds_A[TILE_SIZE][TILE_SIZE];   
    __shared__ float ds_B[TILE_SIZE][TILE_SIZE];    
    
    int Row = (blockIdx.y * TILE_SIZE) + threadIdx.y; 
    int Col = (blockIdx.x * TILE_SIZE) + threadIdx.x; 

    float Pvalue = 0;
    for(int p = 0; p < (k-1)/TILE_SIZE + 1; ++p)
    {      
    if((Row < m) && (p*TILE_SIZE+threadIdx.x < k))
    {   
            ds_A[threadIdx.y][threadIdx.x] = A[Row*k + p*TILE_SIZE + threadIdx.x];  
        }
    else {
      	    ds_A[threadIdx.y][threadIdx.x] = 0.0;
        }
    if((p*TILE_SIZE+threadIdx.y < k) && (Col < n ))
    {      
	    ds_B[threadIdx.y][threadIdx.x] = B[(p*TILE_SIZE + threadIdx.y)*n + Col];  
        } 
    else {
	    ds_B[threadIdx.y][threadIdx.x] = 0.0;
	}
	__syncthreads();
    if((Row < m) && (Col < n))
    {      
    for(int i = 0; i < TILE_SIZE; ++i)
    {
	 	Pvalue += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
	 }
	}
	__syncthreads();   
    } 
    if((Row < m) && (Col < n))
    {
	C[Row*n + Col] = Pvalue;	
    }
    /*************************************************************************/
}
void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------
    const unsigned int BLOCK_SIZE = TILE_SIZE;
    /*************************************************************************/
    //INSERT CODE HERE
    dim3 dim_grid(((n-1) / BLOCK_SIZE) + 1, ((m-1) / BLOCK_SIZE) + 1, 1);
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    /*************************************************************************/
    // Invoke CUDA kernel -----------------------------------------------------
    /*************************************************************************/
    //INSERT CODE HERE
    mysgemm<<<dim_grid, dim_block>>>(m, n, k, A, B, C);	
    /*************************************************************************/
}


