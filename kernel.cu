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
    __shared__ float ds_A[TILE_SIZE][TILE_SIZE];   //declaring shared memory for A and B
    __shared__ float ds_B[TILE_SIZE][TILE_SIZE];    
    
    int Row = (blockIdx.y * TILE_SIZE) + threadIdx.y; //mapping row and col, tilesize as blockDim
    int Col = (blockIdx.x * TILE_SIZE) + threadIdx.x; 

      //starts with loading shared  element along phase change
    float Pvalue = 0;
    for(int p = 0; p < (k-1)/TILE_SIZE + 1; ++p){      //k as width for M to check phase needed
	
	if((Row < m) && (p*TILE_SIZE+threadIdx.x < k)){   //check Row and column are under A size
            ds_A[threadIdx.y][threadIdx.x] = A[Row*k + p*TILE_SIZE + threadIdx.x];  
        //see A as horizontal blocks, Row*k(Acol) to get to the block, p*tilewidth+x for smaller
        } else {
      	    ds_A[threadIdx.y][threadIdx.x] = 0.0;
        }
        if((p*TILE_SIZE+threadIdx.y < k) && (Col < n )){      //check under B size kxn
	    ds_B[threadIdx.y][threadIdx.x] = B[(p*TILE_SIZE + threadIdx.y)*n + Col];  //don't know?
        } else {
	    ds_B[threadIdx.y][threadIdx.x] = 0.0;
	}

	__syncthreads();
	// calculate the partial sum
        if((Row < m) && (Col < n)){      //checking output size of mxn
	 for(int i = 0; i < TILE_SIZE; ++i){
	 	Pvalue += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
	 }
	}
	__syncthreads();   //inside if loop cause problems
 
    } //end of outer for loop? shouldn't it be inside phase?
    if((Row < m) && (Col < n)){
	C[Row*n + Col] = Pvalue;	
    }
  // } //
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


