#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define MY_RAND_MAX 100000000.0

#define WIDTH_A     (int)640
#define HEIGHT_A    (int)640
#define WIDTH_B     (int)640
#define HEIGHT_B    (int)640
#define BLOCK_SIZE  16

__global__ void matrixMultiply( float* C, 
                                const float* A, 
                                const float* B, 
                                const int widthA, 
                                const int widthB
                                )
{
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
/*******************************************************************************
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    int row = block_y * BLOCK_SIZE + thread_y;
    int col = block_x * BLOCK_SIZE + thread_x;

    float C_submatrix = 0.0;

    for(int i = 0; i < widthB / BLOCK_SIZE; i++)
    {
        shared_A[thread_y][thread_x] = A[row * widthA + thread_x + i * BLOCK_SIZE];
        shared_B[thread_y][thread_x] = B[col + (i * BLOCK_SIZE + thread_y) * widthB];

        __syncthreads();
        for(int j = 0; j < BLOCK_SIZE; j++)
        {
            C_submatrix += shared_A[thread_y][j] * shared_B[j][thread_x];
        }
        __syncthreads();
    }
    C[row * widthB + col] = C_submatrix;
*******************************************************************************/
/*******************************************************************************/
    int A_start = widthA * BLOCK_SIZE * block_y;
    int A_end   = A_start + widthA - 1;
    int A_step  = BLOCK_SIZE;
    int B_start = BLOCK_SIZE * block_x;
    int B_step  = BLOCK_SIZE * widthB;

    float C_submatrix = 0.0;

    for(int a = A_start, b = B_start; a <= A_end; a += A_step, b += B_step)
    {
        __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

        shared_A[thread_y][thread_x] = A[widthA * thread_y + thread_x + a];
        shared_B[thread_y][thread_x] = B[widthB * thread_y + thread_x + b];

        __syncthreads();

        // 循环展开
        #pragma unroll

        for(int k = 0; k < BLOCK_SIZE; k++)
        {
            C_submatrix += shared_A[thread_y][k] * shared_B[k][thread_x];
        }

        __syncthreads();
    }
    int blo_bias = ( widthB * block_y  + block_x  ) * BLOCK_SIZE;
    int ele_bias = ( widthB * thread_y + thread_x );
    C[ blo_bias + ele_bias ] = C_submatrix;
/*******************************************************************************/
}

int main()
{
    struct timeval beginTime, endTime;

    size_t size_A = sizeof(float) * WIDTH_A * HEIGHT_A;
    size_t size_B = sizeof(float) * WIDTH_B * HEIGHT_B;
    size_t size_C = sizeof(float) * WIDTH_B * HEIGHT_A;

    float* host_A        = (float* )malloc( size_A );
    float* host_B        = (float* )malloc( size_B );
    float* host_C        = (float* )malloc( size_C );
    float* host_C_Serial = (float* )malloc( size_C );

    float* device_A = NULL;
    float* device_B = NULL;
    float* device_C = NULL;

    cudaMalloc( (void**)&device_A, size_A );
    cudaMalloc( (void**)&device_B, size_B );
    cudaMalloc( (void**)&device_C, size_C );


    for(int j = 0; j < WIDTH_A * HEIGHT_A; j++)
    {
        host_A[j] = rand() / MY_RAND_MAX;
    }
    for(int j = 0; j < WIDTH_B * HEIGHT_B; j++)
    {
        host_B[j] = rand() / MY_RAND_MAX;
    }

    dim3 block( BLOCK_SIZE, BLOCK_SIZE) ;
    dim3 grid(  WIDTH_B / BLOCK_SIZE, HEIGHT_A / BLOCK_SIZE);

    gettimeofday(&beginTime, NULL);
//------------------------------------------------------------------------------------
    cudaMemcpy(device_A, host_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, size_B, cudaMemcpyHostToDevice);
    matrixMultiply<<<grid, block>>>(device_C, device_A, device_B, WIDTH_A, WIDTH_B);
    cudaMemcpy(host_C, device_C, size_C, cudaMemcpyDeviceToHost);
//------------------------------------------------------------------------------------
    gettimeofday(&endTime, NULL);

    int cudaTime_us = (endTime.tv_sec - beginTime.tv_sec) * 1e6 + (endTime.tv_usec - beginTime.tv_usec);

    gettimeofday(&beginTime, NULL);
//------------------------------------------------------------------------------------
    for(int i = 0; i < HEIGHT_A; i++)
    {
        for(int j = 0; j < WIDTH_B; j++)
        {
            host_C_Serial[i * WIDTH_B + j] = 0.0;
            for(int k = 0; k < WIDTH_A; k++)
            {
                host_C_Serial[i * WIDTH_B + j] += host_A[i * WIDTH_A + k] * host_B[k * WIDTH_B + j];
            }
        }
    }
//------------------------------------------------------------------------------------
    gettimeofday(&endTime, NULL);

    int serialTime_us = (endTime.tv_sec - beginTime.tv_sec) * 1e6 + (endTime.tv_usec - beginTime.tv_usec);

    for(int i = 0; i < HEIGHT_A; i++)
    {
        for(int j = 0; j < WIDTH_B; j++)
        {
            
            if(fabs(host_C[i * WIDTH_B + j] - host_C_Serial[i * WIDTH_B + j]) > 1e-1)
            {
                printf("Seems Wrong at %d, %d , %f, %f.\n", i, j, host_C[i * WIDTH_B + j],host_C_Serial[i * WIDTH_B + j]);
                exit(0);
            }
        }
    }

    printf("Speedup ratio = %.6f\n", (float)serialTime_us / (float)cudaTime_us );

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    free(host_A);
    free(host_B);
    free(host_C);
    free(host_C_Serial);

    return 0;
}