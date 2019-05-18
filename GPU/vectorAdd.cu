#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define MY_RAND_MAX 100.0

__global__ void vectorAdd(const float* A, const float* B, float* C, int length)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < length)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
    struct timeval beginTime, endTime;
    int lengths[6] = {(int)1e5, (int)2e5, (int)1e6, (int)2e6, (int)1e7, (int)2e7};

    for(int i = 0; i < 6; i++)
    {
        size_t size = lengths[i] * sizeof(float);

        float* host_A = (float*)malloc(size);
        float* host_B = (float*)malloc(size);
        float* host_C = (float*)malloc(size);
        float* host_C_Serial = (float*)malloc(size);

        for(int j = 0; j < lengths[i]; j++)
        {
            host_A[j] = rand() / MY_RAND_MAX;
            host_B[j] = rand() / MY_RAND_MAX;
        }

        float* device_A = NULL;
        float* device_B = NULL;
        float* device_C = NULL;

        cudaMalloc((void**)&device_A, size);
        cudaMalloc((void**)&device_B, size);
        cudaMalloc((void**)&device_C, size);

        int threadsPerBlock = 1024;
        int blocksPerGrid = (lengths[i] + threadsPerBlock - 1) / threadsPerBlock;

        gettimeofday(&beginTime, NULL);
    //------------------------------------------------------------------------------------
        cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice);
        //gettimeofday(&beginTime, NULL);
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(device_A, device_B, device_C, lengths[i]);
        //gettimeofday(&endTime, NULL);
        cudaMemcpy(host_C, device_C, size, cudaMemcpyDeviceToHost);
    //------------------------------------------------------------------------------------
        gettimeofday(&endTime, NULL);

        int cudaTime_us = (endTime.tv_sec - beginTime.tv_sec) * 1e6 + (endTime.tv_usec - beginTime.tv_usec);
        
        gettimeofday(&beginTime, NULL);
    //------------------------------------------------------------------------------------
        for(int j = 0; j < lengths[i]; j++)
        {
            host_C_Serial[j] = host_A[j] + host_B[j];
        }
    //------------------------------------------------------------------------------------
        gettimeofday(&endTime, NULL);

        int serialTime_us = (endTime.tv_sec - beginTime.tv_sec) * 1e6 + (endTime.tv_usec - beginTime.tv_usec);

        for(int j = 0; j < lengths[i]; j++)
        {
            if(fabs(host_C[j] - host_C_Serial[j]) > 1e-5)
            {
                printf("Seems Wrong.\n");
            }
        }
        printf("length: %10d, Speedup ratio = %.6f\n", lengths[i], (float)serialTime_us / (float)cudaTime_us );

        cudaFree(device_A);
        cudaFree(device_B);
        cudaFree(device_C);

        free(host_A);
        free(host_B);
        free(host_C);
        free(host_C_Serial);
    }
    return 0;
}