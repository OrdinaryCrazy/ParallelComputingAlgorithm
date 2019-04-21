/*******************************************************************************
 * PSRS (Parallel Sorting by Regular Sampling) 排序算法：
 * STEP1 均匀划分: 将n个元素A[1,...,n]均匀划分为p段，每个pi处理A[(i-1)n/p+1,...,in/p]
 * STEP2 局部排序: pi调用串行排序算法对A[(i-1)n/p+1,...,in/p]排序
 * STEP3 正则采样: pi从有序子序列A[(i-1)n/p+1,...,in/p]中选取p个样本元素
 * STEP4 采样排序: 用一台处理器对p^2个样本元素进行串行排序
 * STEP5 选择主元: 用一台处理器从排好序的样本序列中选取p-1个主元，并传播给其他pi
 * STEP6 主元划分: pi按主元将有序段A[(i-1)n/p+1,...,in/p]划分成p段
 * STEP7 全局交换: 各处理器将其有序段按段号交换到对应的处理器中
 * STEP8 局部排序: 各处理器对接收到的元素进行局部排序
********************************************************************************/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define NUM_THREADS 3   // 并行线程数

#define RANDOM_LIMIT 50
#define TEST_SIZE 81
#define SHOW_CORRECTNESS
//#define SHOW_DISTRIBUTION

double Myrandom(void){
    int Sign = rand() % 2;
    return (rand() % RANDOM_LIMIT) / pow(-1,Sign + 2); 
}
void swap(int* a, int* b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}
int partition(int* array, int left, int right)
{
    int x = array[right];
    int i = left - 1;
    for(int j = left; j < right; j++)
    {
        if(array[j] <= x)
        {
            swap(&array[++i],&array[j]);
        }
    }
    swap(&array[i + 1],&array[right]);
    return i + 1;
}
void quickSort(int* array, int left, int right)
{
    if(left < right)
    {
        int q = partition(array, left, right);
        quickSort(array, left, q - 1);
        quickSort(array, q + 1, right);
    }
}
void PSRSSort(int* array, int length)
{
    int base = length / NUM_THREADS;                    // 划分段长度(这里假设能整除，不能整除补无穷大)
    int sample[NUM_THREADS * NUM_THREADS];              // 正则采样数
    int pivot[NUM_THREADS - 1];                         // 主元
    int count[NUM_THREADS][NUM_THREADS] = {0};          // 各cpu段长度
    int pivotArray[NUM_THREADS][NUM_THREADS][50] = {0}; // 各cpu段

    omp_set_num_threads(NUM_THREADS);   // 设置线程数量
    //=================================================================================
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        // 并行局部排序
        quickSort(array, id * base, (id + 1) * base - 1);
        // 正则采样
        for(int j = 0; j < NUM_THREADS; j++)
        {
            sample[id * NUM_THREADS + j] = array[id * base + (j + 1) * base / (NUM_THREADS + 1)];
        }
        #pragma omp barrier             // 设置路障，同步队列中的所有线程
        //*****************************************************************************
        #pragma omp master              
        {   // 主线程采样排序
            quickSort(sample, 0, NUM_THREADS * NUM_THREADS - 1);
            // 选择主元
            for(int i = 1; i < NUM_THREADS; i++)
            {
                pivot[i - 1] = sample[i * NUM_THREADS];
            }
        }
        #pragma omp barrier             // 设置路障，同步队列中的所有线程
        //*****************************************************************************
        for(int k = 0, m = 0/*主元指针*/; k < base; k++)       // 主元划分
        {
            if(array[id * base + k] < pivot[m])
            {   
                pivotArray[id][m][count[id][m]++] = array[id * base + k];
            }
            else
            {   
                m != NUM_THREADS - 1 ? m++ : 0;     // 最后一段的处理
                pivotArray[id][m][count[id][m]++] = array[id * base + k];
            }
        }
        //*****************************************************************************
        #pragma omp barrier             // 设置路障，同步队列中的所有线程
        // 全局交换
        for(int k = 0; k < NUM_THREADS; k++)
        {
            if(k != id)
            {
                memcpy(pivotArray[id][id] + count[id][id], pivotArray[k][id], sizeof(int) * count[k][id]);
                count[id][id] += count[k][id];
            }
        }
        // 局部排序
        quickSort(pivotArray[id][id], 0, count[id][id] - 1);
    }
    //=================================================================================
    // 结果输出
    #ifdef SHOW_DISTRIBUTION
    for(int z = 0; z < NUM_THREADS; z++)
        printf("%d\t",count[z][z]);
    printf("\n");
    #endif
    #ifdef SHOW_CORRECTNESS
    printf("The Sorted Array is:\n");
    for(int x = 0; x < NUM_THREADS; x++)
    {
        for(int y = 0; y < count[x][x]; y++)
            printf("%d\t",pivotArray[x][x][y]);
        printf("\n");
    }
    #endif
}
int main(int argc, char* argv[])
{   //====================================================================
    srand((unsigned int)time(NULL));
    int test[TEST_SIZE];
    for(int i = 0;i < TEST_SIZE;i++)
        test[i] = Myrandom();
    //====================================================================
    #ifdef SHOW_CORRECTNESS
    printf("The Original Array is:\n");
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
            printf("%d\t",test[i * 9 + j]);
        printf("\n");
    }
    #endif
    //====================================================================
    PSRSSort(test, TEST_SIZE);
    //====================================================================
    return 0;
}