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
#include <mpi.h>
//============================= 测试用参数和辅助函数 ===================================
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
//===============================================================================
void PSRSSort(int* array, int length)
{
    int groupSize,  // 通信域大小
        myRank,     // 通信进程号
        startIndex, // 子数组起始位置
        endIndex,   // 子数组结束为止
        *newlocala, // 最后的局部数组
        *pivot,     // 主元数组
        *count,     // 主元划分段长度
        *newcount,  // 接收段长度
        *sample,    // 局部采样数组
        *sampleAll; // 全局采样数组
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &groupSize);

    //=================================================================================
    // 均匀划分
    startIndex = myRank * length / groupSize;
    endIndex = (myRank + 1 == groupSize) ? length : (myRank + 1) * length / groupSize;
    // 局部排序
    quickSort(array, startIndex, endIndex - 1);
    // 正则采样
    sample = (int*)malloc(groupSize * sizeof(int));
    for(int i = 0; i < groupSize; i++)
    {
        sample[i] = array[startIndex + i * (length / (groupSize * groupSize))];
    }
    //=================================================================================

    //*********************************************************************************
    MPI_Barrier(MPI_COMM_WORLD);
    //*********************************************************************************

    //=================================================================================
    // 采样收集
    sampleAll = (int*)malloc(groupSize * groupSize * sizeof(int));
    pivot = (int*)malloc((groupSize - 1) * sizeof(int));
    MPI_Gather( sample,     // 从哪发
                groupSize,  // 发几个元素
                MPI_INT,    // 元素类型
                sampleAll,  // 发到哪
                groupSize,  // 一次收几个元素
                MPI_INT,
                0,          // 接收进程号
                MPI_COMM_WORLD
                );
    if(myRank == 0)
    {   // 采样排序
        quickSort(sampleAll, 0, groupSize * groupSize);
        // 选择主元
        for(int i = 1; i < groupSize; i++)
        {
            pivot[i - 1] = sampleAll[i * groupSize];
        }
    }
    // 主元广播
    MPI_Bcast(  &pivot,         /*发啥*/ 
                groupSize - 1,  /*发几个*/ 
                MPI_INT,        /*发的是啥*/
                0,              /*谁发*/
                MPI_COMM_WORLD  /*给谁发*/
                );
    // 主元划分
    count = (int*)malloc(groupSize * sizeof(int));
    for(int i = startIndex, m = 0/*主元指针*/; i < endIndex; i++)
    {
        (array[i] > pivot[m]) ? m++ : 0;
        if(m == groupSize)
        {
            count[m - 1] = endIndex - startIndex - i + 1;
            break;
        }
        count[m]++;
    }
    // 全局交换
    newcount = (int*)malloc(groupSize * sizeof(int));
    // 第一步：发送段长
    MPI_Alltoall(   count,
                    1,
                    MPI_INT,
                    newcount,
                    1,
                    MPI_INT,
                    MPI_COMM_WORLD
                );
    // 第二步：新的局部段
    int totalSize = 0;
    int* sendShift = (int*)malloc(groupSize * sizeof(int));  // 发送缓冲偏移量
    int* recvShift = (int*)malloc(groupSize * sizeof(int));  // 接收缓冲偏移量
    for(int i = 0; i < groupSize; i++)
    {
        totalSize += newcount[i];
    }
    newlocala = (int*)malloc(totalSize * sizeof(int));
    sendShift[0] = 0;
    recvShift[0] = 0;
    for(int i = 1; i < groupSize; i++)
    {
        sendShift[i] = sendShift[i - 1] + count[i - 1];
        recvShift[i] = recvShift[i - 1] + newcount[i - 1];
    }
    // 第三步：发送数据
    MPI_Alltoallv(  &array[startIndex], // 发送缓冲区起始地址
                    count,              // 发送段长数组
                    sendShift,          // 发送偏移数组
                    MPI_INT,
                    newlocala,
                    newcount,
                    recvShift,
                    MPI_INT,
                    MPI_COMM_WORLD
                    );
    free(sendShift);
    free(recvShift);
    // 局部排序
    quickSort(newlocala, 0, totalSize - 1);
    // 全局排序结果收集
    // 第一步：收集局部数组长度
    int* subArrayLen = (int*)malloc(groupSize * sizeof(int));
    MPI_Gather( &totalSize,
                1,
                MPI_INT,
                subArrayLen,
                1,
                MPI_INT,
                0,
                MPI_COMM_WORLD
                );
    // 第二步：计算偏移量
    if(myRank == 0)
    {
        recvShift = (int*)malloc(groupSize * sizeof(int));
        recvShift[0] = 0;
        for(int i = 1; i < groupSize; i++)
        {
            recvShift[i] = recvShift[i - 1] + subArrayLen[i - 1];
        }
    }
    // 第三步：收集子序列
    MPI_Gatherv(    newlocala,
                    totalSize,
                    MPI_INT,
                    array,
                    subArrayLen,
                    recvShift,
                    MPI_INT,
                    0,
                    MPI_COMM_WORLD
                    );
    //=================================================================================
    free(subArrayLen);
    free(recvShift);
    free(sample);
    free(sampleAll);
    free(pivot);
    free(count);
    free(newcount);
    //=================================================================================
    MPI_Finalize();
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
    MPI_Init(&argc, &argv);
    PSRSSort(test, TEST_SIZE);
    //====================================================================
    #ifdef SHOW_CORRECTNESS
    printf("The Sorted Array is:\n");
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
            printf("%d\t",test[i * 9 + j]);
        printf("\n");
    }
    #endif
    //====================================================================
    return 0;
}