#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include "quickSort.h"
#include "mergeSort.h"

#define ALLTOONE_TYPE   100
#define MULTI_TYPE      300
#define MULTI_LEN       600

long    arrayLen;
int*    array;
int*    tempArray;
int     localArrayLen;
int*    sample;         // 采样，主元，段长
int*    pivotIndex;

void PSRSSort()
{
    int localID, groupSize;

    MPI_Comm_rank(MPI_COMM_WORLD, &localID);
    MPI_Comm_size(MPI_COMM_WORLD, &groupSize);

    MPI_Status status[groupSize];
    MPI_Request request[groupSize];

    array = (int*)malloc(arrayLen * sizeof(int));
    tempArray = (int*)malloc(arrayLen * sizeof(int));
    if(groupSize > 1)
    {
        sample = (int*)malloc(groupSize * (groupSize - 1) * sizeof(int));
        pivotIndex = (int*)malloc(groupSize * 2 * sizeof(int));
    }
//=======================================================================
    // 均匀划分
    MPI_Barrier(MPI_COMM_WORLD);
    localArrayLen = arrayLen / groupSize;
    srand((unsigned int)time(NULL) + localID);
    usleep(5 * localID * 1000);
    printf("On Process %d the input data is:\n", localID);
    for(int i = 0; i < localArrayLen; i++)
    {
        array[i] = Myrandom();
        printf("%d\t",array[i]);
    }
    printf("\n");
//=======================================================================
    // 局部排序
    MPI_Barrier(MPI_COMM_WORLD);
    quickSort(array, 0, localArrayLen - 1);
//=======================================================================
    MPI_Barrier(MPI_COMM_WORLD);
    if(groupSize > 1)
    {   
        MPI_Barrier(MPI_COMM_WORLD);
        // 正则采样
        int step = (int)(localArrayLen / groupSize);
        for(int i = 0; i < groupSize - 1; i++)
        {
            sample[i] = array[(i + 1) * step - 1];
        }
//=================================================================================================================================
        MPI_Barrier(MPI_COMM_WORLD);
        if(localID == 0)    // 主进程收集采样
        {
            for(int i = 1, j = 0; i < groupSize; i++, j++)
            {   // Begins a nonblocking receive
                MPI_Irecv(  &sample[i * (groupSize - 1)],   // initial address of receive buffer (choice)
                            sizeof(int) * (groupSize - 1),  // number of elements in receive buffer (integer)
                            MPI_CHAR,                       // datatype of each receive buffer element (handle)
                            i,                              // rank of source (integer)
                            ALLTOONE_TYPE + i,              // message tag (integer)
                            MPI_COMM_WORLD,                 // communicator (handle)
                            &request[j]                     // communication request (handle)
                            );
            }
            // Waits for all given MPI Requests to complete
            MPI_Waitall (   groupSize - 1,              // list length (integer)
                            request,                    // array of request handles (array of handles)
                            status                      // array of status objects (array of Statuses). May be MPI_STATUSES_IGNORE.
                        );
//=================================================================================================================================
            MPI_Barrier(MPI_COMM_WORLD);
            // 采样排序
            quickSort(sample, 0, groupSize * (groupSize - 1) - 1);
            MPI_Barrier(MPI_COMM_WORLD);
//=================================================================================================================================
            for(int i = 1; i < groupSize; i++)
            {
                sample[i] = sample[i * (groupSize - 1) - 1];
            }
            // 主元广播
            // Broadcasts a message from the process with rank "root" to all other processes of the communicator
            MPI_Bcast(  sample,                     // starting address of buffer (choice)
                        groupSize * sizeof(int),    // number of entries in buffer (integer)
                        MPI_CHAR,                   // data type of buffer (handle)
                        0,                          // rank of broadcast root (integer)
                        MPI_COMM_WORLD              // communicator (handle)
                        );
            MPI_Barrier(MPI_COMM_WORLD);
//=================================================================================================================================
        }
        else
        {   // 局部采样结果发出
            MPI_Send(   sample,                         // initial address of send buffer (choice)
                        sizeof(int) * (groupSize - 1),  // number of elements in send buffer (nonnegative integer)
                        MPI_CHAR,                       // datatype of each send buffer element (handle)
                        0,                              // rank of destination (integer)
                        ALLTOONE_TYPE + localID,        // message tag (integer)
                        MPI_COMM_WORLD                  // communicator (handle)
                        );
//=================================================================================================================================
            MPI_Barrier(MPI_COMM_WORLD);
            // 采样排序
            // quickSort(sample, 0, groupSize * (groupSize - 1) - 1);
            MPI_Barrier(MPI_COMM_WORLD);
//=================================================================================================================================
            // 接收主元
            MPI_Bcast(  sample,                     // starting address of buffer (choice)
                        groupSize * sizeof(int),    // number of entries in buffer (integer)
                        MPI_CHAR,                   // data type of buffer (handle)
                        0,                          // rank of broadcast root (integer)
                        MPI_COMM_WORLD              // communicator (handle)
                        );
            MPI_Barrier(MPI_COMM_WORLD);
//=================================================================================================================================
        }
        // 主元划分
        int m = 1;  /*主元指针*/
        pivotIndex[0] = 0;
        for(int i = 0; i < localArrayLen && m < groupSize;)
        {
            if(array[i] > sample[m])
            {
                pivotIndex[2 * m    ] = i;
                pivotIndex[2 * m - 1] = i;
                m++;
            }
            else
            {
                i++;
            }
            
        }
        while(m != groupSize){
            pivotIndex[2 * m    ] = localArrayLen;
            pivotIndex[2 * m - 1] = localArrayLen;
            m++;
        }
        pivotIndex[2 * m - 1] = localArrayLen;
//=================================================================================================================================
        MPI_Barrier(MPI_COMM_WORLD);
        // 全局交换
        for(int i = 0, j = 0; i < groupSize; i++)
        {
            if(i == localID)
            {   // 划分段长度，先发射出去，就知道下一步真正传数据要传多少
                sample[i] = pivotIndex[2 * i + 1] - pivotIndex[2 * i];
                for(int m = 0, n; m < groupSize; m++)
                {
                    if(m != localID)
                    {
                        n = pivotIndex[2 * m + 1] - pivotIndex[2 * m];
                        MPI_Send(   &n,                     // initial address of send buffer (choice)
                                    sizeof(int),            // number of elements in send buffer (nonnegative integer)
                                    MPI_CHAR,               // datatype of each send buffer element (handle)
                                    m,                      // rank of destination (integer)
                                    MULTI_LEN + localID,    // message tag (integer)
                                    MPI_COMM_WORLD          // communicator (handle)
                                    );
                    }
                }
            }
            else
            {   // Blocking receive for a message
                MPI_Recv(   &sample[i],     // initial address of receive buffer (choice)
                            sizeof(int),    // maximum number of elements in receive buffer (integer)
                            MPI_CHAR,       //
                            i,              // rank of source (integer)
                            MULTI_LEN + i,  // message tag (integer)
                            MPI_COMM_WORLD, //
                            &status[j++]    // status object (Status)
                            );
            }
        }
//=================================================================================================================================
        MPI_Barrier(MPI_COMM_WORLD);
        int localPointer = 0;
        for(int i = 0, j = 0; i < groupSize; i++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if(i == localID)
            {
                for(int n = pivotIndex[2 * i]; n < pivotIndex[2 * i + 1]; n++)
                {
                    tempArray[localPointer++] = array[n];
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if(i == localID)
            {
                for(int m = 0, n = 0; m < groupSize; m++)
                {
                    if(m != localID)
                    {
                        MPI_Send(   &array[pivotIndex[2 * m]],                                  // initial address of send buffer (choice)
                                    sizeof(int) * (pivotIndex[2 * m + 1] - pivotIndex[2 * m]),  // number of elements in send buffer (nonnegative integer)
                                    MPI_CHAR,               // datatype of each send buffer element (handle)
                                    m,                      // rank of destination (integer)
                                    MULTI_TYPE + localID,   // message tag (integer)
                                    MPI_COMM_WORLD          // communicator (handle)
                                    );
                    }
                }
            }
            else
            {
                MPI_Recv(   &tempArray[localPointer], // initial address of receive buffer (choice)
                            sizeof(int) * sample[i],    // maximum number of elements in receive buffer (integer)
                            MPI_CHAR,       //
                            i,              // rank of source (integer)
                            MULTI_TYPE + i, // message tag (integer)
                            MPI_COMM_WORLD, //
                            &status[j++]    // status object (Status)
                            );
                localPointer += sample[i];
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        localArrayLen = localPointer;
        MPI_Barrier(MPI_COMM_WORLD);
        // 归并排序
        multiMergeSort(tempArray, sample, array, groupSize);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //***********************************************************************************
    usleep(5 * localID * 1000);
    if(localID == 0)printf("\n===============================================================================================\n\n");
    printf("Process %d's sorted data:\n",localID);
    for(int i = 0; i < localArrayLen; i++)
    {
        printf("%d\t",array[i]);
    }
    printf("\n");
    //***********************************************************************************
}

int main(int argc, char* argv[])
{
    int localPID;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &localPID);
    arrayLen = 64;

    PSRSSort();
    MPI_Finalize();
    return 0;
}
