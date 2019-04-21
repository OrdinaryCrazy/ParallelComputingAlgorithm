// 使用private子句和critical部分并行化
#include <stdio.h>
#include <omp.h>

static long num_steps = 1e5;    // 积分区间数
double step;                    // 积分步长

#define NUM_THREADS 2   // 并行线程数

void main()
{
    int i;
    double pi = 0.0, sum = 0.0, x = 0.0;
    step = 1.0 / (double)num_steps;

    omp_set_num_threads(NUM_THREADS);   // 设置线程数量
    //=================================================================
    #pragma omp parallel private(i,x,sum) // private子句表示x,sum变量对于每个线程是私有的
    {
        int id = omp_get_thread_num();
        for(i = id, sum = 0.0; i < num_steps; i = i + NUM_THREADS)
        {   // NUM_THREADS 个线程参加计算，
            // 其中线程 NUM_THREADS-1 迭代 NUM_THREADS-1, 2*NUM_THREADS-1, ... 步
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
        //*************************************************************
        #pragma omp critical            // critical代码段在同一时刻只能由一个线程执行
        {                               // 当某线程在这里执行时，其他到达该段代码的线程
            pi += sum * step;           // 被阻塞直到正在执行的线程退出临界区                                 
        }
        //*************************************************************
    }
    //=================================================================
    printf("%lf\n",pi);
}
