// 使用共享任务结构并行化
#include <stdio.h>
#include <omp.h>

static long num_steps = 1e5;    // 积分区间数
double step;                    // 积分步长

#define NUM_THREADS 2   // 并行线程数

void main()
{
    int i;
    double pi, sum[NUM_THREADS];
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);   // 设置线程数量
    //===================================================================
    #pragma omp parallel                // 并行域开始，每个线程各自执行这段代码
    {
        double x;
        int id = omp_get_thread_num();
        sum[id] = 0.0;
        //*************************************************************************
        #pragma omp for                 // 未指定chunk，迭代**连续而平均地**分配给各线程
        for(i = 0; i < num_steps; i++)
        {   // 两个线程参加计算，线程0进行迭代步0~49999，线程1进行迭代步50000~99999
            x = (i + 0.5) * step;
            sum[id] += 4.0 / (1.0 + x * x);
        }
        //*************************************************************************
    }
    //===================================================================
    for(i = 0, pi = 0.0; i < NUM_THREADS; i++)
    {
        pi += sum[i] * step;
    }
    printf("%lf\n", pi);
}