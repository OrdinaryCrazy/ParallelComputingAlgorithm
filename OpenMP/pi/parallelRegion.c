// 使用并行域并行化
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
    //===========================================================================================
    #pragma omp parallel private(i)     // 并行域开始，每个线程各自执行这段代码
    {
        double x;
        int id = omp_get_thread_num();
        for(i = id, sum[id] = 0.0; i < num_steps; i = i + NUM_THREADS)
        {   // NUM_THREADS 个线程参加计算，
            // 其中线程 NUM_THREADS-1 迭代 NUM_THREADS-1, 2*NUM_THREADS-1, ... 步
            x = (i + 0.5) * step;
            sum[id] += 4.0 / (1.0 + x * x);
        }
    }
    //===========================================================================================
    for(i = 0, pi = 0.0; i < NUM_THREADS; i++)
    {
        pi += sum[i] * step;
    }
    printf("%lf\n",pi);
}
