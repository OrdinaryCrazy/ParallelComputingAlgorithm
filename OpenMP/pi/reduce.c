// 使用并行规约
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
    //================================================================================
    #pragma omp parallel for reduction(+:sum) private(x)
    // 每个线程保留一份私有拷贝sum，x为线程私有
    // 最后对线程中所有sum进行+规约，并更新sum的全局值
    for(i = 0; i < num_steps; i++)
    {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    //================================================================================
    pi = sum * step;
    printf("%lf\n",pi);
}