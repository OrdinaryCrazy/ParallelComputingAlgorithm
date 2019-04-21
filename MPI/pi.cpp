#include <stdio.h>
#include <mpi.h>

long    n;          // 积分区间数
double  sum,        // 进程局部和
        pi,         // 全局pi值
        mypi,       // 局部pi值
        h;          // 积分步长
int     groupSize,  // 通信进程数
        myRank;     // 局部进程号

int main(int argc, char* argv[])
{
    // 进入MPI环境并完成所有的初始化工作
    MPI_Init(&argc, &argv);
    // 获取当前进程号
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    // 获取通信域进程数
    MPI_Comm_size(MPI_COMM_WORLD, &groupSize);

    n = 2000;

    // 把积分区间数广播到整个通信域
    MPI_Bcast(  &n,             /*发啥*/ 
                1,              /*发几个*/ 
                MPI_LONG,       /*发的是啥*/
                0,              /*谁发*/
                MPI_COMM_WORLD  /*给谁发*/
                );

    h = 1.0 / (double)n;
    sum = 0.0;
    for(long i = myRank; i < n; i += groupSize)
    {
        double x = h * (i + 0.5);
        sum += 4.0 / (1.0 + x * x);
    }
    mypi = h * sum;
    // 整个通信域规约
    MPI_Reduce( &mypi,          /*从哪发*/ 
                &pi,            /*发到哪*/ 
                1,              /*发几个*/ 
                MPI_DOUBLE,     /*发的啥*/ 
                MPI_SUM,        /*收到了怎么操作*/ 
                0,              /*谁收*/ 
                MPI_COMM_WORLD  /*收谁*/ 
                );

    if(myRank == 0)
    {
        printf("%.16lf\n", pi);
    }

    // 从MPI环境中退出
    MPI_Finalize();
    return 0;
}