#include <iostream>
#include<sys/time.h>//所需头文件
using namespace std;
int step = 100,i,j,n,f,freq;
void col_major(int n,float* &sum,float* &a,float** &b){//未优化算法
    for(i = 0;i < n;i ++){
        sum[i] = 0.0;
        for(j = 0;j < n;j ++) sum[i] += b[j][i] * a[j];
    }
}
void row_major(int n,float* &sum,float* &a,float** &b){//未优化算法
    for(i = 0;i < n;i++) sum[i] = 0.0;
    for(j = 0;j < n;j++)
        for(i = 0;i < n;i++) sum[i] += b[j][i] * a[j];
}
int main()
{
    struct timeval head;
    struct timeval tail;
    for(n = 100;n <= 10000;n += step){
        float* a = new float [n];
        float** b = new float* [n];
        float* sum = new float [n];

        /*赋初值*/
        for(i = 0;i < n;i ++){//设置步长，当n＞1000时步长扩大1000
            a[i] = i + 0.9;
            b[i] = new float [n];
            for(j = 0;j < n;j ++) b[i][j] = i + j + 0.2;
        }
        /*当n小于1000时需要重复计算取平均值*/
        if(n < 1000) freq = 40000/n;
        else freq = 1;
        gettimeofday(&head,NULL);//计时方法
        for(f = 0;f < freq;f++){//不需要重复计算时f=1
            col_major(n,sum,a,b);//未优化算法
        }
        gettimeofday(&tail,NULL);
        cout<<n<<" "<<freq<<" "<<((tail.tv_sec-head.tv_sec)*1000000+(tail.tv_usec-head.tv_usec)) / (1000.0 * freq)<<"ms"<<endl;//输出数据规模、重复计算次数和耗时
        gettimeofday(&head,NULL);
        for(f = 0;f < freq;f++){//优化算法
            row_major(n,sum,a,b);
        }
        gettimeofday(&tail,NULL);
        cout<<n<<" "<<freq<<" "<<((tail.tv_sec-head.tv_sec)*1000000+(tail.tv_usec-head.tv_usec)) / (1000.0 * freq)<<"ms"<<endl;
        if(n >= 1000) step = 1000;
    }
    return 0;
}
