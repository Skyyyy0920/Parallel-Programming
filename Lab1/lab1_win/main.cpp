#include <iostream>
#include<windows.h>//所需头文件
using namespace std;
int step = 100,i,j,n,f,freq;
void col_major(int n,float* &sum,float* &a,float** &b){//未优化算法
    for(i = 0;i < n;i ++){
        sum[i] = 0.0;
        for(j = 0;j < n;j ++) sum[i] += b[j][i] * a[j];
    }
}
void row_major(int n,float* &sum,float* &a,float** &b){//优化算法
    for(i = 0;i < n;i++) sum[i] = 0.0;
    for(j = 0;j < n;j++)
        for(i = 0;i < n;i++) sum[i] += b[j][i] * a[j];
}
int main()
{
    long long head,tail;
    for(n = 100;n <= 10000;n += step){//设置步长，当n＞1000时步长扩大1000
        float* a = new float [n];
        float** b = new float* [n];
        float* sum = new float [n];

        /*赋初值*/
        for(i = 0;i < n;i ++){
            a[i] = i + 0.9;
            b[i] = new float [n];
            for(j = 0;j < n;j ++) b[i][j] = i + j + 0.2;
        }
        /*当n小于1000时需要重复计算取平均值*/
        if(n < 1000) freq = 40000/n;
        else freq = 1;
        QueryPerformanceCounter((LARGE_INTEGER *)&head);//计时方法
        for(f = 0;f < freq;f++){//不需要重复计算时f=1
            col_major(n,sum,a,b);//未优化算法
        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<n<<" "<<freq<<" "<<(tail - head) / (freq * 10000.0)<<"ms"<<endl;//输出数据规模、重复计算次数和耗时
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        for(f = 0;f < freq;f++){
            row_major(n,sum,a,b);//优化算法
        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<n<<" "<<freq<<" "<<(tail - head) / (freq * 10000.0)<<"ms"<<endl;
        if(n >= 1000) step = 1000;
    }
    return 0;
}
