#include <iostream>
#include<windows.h>//����ͷ�ļ�
using namespace std;
int step = 100,i,j,n,f,freq;
void col_major(int n,float* &sum,float* &a,float** &b){//δ�Ż��㷨
    for(i = 0;i < n;i ++){
        sum[i] = 0.0;
        for(j = 0;j < n;j ++) sum[i] += b[j][i] * a[j];
    }
}
void row_major(int n,float* &sum,float* &a,float** &b){//�Ż��㷨
    for(i = 0;i < n;i++) sum[i] = 0.0;
    for(j = 0;j < n;j++)
        for(i = 0;i < n;i++) sum[i] += b[j][i] * a[j];
}
int main()
{
    long long head,tail;
    for(n = 100;n <= 10000;n += step){//���ò�������n��1000ʱ��������1000
        float* a = new float [n];
        float** b = new float* [n];
        float* sum = new float [n];

        /*����ֵ*/
        for(i = 0;i < n;i ++){
            a[i] = i + 0.9;
            b[i] = new float [n];
            for(j = 0;j < n;j ++) b[i][j] = i + j + 0.2;
        }
        /*��nС��1000ʱ��Ҫ�ظ�����ȡƽ��ֵ*/
        if(n < 1000) freq = 40000/n;
        else freq = 1;
        QueryPerformanceCounter((LARGE_INTEGER *)&head);//��ʱ����
        for(f = 0;f < freq;f++){//����Ҫ�ظ�����ʱf=1
            col_major(n,sum,a,b);//δ�Ż��㷨
        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<n<<" "<<freq<<" "<<(tail - head) / (freq * 10000.0)<<"ms"<<endl;//������ݹ�ģ���ظ���������ͺ�ʱ
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        for(f = 0;f < freq;f++){
            row_major(n,sum,a,b);//�Ż��㷨
        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<n<<" "<<freq<<" "<<(tail - head) / (freq * 10000.0)<<"ms"<<endl;
        if(n >= 1000) step = 1000;
    }
    return 0;
}
