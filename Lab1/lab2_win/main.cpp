#include <iostream>
#include <windows.h>//����ͷ�ļ�
using namespace std;
int i,n,freq,f,m;//д��ȫ�ֱ������Ż�
float sum = 0.0;
float* a;
void trivial(){//ƽ���㷨
    sum = 0.0;
    for(i = 0;i < n;i ++) sum += a[i];
}
void optimal_multilink(){//�Ż�1����·��ʽ�㷨
    float sum1 = 0.0,sum2 = 0.0;
    for(i = 0;i < n;i += 2){
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    sum = sum1 + sum2;
}
void optimal_recursion(int n){//�Ż�2���ݹ��㷨
    if(n == 1) return;
    else{
        for(i = 0;i < n / 2;i++)
            a[i] += a[n - i - 1];
        n = n / 2;
        optimal_recursion(n);
    }
}
void optimal_loop(){//�Ż�3������ѭ���㷨
    for(m = n;m > 1;m /= 2)
        for(i = 0;i < m / 2;i++) a[i] = a[i * 2] + a[i * 2 + 1];
}
int main()
{
    long long head,tail;
    for(n = 128;n <= 2000000;n *= 2){//���ݹ�ģ��2����
        freq = 2000000/n;//���ݲ�ͬ���ݹ�ģ�����ظ��������
        a = new float [n];

        QueryPerformanceCounter((LARGE_INTEGER *)&head);//��ʱ����
        for(f = 0;f < freq;f++){
           for(i = 0;i < n;i += 4){//ѭ��չ��(unroll)������ʼ��
               a[i] = i + 0.9;
               a[i + 1] = i + 1.9;
               a[i + 2] = i + 2.9;
               a[i + 3] = i + 3.9;
           }
           trivial();
        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<n<<" "<<freq<<" "<<(tail - head) / (freq * 10000.0)<<"ms"<<endl;
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        for(f = 0;f < freq;f++){
            for(i = 0;i < n;i += 4){//ѭ��չ��(unroll)������ʼ��
               a[i] = i + 0.9;
               a[i + 1] = i + 1.9;
               a[i + 2] = i + 2.9;
               a[i + 3] = i + 3.9;
            }
            optimal_multilink();
        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<n<<" "<<freq<<" "<<(tail - head) / (freq * 10000.0)<<"ms"<<endl;
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        for(f = 0;f < freq;f++){//���ڵݹ���޸�ԭ���飬������ÿ�εݹ�ǰ��ԭ���飻Ϊ���Ʊ����������㷨����ǰҲ�踴ԭ����
            for(i = 0;i < n;i += 4){//ѭ��չ��(unroll)������ʼ��
               a[i] = i + 0.9;
               a[i + 1] = i + 1.9;
               a[i + 2] = i + 2.9;
               a[i + 3] = i + 3.9;
            }
            optimal_recursion(n);
        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<n<<" "<<freq<<" "<<(tail - head) / (freq * 10000.0)<<"ms"<<endl;
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        for(f = 0;f < freq;f++){//���ڶ���ѭ�����޸�ԭ���飬������ÿ�εݹ�ǰ��ԭ���飻Ϊ���Ʊ����������㷨����ǰҲ�踴ԭ����
            for(i = 0;i < n;i += 4){//ѭ��չ��(unroll)������ʼ��
               a[i] = i + 0.9;
               a[i + 1] = i + 1.9;
               a[i + 2] = i + 2.9;
               a[i + 3] = i + 3.9;
            }
            optimal_loop();
        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout<<n<<" "<<freq<<" "<<(tail - head) / (freq * 10000.0)<<"ms"<<endl;

    }
    return 0;
}
