#include <iostream>
#include <sys/time.h>//所需头文件
using namespace std;
int i,n,freq,f,m;//写在全局变量以优化
float sum = 0.0;
float* a;
void trivial(){//平凡算法
    sum = 0.0;
    for(i = 0;i < n;i ++) sum += a[i];
}
void optimal_multilink(){//优化1：多路链式算法
    float sum1 = 0.0,sum2 = 0.0;
    for(i = 0;i < n;i += 2){
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    sum = sum1 + sum2;
}
void optimal_recursion(int n){//优化2：递归算法
    if(n == 1) return;
    else{
        for(i = 0;i < n / 2;i++)
            a[i] += a[n - i - 1];
        n = n / 2;
        optimal_recursion(n);
    }
}
void optimal_loop(){//优化3：二重循环算法
    for(m = n;m > 1;m /= 2)
        for(i = 0;i < m / 2;i++) a[i] = a[i * 2] + a[i * 2 + 1];
}
int main()
{
    struct timeval head;
    struct timeval tail;
    for(n = 128;n <= 2000000;n *= 2){//数据规模是2的幂
        freq = 2000000/n;
        a = new float [n];

        gettimeofday(&head,NULL);//计时方法
        for(f = 0;f < freq;f++){//根据不同数据规模决定重复计算次数
            for(i = 0;i < n;i += 4){//循环展开(unroll)方法初始化
            a[i] = i + 0.9;
            a[i + 1] = i + 1.9;
            a[i + 2] = i + 2.9;
            a[i + 3] = i + 3.9;
        }
           trivial();
        }
        gettimeofday(&tail,NULL);
        cout<<n<<" "<<freq<<" "<<((tail.tv_sec-head.tv_sec)*1000000+(tail.tv_usec-head.tv_usec)) / (1000.0 * freq)<<"ms"<<endl;
        gettimeofday(&head,NULL);
        for(f = 0;f < freq;f++){
        for(i = 0;i < n;i += 4){//循环展开(unroll)方法初始化
            a[i] = i + 0.9;
            a[i + 1] = i + 1.9;
            a[i + 2] = i + 2.9;
            a[i + 3] = i + 3.9;
        }
            optimal_multilink();
        }
        gettimeofday(&tail,NULL);
        cout<<n<<" "<<freq<<" "<<((tail.tv_sec-head.tv_sec)*1000000+(tail.tv_usec-head.tv_usec)) / (1000.0 * freq)<<"ms"<<endl;
        gettimeofday(&head,NULL);
        for(f = 0;f < freq;f++){//由于递归会修改原数组，必须在每次递归前复原数组；为控制变量，其他算法调用前也需复原数组
        for(i = 0;i < n;i += 4){//循环展开(unroll)方法初始化
            a[i] = i + 0.9;
            a[i + 1] = i + 1.9;
            a[i + 2] = i + 2.9;
            a[i + 3] = i + 3.9;
        }
            optimal_recursion(n);
        }
        gettimeofday(&tail,NULL);
        cout<<n<<" "<<freq<<" "<<((tail.tv_sec-head.tv_sec)*1000000+(tail.tv_usec-head.tv_usec)) / (1000.0 * freq)<<"ms"<<endl;
        gettimeofday(&head,NULL);
        for(f = 0;f < freq;f++){//由于二重循环会修改原数组，必须在每次递归前复原数组；为控制变量，其他算法调用前也需复原数组
            for(i = 0;i < n;i += 4){//循环展开(unroll)方法初始化
            a[i] = i + 0.9;
            a[i + 1] = i + 1.9;
            a[i + 2] = i + 2.9;
            a[i + 3] = i + 3.9;
        }
            optimal_loop();
        }
        gettimeofday(&tail,NULL);
        cout<<n<<" "<<freq<<" "<<((tail.tv_sec-head.tv_sec)*1000000+(tail.tv_usec-head.tv_usec)) / (1000.0 * freq)<<"ms"<<endl;

    }
    return 0;
}

