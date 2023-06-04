#include <iostream>
#include<sys/time.h>
#include<stdlib.h>
using namespace std;
const int n = 1024;
float A[n][n];
void ReSet(){
    for(int i = 0;i < n; i++){
        for(int j = 0;j < i; j++)
            A[i][j] = 0;
        A[i][i] = 1.0;
        for(int j = i + 1;j < n; j++)
            A[i][j] = rand() % 100;
    }
    for(int i = 0;i < n; i++){
        int k1 = rand() % n;
        int k2 = rand() % n;
        for(int j = 0;j < n; j++){
            A[i][j] += A[0][j];
            A[k1][j] += A[k2][j];
        }
    }

}
void LU(){
    for(int k = 0;k < n; k++){
        for(int j = k + 1;j < n; j++){
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < n; i++){
            for(int j = k + 1;j < n; j++){
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
    }
    return;
}
void Print(){
    for(int i = 0;i < n; i++){
        for(int j = 0;j < n; j++)
            cout<<A[i][j]<<" ";
        cout<<endl;
    }


}
int main()
{
    struct timeval head;
    struct timeval tail;

    ReSet();
    gettimeofday(&head,NULL);
    LU();
    gettimeofday(&tail,NULL);
    cout<<"N: "<<n<<" Time: "<<(tail.tv_sec-head.tv_sec)*1000.0+(tail.tv_usec-head.tv_usec)/1000.0<<"ms";


    return 0;
}
