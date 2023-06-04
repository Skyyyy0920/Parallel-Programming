#include <malloc.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
//进程数
int n_threads = 8;

//方阵
const int N = 512;
float A[N][N];

//初始化
void init() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++)
            A[i][j] = 0;
        A[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            A[i][j] = (i + j) % 100;
    }
    for (int i = 0; i < N; i++) {
        int k1 = rand() % N;
        int k2 = rand() % N;
        for (int j = 0; j < N; j++) {
            A[i][j] += A[0][j];
            A[k1][j] += A[k2][j];
        }
    }
}
void Print() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << A[i][j] << " ";
        }
        cout << endl;
    }

}
int main(int argc, char* argv[]) {
    int myid = 0;
    MPI_Init(&argc, &argv);//MPI进行必要的初始化工作
    MPI_Comm_size(MPI_COMM_WORLD, &n_threads);//设置进程数为n_threads
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);//识别调用进程的rank，值从0到size-1

    //初始化矩阵A
    init();//question:是分别调用init吗？

    //计时开始
    double ts = MPI_Wtime();

    int tmp = (N - N % n_threads) / n_threads;
    int r1 = myid * tmp;
    int r2 = myid * tmp + tmp - 1;
    //逐行作为主行元素，进行初等行变换
    for (int k = 0; k < N; k++) {
        for(int i = k;i < N; i++){
            if(i < N / 3 && k < N / 2){
                MPI_Bcast(&(A[i][k]), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            }
            if(i < N / 3 && k >= N * 2 / 3){
                MPI_Bcast(&(A[i][k]), 1, MPI_FLOAT, 1, MPI_COMM_WORLD);
            }
            if(i >= N / 3 && i < N * 2 / 3 && k < N / 3){
                MPI_Bcast(&(A[i][k]), 1, MPI_FLOAT, 2, MPI_COMM_WORLD);
            }
            if(i >= N / 3 && i < N * 2 / 3 && k >= N / 3 && k < N * 2 / 3){
                MPI_Bcast(&(A[i][k]), 1, MPI_FLOAT, 3, MPI_COMM_WORLD);
            }
            if(i >= N / 3 && i < N * 2 / 3 && k >= N * 2 / 3){
                MPI_Bcast(&(A[i][k]), 1, MPI_FLOAT, 4, MPI_COMM_WORLD);
            }
            if(i >= N * 2 / 3 && k < N / 3){
                MPI_Bcast(&(A[i][k]), 1, MPI_FLOAT, 5, MPI_COMM_WORLD);
            }
            if(i >= N * 2 / 3 && k >= N / 3 && k < N * 2 / 3){
                MPI_Bcast(&(A[i][k]), 1, MPI_FLOAT, 6, MPI_COMM_WORLD);
            }
            if(i >= N * 2 / 3 && k >= N  * 2 / 3){
                MPI_Bcast(&(A[i][k]), 1, MPI_FLOAT, 7, MPI_COMM_WORLD);
            }
        }
        for(int j = k + 1;j < N; j++){
            A[k][j] =  A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for(int j = k + 1;j < N; j++){
            if(k < N / 3 && j < N / 2){
                MPI_Bcast(&(A[k][j]), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            }
            if(k < N / 3 && j >= N * 2 / 3){
                MPI_Bcast(&(A[k][j]), 1, MPI_FLOAT, 1, MPI_COMM_WORLD);
            }
            if(k >= N / 3 && k < N * 2 / 3 && j < N / 3){
                MPI_Bcast(&(A[k][j]), 1, MPI_FLOAT, 2, MPI_COMM_WORLD);
            }
            if(k >= N / 3 && k < N * 2 / 3 && j >= N / 3 && j < N * 2 / 3){
                MPI_Bcast(&(A[k][j]), 1, MPI_FLOAT, 3, MPI_COMM_WORLD);
            }
            if(k >= N / 3 && k < N * 2 / 3 && j >= N * 2 / 3){
                MPI_Bcast(&(A[k][j]), 1, MPI_FLOAT, 4, MPI_COMM_WORLD);
            }
            if(k >= N * 2 / 3 && j < N / 3){
                MPI_Bcast(&(A[k][j]), 1, MPI_FLOAT, 5, MPI_COMM_WORLD);
            }
            if(k >= N * 2 / 3 && j >= N / 3 && j < N * 2 / 3){
                MPI_Bcast(&(A[k][j]), 1, MPI_FLOAT, 6, MPI_COMM_WORLD);
            }
            if(k >= N * 2 / 3 && j >= N  * 2 / 3){
                MPI_Bcast(&(A[k][j]), 1, MPI_FLOAT, 7, MPI_COMM_WORLD);
            }
        }
        int i = k + 1;
        if(myid == 0){
            if(k + 1 >= N / 3) continue;
            for(int i = k + 1;i < N / 3; i++){
                for(int j = k + 1;j < N * 2 / 3; j++)
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }
        if(myid == 1){
            if(k + 1 >= N / 3) continue;
            for(int i = k + 1;i < N / 3; i++){
                for(int j = N * 2 / 3;j < N; j++)
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }
        if(myid == 2){
            if(k + 1 >= N / 3) continue;
            for(int i = N / 3;i < N * 2 / 3; i++){
                for(int j = k + 1;j < N / 3; j++)
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }
        if(myid == 3){
            if(k + 1 >= N * 2 / 3) continue;
            if(k + 1 <= N / 3){
                for(int i = N / 3;i < N * 2 / 3; i++){
                    for(int j = N / 3;j < N * 2 / 3; j++)
                        A[i][j] = A[i][j] - A[i][k] * A[k][j];
                }
            }
            else{
                for(int i = k + 1;i < N * 2 / 3; i++){
                    for(int j = k + 1;j < N * 2 / 3; j++)
                        A[i][j] = A[i][j] - A[i][k] * A[k][j];
                }
            }
        }
        if(myid == 4){
            if(k + 1 >= N * 2 / 3) continue;
            for(int i = k + 1;i < N * 2 / 3; i++){
                for(int j = N * 2 / 3;j < N; j++)
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }
        if(myid == 5){
            if(k + 1 >= N / 3) continue;
            for(int i = N * 2 / 3;i < N; i++){
                for(int j = k + 1;j < N / 3; j++)
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }
        if(myid == 6){
            if(k + 1 >= N * 2 / 3) continue;
            for(int i = N * 2 / 3;i < N; i++){
                for(int j = k + 1;j < N * 2 / 3; j++)
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }
        if(myid == 7){
            for(int i = N * 2 / 3;i < N; i++){
                for(int j = N * 2 / 3;j < N; j++)
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }
    }
    //将计算结果汇总到0号进程
    if (myid != 0) {
        //各进程发送自己负责计算的那些行
        //for (int i = myid; i < N; i += n_threads) {
         //   MPI_Send(A[i], N, MPI_FLOAT, 0, i, MPI_COMM_WORLD);
        //}
    }
    else {
        // 0号进程依次接收
        //for (int i = 0; i < N; i++) {
         //   if (i % n_threads != 0) {
          //      MPI_Recv(A[i], N, MPI_FLOAT, i % n_threads, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         //   }
      //  }

        //计时结束
        double te = MPI_Wtime();
        cout << "N:" << N << ",Time:" << te - ts << "s";
        //Print();
    }
    MPI_Finalize();
    return 0;
}
