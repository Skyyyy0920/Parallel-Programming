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
void Print(){
    for(int i = 0;i < N; i++){
        for(int j = 0;j < N; j++){
            cout<<A[i][j]<<" ";
        }
        cout<<endl;
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
        if(k % n_threads == myid){
            for(int j = k + 1;j < N; j++){
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for(int j = 0;j < n_threads; j++){
                if(j == myid) continue;
                MPI_Send(&A[k][0],N,MPI_FLOAT,j,100 - myid,MPI_COMM_WORLD);
            }
        }
        else{
            MPI_Recv(&A[k][0],N,MPI_FLOAT,k % n_threads,100 - (k % n_threads),MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        if(r2 >= k + 1 && r1 < k + 1){
            for(int i = k + 1;i <= r2; i++){
                for(int j = k + 1;j < N; j++){
                    A[i][j] = A[i][j] - A[k][j] * A[i][k];
                }
                A[i][k] = 0;
            }
        }
        if(r1 >= k + 1){
            for(int i = r1;i <= r2 && i < N; i++){
                for(int j = k + 1;j < N; j++){
                    A[i][j] = A[i][j] - A[k][j] * A[i][k];
                }
                A[i][k] = 0;
            }
        }

        /*if(myid == 0){
            for(int i = k + 1;i < N; i++){
                for(int j = k + 1;j < N; j++){
                    A[i][j] = A[i][j] - A[k][j] * A[i][k];
                }
            A[i][k] = 0;
            }
        }*/
        /*int i = k + 1;
        if(myid != (i % n_threads))
        for(int i = k + 1 + myid;i < N; i += n_threads){
            for(int j = k + 1;j < N; j++){
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
            }
            A[i][k] = 0;
            for(int j = 0;j < n_threads; j++){
                if(j == myid) continue;
                MPI_Send(&A[i][0],N,MPI_FLOAT,j,1000 - myid,MPI_COMM_WORLD);
            }
        }
        for(int i = k + 1;i < N; i++){
            if((i - (k + 1 + myid)) % n_threads != 0){
                int source = 0;
                while((i - (k + 1 + myid)) % n_threads == 0) source++;
                MPI_Recv(&A[i][0],N,MPI_FLOAT,source,1000 - source,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
        }*/
    }
    //cout<<myid<<endl;
    //将计算结果汇总到0号进程
    if (myid != 0) {
        //各进程发送自己负责计算的那些行
       // for (int i = myid; i < N; i += n_threads) {
         //   MPI_Send(A[i], N, MPI_INT, 0, i, MPI_COMM_WORLD);
        //}
    }
    else {
        // 0号进程依次接收
      //  for (int i = 0; i < N; i++) {
         //   if (i % n_threads != 0) {
          ////      MPI_Recv(A[i], N, MPI_INT, i % n_threads, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          //  }
       // }

        //计时结束
        double te = MPI_Wtime();
        cout<<"N:"<<N<<",Time:"<<te - ts<<"s";
        //Print();
    }
    MPI_Finalize();
    return 0;
}
