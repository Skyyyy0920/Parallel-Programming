#include <iostream>
#include<arm_neon.h>
#include<sys/time.h>
using namespace std;
const int maxN = 500;
float32_t A[maxN][maxN];
float32x4_t va = vmovq_n_f32(0);
float32x4_t vx = vmovq_n_f32(0);
float32x4_t vaij = vmovq_n_f32(0);
float32x4_t vaik = vmovq_n_f32(0);
float32x4_t vakj = vmovq_n_f32(0);
void ReSet(){
    for(int i = 0;i < maxN; i++){
        for(int j = 0;j < i; j++)
            A[i][j] = 0;
        A[i][i] = 1.0;
        for(int j = i + 1;j < maxN; j++)
            A[i][j] = rand();
    }
    for(int k = 0;k < maxN; k++)
        for(int i = k + 1;i < maxN; i++)
            for(int j = 0;j < maxN; j++)
                A[i][j] += A[k][j];
}
void LU(){
    for(int k = 0;k < maxN; k++){
        float32x4_t vt = vmovq_n_f32(A[k][k]);
        int j = k + 1;
        for(;j + 4 <= maxN; j += 4){
            va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va,vt);
            vst1q_f32(&A[k][j],va);
        }
        for(;j < maxN; j++){
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < maxN; i++){
            float32x4_t vaik = vmovq_n_f32(A[i][k]);
            int j = k + 1;
            for(;j + 4 <= maxN;j += 4){
                vakj = vld1q_f32(&A[k][j]);
                vaij = vld1q_f32(&A[i][j]);
                vx = vmulq_f32(vakj,vaik);
                vaij = vsubq_f32(vaij,vx);
                vst1q_f32(&A[i][j],vaij);
            }
            for(;j < maxN; j++){
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
            }
            A[i][k] = 0.0;
        }
    }
}
void Align_LU(){
    for(int k = 0;k < maxN; k++){
        float32x4_t vt = vmovq_n_f32(A[k][k]);
        int j = k + 1;
        while((k * maxN + j) % 4 != 0){//do the alignment
            A[k][j] = A[k][j] * 1.0 / A[k][k];
            j++;
        }
        for(;j + 4 <= maxN; j += 4){
            va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va,vt);
            vst1q_f32(&A[k][j],va);
        }
        for(;j < maxN; j++){
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < maxN; i++){
            float32x4_t vaik = vmovq_n_f32(A[i][k]);
            int j = k + 1;
            while((k * maxN + j) % 4 != 0){//do the alignment when j % 4 != 0
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
                j++;
            }
            for(;j + 4 <= maxN;j += 4){
                vakj = vld1q_f32(&A[k][j]);
                vaij = vld1q_f32(&A[i][j]);
                vx = vmulq_f32(vakj,vaik);
                vaij = vsubq_f32(vaij,vx);
                vst1q_f32(&A[i][j],vaij);
            }
            for(;j < maxN; j++){
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
            }
            A[i][k] = 0.0;
        }
    }
}
void Division(){
    for(int k = 0;k < maxN; k++){
        float32x4_t vt = vmovq_n_f32(A[k][k]);
        int j = k + 1;
        for(;j + 4 <= maxN; j += 4){
            va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va,vt);
            vst1q_f32(&A[k][j],va);
        }
        for(;j < maxN; j++){
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < maxN; i++){
            int j = k + 1;
            for(;j < maxN; j++){
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
    }
}
void Align_Division(){
    for(int k = 0;k < maxN; k++){
        float32x4_t vt = vmovq_n_f32(A[k][k]);
        int j = k + 1;
        while((k * maxN + j) % 4 != 0){//do the alignment when j % 4 != 0
            A[k][j] = A[k][j] * 1.0 / A[k][k];
            j++;
        }
        for(;j + 4 <= maxN; j += 4){
            va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va,vt);
            vst1q_f32(&A[k][j],va);
        }
        for(;j < maxN; j++){
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < maxN; i++){
            int j = k + 1;
            for(;j < maxN; j++){
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
    }
}
void Elimination(){
    for(int k = 0;k < maxN; k++){
        for(int j = k + 1;j < maxN; j++){
            A[k][j] = A[k][j] * 1.0 / A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < maxN; i++){
            float32x4_t vaik = vmovq_n_f32(A[i][k]);
            int j = k + 1;
            for(;j + 4 <= maxN;j += 4){
                vakj = vld1q_f32(&A[k][j]);
                vaij = vld1q_f32(&A[i][j]);
                vx = vmulq_f32(vakj,vaik);
                vaij = vsubq_f32(vaij,vx);
                vst1q_f32(&A[i][j],vaij);
            }
            for(;j < maxN; j++){
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
            }
            A[i][k] = 0.0;
        }
    }
}
void Align_Elimination(){
    for(int k = 0;k < maxN; k++){
        for(int j = k + 1;j < maxN; j++){
            A[k][j] = A[k][j] * 1.0 / A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < maxN; i++){
            float32x4_t vaik = vmovq_n_f32(A[i][k]);
            int j = k + 1;
            while((k * maxN + j) % 4 != 0){//do the alignment when j % 4 != 0
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
                j++;
            }
            for(;j + 4 <= maxN;j += 4){
                vakj = vld1q_f32(&A[k][j]);
                vaij = vld1q_f32(&A[i][j]);
                vx = vmulq_f32(vakj,vaik);
                vaij = vsubq_f32(vaij,vx);
                vst1q_f32(&A[i][j],vaij);
            }
            for(;j < maxN; j++){
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
            }
            A[i][k] = 0.0;
        }
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
    cout<<"LU Method, N: "<<maxN<<", Time: "<<(tail.tv_sec-head.tv_sec)*1000.0+(tail.tv_usec-head.tv_usec)/1000.0<<"ms"<<endl;
 //   ReSet();
  //  gettimeofday(&head,NULL);
  //  Division();
  //  gettimeofday(&tail,NULL);
  //  cout<<"Devision Method, N: "<<maxN<<", Time: "<<(tail.tv_sec-head.tv_sec)*1000+(tail.tv_usec-head.tv_usec)/1000<<"ms"<<endl;
  //  ReSet();
   // gettimeofday(&head,NULL);
   // Elimination();
   // gettimeofday(&tail,NULL);
   // cout<<"Elimination Method, N: "<<maxN<<", Time: "<<(tail.tv_sec-head.tv_sec)*1000+(tail.tv_usec-head.tv_usec)/1000<<"ms"<<endl;
  //  ReSet();
  //  gettimeofday(&head,NULL);
  //  Align_LU();
  //  gettimeofday(&tail,NULL);
  //  cout<<"LU Align Method, N: "<<maxN<<", Time: "<<(tail.tv_sec-head.tv_sec)*1000+(tail.tv_usec-head.tv_usec)/1000<<"ms"<<endl;


    return 0;
}
