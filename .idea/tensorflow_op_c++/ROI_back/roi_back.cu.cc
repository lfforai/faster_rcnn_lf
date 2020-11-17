// kernel_example.cu.cc
#pragma once

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "roi_back.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;

#define IN0AT(z,index_pooled_H,index_pooled_W,i) (in0+z*stride_pooled_4+index_pooled_H*pooled_W*4+index_pooled_W*4+i)
#define IN1AT(z,index_pooled_H,index_pooled_W) (in1+z*stride_pooled+index_pooled_H*pooled_W+index_pooled_W)
#define OUT0At(z,y,x)  (out0+z*H_W+y*W+x)
using GPUDevice = Eigen::GpuDevice;

//in0:rercord for Back Propagation=[C,pooled_h,pooled_w,4],4:[index_H,index_W,frac(H),frac(W)] for back
//in1:grad_of_output=[C,pooled_h,pooled_w]
//out0:grad_of_input =[C,H,W]
template <typename T>
__global__ void ROI_child(const T* in0,const T* in1,volatile T *out0,int pooled_H,int pooled_W,int C,int H,int W)
{   int  index=blockIdx.x*blockDim.x+threadIdx.x;
    int  pitch_size=blockDim.x*gridDim.x;
    int  len=C*pooled_W*pooled_H;
    int  H_W=H*W;

    //stride
    int stride_pooled=pooled_W*pooled_H;
    int stride_pooled_4=pooled_W*pooled_H*4;

    //index of in0
    int index_pooled_W;
    int index_pooled_H;
    int z;
    int HW;

    //index of featuremap
    float frac_x;
    float frac_y;
    int x;
    int y;

    float temp;
    for(int i=index;i<len; i=pitch_size+i)
       { //index of in0:rercord for Back Propagation=[C,pooled_h,pooled_w,4],4:[index_H,index_W,frac(H),frac(W)] for back
         z=(int)(i/stride_pooled);
         HW= (int)fmodf((float)i,(float)stride_pooled);
		 index_pooled_H=(int)(HW/pooled_W);
         index_pooled_W=(int)fmodf((float)HW,(float)pooled_W);

         //index of featuremap
         y=(int)*IN0AT(z,index_pooled_H,index_pooled_W,0);//h
         x=(int)*IN0AT(z,index_pooled_H,index_pooled_W,1);//w
         frac_y=*IN0AT(z,index_pooled_H,index_pooled_W,2);//frac_y
         frac_x=*IN0AT(z,index_pooled_H,index_pooled_W,3);//frac_x

         temp=*IN1AT(z,index_pooled_H,index_pooled_W);//grad valuep

         atomicAdd((T*)OUT0At(z,y,x),temp*(1.0-frac_y)*(1.0-frac_x));//(y,x)
         //printf("temp: %f \n",*OUT0At(z,y,x));
         atomicAdd((T*)OUT0At(z,y,x+1),temp*(1.0-frac_y)*frac_x);//(y,x+1)
         atomicAdd((T*)OUT0At(z,y+1,x+1),temp*frac_y*frac_x);//(y+1,x+1)
         atomicAdd((T*)OUT0At(z,y+1,x),temp*frac_y*(1-frac_x));//(y+1,x)
       }
    __syncthreads();
}

//in0:rercord for Back Propagation=[pitch,C,pooled_h,pooled_w,4]
//in1:grad_of_output=[pitch,C,pooled_h,pooled_w]
//out0:grad_of_input =[C,H,W]
template <typename T>
__global__ void ROI_parent(const T* in0,const T *in1,volatile T *out0,int roi_num,int pooled_H,int pooled_W,int C,int H,int W)
{
    int  index=blockIdx.x*blockDim.x+threadIdx.x;
    int  pitch_size=blockDim.x*gridDim.x;

    int  stride_out0=C*pooled_H*pooled_W*4;
    int  stride_out1=C*pooled_H*pooled_W;

    int  gridsize=1;
    int  threadsize=1024;
    for(int i=index;i<roi_num; i=pitch_size+i){
        ROI_child<T><<<gridsize,threadsize>>>(in0+stride_out0*i,in1+stride_out1*i,out0,pooled_H,pooled_W,C,H,W);
        __syncthreads();
	}
    cudaDeviceSynchronize();
}

// Define the GPU implementation that launches the CUDA kernel.
//in0:rercord for Back Propagation=[pitch,C,pooled_h,pooled_w,4]
//in1:grad_of_output=[pitch,C,pooled_h,pooled_w]
//out0:grad_of_input =[C,H,W]
template <typename T>
void ExampleFunctor<GPUDevice, T>::operator()(
		const GPUDevice& d,int roi_num,int pooled_H,int pooled_W,int C,int H,int W, const T* in0,const T* in1,volatile T* out0)
{
  int thread_per_block = 32;
  int block_count = 1;
  ROI_parent<T><<<block_count,thread_per_block,0,d.stream()>>>(in0,in1,out0,roi_num,pooled_H,pooled_W,C,H,W);
  cudaStreamSynchronize(d.stream());
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct ExampleFunctor<GPUDevice, float>;
#endif  // GOOGLE_CUDA
