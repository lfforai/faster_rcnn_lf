// kernel_example.cu.cc
#pragma once

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "roi_layer.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "roi_layer_Texture3D.h"

using namespace tensorflow;

#define OUT1AT(pooled_W_index,pooled_H_index,z) (out1+stride_pooled*z+pooled_H_index*pooled_W+pooled_W_index)
#define OUT0AT(i,pooled_W_index,pooled_H_index,z) (out0+stride_pooled_4*z+pooled_H_index*pooled_W*4+pooled_W_index*4+i)
using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__device__ void ExampleCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 4 * __ldg(in + i);
  }
}

//spatial_scale=1/16,C=512 or 256
template <typename T>
__global__ void ROI_child(const T* in1,T *out0,volatile T *out1,int pooled_H,int pooled_W,float spatial_scale,int C,int H,int W,cudaTextureObject_t texObj)
{   __shared__ float proposal_lx_ly_rx_ry[4];//首选框坐标
    __shared__ float p_xy_int[4];
    __shared__ float p_xy_float[4];
    __shared__ int   p_HW_max[2];
    __shared__ float P_HW_stide[2];

    if(threadIdx.x==0)
	  {   proposal_lx_ly_rx_ry[0]=in1[0]*spatial_scale;//lx---w
		  proposal_lx_ly_rx_ry[1]=in1[1]*spatial_scale;//ly
		  proposal_lx_ly_rx_ry[2]=in1[2]*spatial_scale;//rx---w
		  proposal_lx_ly_rx_ry[3]=in1[3]*spatial_scale;//ry

		  //make sure roi的像素点分隔距离是大过1的
		  p_HW_max[1]=(int)(floorf(proposal_lx_ly_rx_ry[2])-ceilf(proposal_lx_ly_rx_ry[0]))+1; //less(rx)-big(lx)=W
		  p_HW_max[0]=(int)(floorf(proposal_lx_ly_rx_ry[3])-ceilf(proposal_lx_ly_rx_ry[1]))+1; //ry-ly=H

		  //each stride ,H , W
		  P_HW_stide[1]=((proposal_lx_ly_rx_ry[2]-proposal_lx_ly_rx_ry[0])/(p_HW_max[1]+0.0));//W
		  P_HW_stide[0]=((proposal_lx_ly_rx_ry[3]-proposal_lx_ly_rx_ry[1])/(p_HW_max[0]+0.0));//H

//		  p_xy_int[0]=floorf(proposal_lx_ly_rx_ry[0]);//xl_base=fllorf(12.34)=12.0
//		  p_xy_int[1]=floorf(proposal_lx_ly_rx_ry[1]);//yl_base
//		  p_xy_int[2]=floorf(proposal_lx_ly_rx_ry[2]);//xr_base
//		  p_xy_int[3]=floorf(proposal_lx_ly_rx_ry[3]);//yr_base
//
//		  p_xy_float[0]=proposal_lx_ly_rx_ry[0]-p_xy_int[0];//xl_float=fllorf(12.34)=12.0
//		  p_xy_float[1]=proposal_lx_ly_rx_ry[1]-p_xy_int[1];//yl_float
//		  p_xy_float[2]=proposal_lx_ly_rx_ry[2]-p_xy_int[2];//xr_float
//		  p_xy_float[3]=proposal_lx_ly_rx_ry[3]-p_xy_int[3];//yr_float
          //printf("%d,%d,%f,%f,%f,%f,%f \n",p_HW_max[1],p_HW_max[0],P_HW_stide[1], P_HW_stide[0],proposal_lx_ly_rx_ry[0],p_xy_int[0],p_xy_float[0]);
	  }
	__syncthreads();

    int  index=blockIdx.x*blockDim.x+threadIdx.x;
    int  pitch_size=blockDim.x*gridDim.x;
    int  len= p_HW_max[1]*p_HW_max[0];

    int index_H;
    int index_W;
    float x;
    float y;

    float pooled_W_scale=floorf(p_HW_max[1]/pooled_W); //W
    float pooled_H_scale=floorf(p_HW_max[0]/pooled_H); //H

    int pooled_W_index;
    int pooled_H_index;
    int stride_pooled=pooled_W*pooled_H;
    int stride_pooled_4=pooled_W*pooled_H*4;
    float temp;//for compare
    float compare;

    //out0: for Back Propagation=[pitch,C,pooled_num,pooled_num,4] 4:[index_H,index_W,frac(H),frac(W)] for back
    //out1: ROI_feature[pitch,C,pooled_num,pooled_num]
    for(int i=index;i<len; i=pitch_size+i)
       { //计算maxpooling使用index_HW整数坐标，而计算tex纹理必须使用xy浮点坐标(backward 也使用该坐标寻找)
    	 index_H=(int)(i/p_HW_max[1]); //h_index
         index_W=(int)fmodf(i,p_HW_max[1]); //w_index
         x=index_W*P_HW_stide[1]+proposal_lx_ly_rx_ry[0];//offset+base :h
         y=index_H*P_HW_stide[0]+proposal_lx_ly_rx_ry[1];//offset+base :w

         //max_pooling
         //eg:p_HW_max[1]=7,pooled_W=3,pooled_w_scale=2
         //index_w to pooled_w_index:(0->0),(1->0),(2->1),(3->1),(4->2),(5->2) but (6->3) out of pooled_W=0,1,2
         //so change (6->2),index of 6 really means last 7
         pooled_W_index=floorf(index_W/pooled_W_scale);
         pooled_H_index=floorf(index_H/pooled_H_scale);

         if (pooled_W_index==pooled_W)
        	 pooled_W_index=pooled_W-1;

         if (pooled_H_index==pooled_H)
             pooled_H_index=pooled_H-1;

    	 for(int z = 0; z < C; ++z)
    	    {temp=tex3D<T>(texObj, x+0.5, y+0.5, z+0.5);
    		 compare=*OUT1AT(pooled_W_index,pooled_H_index,z)-temp;
    	     if(compare<0.0)
    	       {//如果当前值大于原始值，需要修改向后求偏导的out0的信息值
    	    	atomicExch((T*)OUT1AT(pooled_W_index,pooled_H_index,z),temp);
    	    	atomicExch(OUT0AT(0,pooled_W_index,pooled_H_index,z),floorf(x));
    	    	atomicExch(OUT0AT(1,pooled_W_index,pooled_H_index,z),floorf(y));
    	    	atomicExch(OUT0AT(2,pooled_W_index,pooled_H_index,z),x-floorf(x));
    	    	atomicExch(OUT0AT(3,pooled_W_index,pooled_H_index,z),y-floorf(y));
    	       }
//               printf("%f,x=%f,y=%f \n",OUT1AT(pooled_H_index,pooled_W_index),x,y);
		    }
        }
    __syncthreads();
    //printf("%f,%d,%d,%d,%d,%f \n",proposal_lx_ly_r_x_ry[0],C,H,W,tex3D<T>(texObj, 1+0.5, 1+0.5, 1+0.5));
}

// Define the GPU implementation that launches the CUDA kernel.
//in0:[1,C,H,W] image_feature,C=512 or 256
//in1:[pitch,[x_left,y_left,x_right,y_right]],protocl
//out0: for Back Propagation=[pitch,C,pooled_num,pooled_num,4] 4:[index_H,index_W,frac(H),frac(W)] for back
//out1: ROI_feature[pitch,C,pooled_num,pooled_num]
template <typename T>
__global__ void ROI_parent(const T* in1,T *out0,volatile T *out1,int roi_num,int pooled_H,int pooled_W,float spatial_scale,int C,int H, int W,cudaTextureObject_t texObj)
{
    int  index=blockIdx.x*blockDim.x+threadIdx.x;
    int  pitch_size=blockDim.x*gridDim.x;

//	int  stride_CHW=C*H*W;
    int  stride_out0=C*pooled_H*pooled_W*4;
    int  stride_out1=C*pooled_H*pooled_W;

    int  gridsize=1;
    int  threadsize=1024;
    for(int i=index;i<roi_num; i=pitch_size+i) {
        ROI_child<T><<<gridsize,threadsize>>>(in1+4*i,out0+stride_out0*i,out1+stride_out1*i,pooled_H,pooled_W,spatial_scale,C,H,W,texObj);
        __syncthreads();
	}
    cudaDeviceSynchronize();
}

// Define the GPU implementation that launches the CUDA kernel.
//in0:[1,C,H,W] image_feature,C=512 or 256
//in1:[pitch,[x_left,y_left,x_right,y_right]],protocl
//out0: for Back Propagation=[pitch,C,pooled_num,pooled_num,4] 4:[index_H,index_W,frac(H),frac(W)] for back
//out1: ROI_feature[pitch,C,pooled_num,pooled_num]
template <typename T>
void ExampleFunctor<GPUDevice, T>::operator()(
		const GPUDevice& d,int batch_num,int batch_size,int roi_num,int pooled_H,int pooled_W,float spatial_scale,int C,int H,int W, const T* in0,const T* in1,T* out0,volatile T* out1)
{
  printf("tensorflow use gpu:\n");
  int thread_per_block = 32;
  int block_count = 1;
  if(batch_num==0)
    init_d_volumeArray(batch_size);
  const cudaExtent volumeSize = make_cudaExtent(W,H,C); //w,h,c
  cudaTextureObject_t &tex=initCuda(batch_num,in0,volumeSize);
  //ExampleCudaKernel<T><<<block_count, thread_per_block, 0, d.stream()>>>(W*H*depth,in,out);
  //d_render<T><<<block_count, thread_per_block, 0, d.stream()>>>(in1,out0,out1,roi_num,pooled_num,C,H,W,tex);
  ROI_parent<T><<<block_count,thread_per_block,0,d.stream()>>>(in1,out0,out1,roi_num,pooled_H,pooled_W,spatial_scale,C,H,W,tex);
  cudaStreamSynchronize(d.stream());
  cleanupCuda(tex,batch_num);
  if(batch_num==batch_size-1)
    delete_d_volumeArray();
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct ExampleFunctor<GPUDevice, float>;
//template __global__ void d_render<float>(float *output,int depth,int imageH, int imageW, cudaTextureObject_t	texObj);
//template struct ExampleFunctor<GPUDevice, double>;
#endif  // GOOGLE_CUDA
