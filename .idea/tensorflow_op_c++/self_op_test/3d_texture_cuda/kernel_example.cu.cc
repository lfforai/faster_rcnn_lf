// kernel_example.cu.cc
#pragma once
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "kernel_example.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "simpleTexture3D_kernel.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void ExampleCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 4 * __ldg(in + i);
  }
}

template <typename T>
//__global__ void d_render()
__global__ void d_render(T *output,int depth,int imageH, int imageW,cudaTextureObject_t texObj)
{   //mul24==multiy
	float len=depth*imageH*imageW;
	float winWH=imageH*imageW;
	int pitchsize= blockDim.x*gridDim.x;

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int x=1;
    int y=1;
	int z=1;
    int WH;
    float voxel=2.0;
    for(int i = index; i< len; i=i+pitchsize)
    {
    	z=(int)((i)/winWH);
    	WH=(int)fmodf((i),winWH);
        y=(int)((WH)/imageW);
        x=(int)fmodf((WH),imageW);
        voxel = tex3D<T>(texObj, x+0.5, y+0.5, z+0.5);
        //printf("voxel:%f,x=%d,y=%d,z=%d \n",voxel,x,y,z);
        output[int(winWH*z+y*imageW+x)]=voxel;
	}
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void ExampleFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d,int batch_num,int batch_size,int depth,int H,int W,const T* in, T* out) {
  int thread_per_block = 1024;
  int block_count = 20;
  if (batch_num==0)
     init_d_volumeArray(batch_size);
  const cudaExtent volumeSize = make_cudaExtent(W,H,depth);
  cudaTextureObject_t &tex=initCuda(batch_num,in,volumeSize);
  //ExampleCudaKernel<T><<<block_count, thread_per_block, 0, d.stream()>>>(W*H*depth,in,out);
  d_render<T><<<block_count, thread_per_block, 0, d.stream()>>>(out,depth,H,W,tex);
  cudaStreamSynchronize(d.stream());
  cleanupCuda(tex,batch_num);
  if(batch_num==batch_size-1);
    delete_d_volumeArray();
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct ExampleFunctor<GPUDevice, float>;
//template __global__ void d_render<float>(float *output,int depth,int imageH, int imageW, cudaTextureObject_t	texObj);
//template struct ExampleFunctor<GPUDevice, double>;

#endif  // GOOGLE_CUDA
