/*
 * kernelexample.h
 *
 *  Created on: 2020年11月3日
 *      Author: root
 */
#pragma once
#ifndef ROI_LAYER_H_
#define ROI_LAYER_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/types.h"
//#include  "simpleTexture3D_kernel.h"

template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d,int batch_num,int batch_size,int roi_num,int pooled_H,int pooled_W,float spatial_scale,int C,int H,int W, const T* in0,const T* in1,T* out0,volatile T* out1);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct ExampleFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d,int batch_num,int batch_size,int roi_num,int pooled_H,int pooled_W,float spatial_scale,int C,int H,int W, const T* in0,const T* in1,T* out0,volatile T* out1);
};
#endif

#endif /* ROI_LAYER_H_ */
