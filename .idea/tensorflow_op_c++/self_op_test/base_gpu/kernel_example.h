/*
 * kernelexample.h
 *
 *  Created on: 2020年11月3日
 *      Author: root
 */

#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/types.h"

template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct ExampleFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};
#endif

#endif /* KERNEL_EXAMPLE_H_ */
