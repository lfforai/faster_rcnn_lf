/*
 * simpleTexture3D_kernel.h
 *
 *  Created on: 2020年11月10日
 *      Author: root
 */
#pragma once

#ifndef SIMPLETEXTURE3D_KERNEL_H_
#define SIMPLETEXTURE3D_KERNEL_H_
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>
#include <helper_math.h>

//extern "C"  //d_volumeArray has already  data
//cudaTextureObject_t* setTextureFilterMode(cudaArray *d_volumeArray);

//extern "C"  //copy data from h_volume to d_volumeArray,no data in d_volumeArray
//volumeSize=x*y*z ，        x,y,z  is 3d latitudes
//template <typename T>
extern "C"
void init_d_volumeArray(int num);

extern "C"
void delete_d_volumeArray();

extern "C"
cudaTextureObject_t& initCuda(int num_Array,const float *h_volume, cudaExtent volumeSize);

extern "C"
void cleanupCuda(cudaTextureObject_t& tex,int num_Array);

#endif /* SIMPLETEXTURE3D_KERNEL_H_ */
