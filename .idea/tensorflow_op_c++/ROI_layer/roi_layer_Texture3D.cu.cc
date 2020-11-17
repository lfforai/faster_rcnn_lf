/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef _SIMPLETEXTURE3D_KERNEL_CU_
#define _SIMPLETEXTURE3D_KERNEL_CU_

#include "roi_layer_Texture3D.h"

typedef unsigned int  uint;
typedef unsigned char uchar;

cudaArray** d_volumeArray;
//cudaTextureObject_t tex;
//cudaArray *d_volumeArray;

//extern "C"  //d_volumeArray has already  data
//cudaTextureObject_t* setTextureFilterMode(cudaArray *d_volumeArray)
//{   cudaTextureObject_t* tex;
//    memset(tex,0,sizeof(cudaTextureObject_t));
////    if (tex)
////    {
////        checkCudaErrors(cudaDestroyTextureObject(tex));
////    }
//    cudaResourceDesc            texRes;
//    memset(&texRes,0,sizeof(cudaResourceDesc));
//
//    texRes.resType            = cudaResourceTypeArray;
//    texRes.res.array.array    = d_volumeArray; //cudaarray
//
//    cudaTextureDesc             texDescr;
//    memset(&texDescr,0,sizeof(cudaTextureDesc));
//
//    texDescr.normalizedCoords = false;
//    texDescr.filterMode       = cudaFilterModeLinear;
//    texDescr.addressMode[0] = cudaAddressModeWrap;
//    texDescr.addressMode[1] = cudaAddressModeWrap;
//    texDescr.addressMode[2] = cudaAddressModeWrap;
//    texDescr.readMode = cudaReadModeElementType;
//
//    checkCudaErrors(cudaCreateTextureObject(tex, &texRes, &texDescr, NULL));
//    return tex;    // 3D texture
//}
extern "C"
void init_d_volumeArray(int num){
	d_volumeArray=(cudaArray**)malloc(num*sizeof(*d_volumeArray));
}

extern "C"
void delete_d_volumeArray(){
	 delete d_volumeArray;
}

extern "C"  //copy data from h_volume to d_volumeArray,no data in d_volumeArray
//volumeSize=x*y*z ，        x,y,z  is 3d latitudes
//template <typename T>
cudaTextureObject_t& initCuda(int num_Array,const float *h_volume, cudaExtent volumeSize)
{   cudaTextureObject_t* tex;
    tex=(cudaTextureObject_t*)malloc(sizeof(cudaTextureObject_t));

    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMalloc3DArray(d_volumeArray+num_Array, &channelDesc, volumeSize));

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void *)h_volume, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray[num_Array];
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyDefault;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc            texRes;
    memset(&texRes,0,sizeof(cudaResourceDesc));

    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = d_volumeArray[num_Array];

    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false; // access with normalized texture coordinates
    texDescr.filterMode       = cudaFilterModeLinear; // linear interpolation
    // wrap texture coordinates
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.addressMode[2] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(tex, &texRes, &texDescr, NULL));
    return *tex;
}

extern "C"
void cleanupCuda(cudaTextureObject_t& tex,int num_Array)
{
    if (tex)
    {
        checkCudaErrors(cudaDestroyTextureObject(tex));
    }

    if (d_volumeArray)
    {
    	checkCudaErrors(cudaFreeArray(d_volumeArray[num_Array]));
    }
}

//extern "C"
//template cudaTextureObject_t* initCuda<float>(cudaArray *d_volumeArray,const float *h_volume, cudaExtent volumeSize);
#endif // #ifndef _SIMPLETEXTURE3D_KERNEL_CU_
