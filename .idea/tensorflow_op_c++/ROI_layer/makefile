################################################################################
# Automatically-generated file. Do not edit!
################################################################################
-include ../makefile.init

RM := rm
OBJS :=roi_layer.cu.o roi_layer.dlink.cu.o roi_layer.so roi_layer_Texture3D.cu.o roi_layer.cu.a
# Add inputs and outputs from these tool invocations to the build variables 
-include sources.mk
-include src/subdir.mk
-include subdir.mk
-include objects.mk

-include ../makefile.defs
TF_CFLAGS :=-I/root/anaconda3/envs/pyp3/lib/python3.7/site-packages/tensorflow/include
TF_LFLAGS :=-L/root/anaconda3/envs/pyp3/lib/python3.7/site-packages/tensorflow

# All Target
all: ROI_cuda
#g++ -fPIC -std=c++11 -shared -o roi_layer.so roi_layer.cc roi_layer.cu.a simpleTexture3D_kernel.cu.o  ${TF_CFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1 -lcudadevrt -lcudart $(TF_LFLAGS) -L /usr/local/cuda/lib64/ -l:libtensorflow_framework.so.2  -I"/usr/local/cuda-11.0/samples/common/inc" $(OBJS) $(USER_OBJS) $(LIBS)
	
# Tool invocations
ROI_cuda:
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC linker'
	nvcc --gpu-architecture=compute_50  -ccbin  g++ -rdc=true -D_MWAITXINTRIN_H_INCLUDED -I"/usr/local/cuda-11.0/samples/common/inc" -std=c++11 -c roi_layer.cu.cc -o roi_layer.cu.o  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC  --expt-relaxed-constexpr  
	nvcc --gpu-architecture=compute_50  -ccbin  g++  -D_MWAITXINTRIN_H_INCLUDED -I"/usr/local/cuda-11.0/samples/common/inc" -std=c++11 -c roi_layer_Texture3D.cu.cc -o roi_layer_Texture3D.cu.o  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler  -fPIC --expt-relaxed-constexpr  
	nvcc --gpu-architecture=compute_50  -ccbin  g++ -Xcompiler '-fPIC' -dlink -o roi_layer.dlink.cu.o roi_layer.cu.o -lcudadevrt -lcudart -L/usr/local/cuda/lib64 
	nvcc --gpu-architecture=compute_50  -ccbin  g++ -Xcompiler '-fPIC' -lib -o roi_layer.cu.a roi_layer.cu.o roi_layer.dlink.cu.o  
	g++ -fPIC -std=c++11 -shared -o roi_layer.so roi_layer.cc roi_layer.cu.a roi_layer_Texture3D.cu.o  ${TF_CFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1 -lcudadevrt -lcudart $(TF_LFLAGS) -L /usr/local/cuda/lib64/ -l:libtensorflow_framework.so.2  -I"/usr/local/cuda-11.0/samples/common/inc" 
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)
	-@echo ' '

.PHONY: all clean dependents

-include ../makefile.targets
