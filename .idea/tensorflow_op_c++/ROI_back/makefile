################################################################################
# Automatically-generated file. Do not edit!
################################################################################
-include ../makefile.init

RM := rm
OBJS :=roi_back.o roi_back.cu.o  roi_back.dlink.cu.o roi_back.cu.a roi_back.so
# Add inputs and outputs from these tool invocations to the build variables 
-include sources.mk
-include src/subdir.mk
-include subdir.mk
-include objects.mk

-include ../makefile.defs
TF_CFLAGS :=-I/root/anaconda3/envs/pyp3/lib/python3.7/site-packages/tensorflow/include
TF_LFLAGS :=-L/root/anaconda3/envs/pyp3/lib/python3.7/site-packages/tensorflow

# All Target
all: ROI_layer_backward

# Tool invocations
ROI_layer_backward:
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC linker'
	nvcc --gpu-architecture=compute_50  -ccbin  g++ -rdc=true -D_MWAITXINTRIN_H_INCLUDED -I"/usr/local/cuda-11.0/samples/common/inc" -std=c++11 -c roi_back.cu.cc -o roi_back.cu.o  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC  --expt-relaxed-constexpr  
	nvcc --gpu-architecture=compute_50  -ccbin  g++ -Xcompiler '-fPIC' -dlink -o roi_back.dlink.cu.o roi_back.cu.o -lcudadevrt -lcudart -L/usr/local/cuda/lib64 
	nvcc --gpu-architecture=compute_50  -ccbin  g++ -Xcompiler '-fPIC' -lib -o roi_back.cu.a roi_back.cu.o roi_back.dlink.cu.o  
	g++ -fPIC -std=c++11 -shared -o roi_back.so roi_back.cc roi_back.cu.a  ${TF_CFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1 -lcudadevrt -lcudart $(TF_LFLAGS) -L /usr/local/cuda/lib64/ -l:libtensorflow_framework.so.2  -I"/usr/local/cuda-11.0/samples/common/inc" 
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)
	-@echo ' '

.PHONY: all clean dependents

-include ../makefile.targets
