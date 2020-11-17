#include "roi_back.h"

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("RoiBack")
.Attr("T:{float}")
.Attr("CHW:shape") //pooled_H and pool_W
.Input("location:T")     //rercord for Back Propagation=[pitch,C,pooled_h,pooled_w,4]
.Input("grad_output:T")  //grad_of_output=[pitch,C,pooled_h,pooled_w]
.Output("grad_input:T"); //grad_of_input =[C,H,W]

// CPU specialization of actual computation.
template <typename T>
struct ExampleFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d,int batch_num,int pooled_H,int pooled_W,int C,int H,int W, const T* in0,const T* in1,volatile T* out0) {
	int size =C*H*W;
    for (int i = 0; i < size; ++i) {
      out0[i] = 3 * in0[i];
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class RoiBackOp : public OpKernel {
 public:
  explicit  RoiBackOp(OpKernelConstruction* context) : OpKernel(context) {
	  OP_REQUIRES_OK(context,context->GetAttr("CHW", &CHW_));
  }

  void Compute(OpKernelContext* context) override {

    // Grab the input tensor
    const Tensor& input_tensor0 = context->input(0); //rercord for Back Propagation=[pitch,C,pooled_h,pooled_w,4]
    OP_REQUIRES(context, input_tensor0.dims() == 5,
                       errors::InvalidArgument("back input0 must be 5-dimensional=[pitch,C,pooled_h,pooled_w,4]"));

    const Tensor& input_tensor1 = context->input(1); //grad_of_output=[C,pooled_h,pooled_w]
    OP_REQUIRES(context, input_tensor1.dims() == 4,
                       errors::InvalidArgument("back input1 must be 4-dimensional=[pitch,C,pooled_h,pooled_w]"));

    OP_REQUIRES(context,input_tensor1.NumElements() <= tensorflow::kint32max,errors::InvalidArgument("Too many elements in input1"));

    //pitch
    int pitch=input_tensor0.dim_size(0);

    //pooled_H,pooled_W
    int pooled_H=input_tensor1.dim_size(2);//h
	int pooled_W=input_tensor1.dim_size(3);//w

    //C,H.W
    int C=(int)CHW_.dim_size(0);
    int H=(int)CHW_.dim_size(1);
    int W=(int)CHW_.dim_size(2);

    //grad_of_input =[1,C,H,W]
    int dims0[4];
    dims0[0]=1;
    dims0[1]=C;
    dims0[2]=H;
    dims0[3]=W;
    TensorShape output_shape0; //grad_input=[1,C,H,W]
    TensorShapeUtils::MakeShape(dims0, 4, &output_shape0);

    Tensor* grad_input = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0,output_shape0,&grad_input));

    ExampleFunctor<Device, T>()(
                context->eigen_device<Device>(),
				pitch,
				pooled_H,
				pooled_W,
				C,
				H,
				W,
                input_tensor0.flat<T>().data(),  //rercord for Back Propagation=[pitch,C,pooled_h,pooled_w,4]
			    input_tensor1.flat<T>().data(),  //grad_of_output=[C,pooled_h,pooled_w]
				grad_input->flat<T>().data()     //grad_of_input =[C,H,W]
				);
  }
 private:
   TensorShape CHW_;
};

//Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("RoiBack").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
	  RoiBackOp<CPUDevice, T>);
REGISTER_CPU(float);
//REGISTER_CPU(double);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class ExampleFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("RoiBack").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
	  RoiBackOp<GPUDevice, T>);
REGISTER_GPU(float);
//REGISTER_GPU(double);
#endif  // GOOGLE_CUDA


