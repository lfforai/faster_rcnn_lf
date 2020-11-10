#include "kernel_example.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Example")
.Attr("T:{float}")
.Input("input: T")
.Output("output: T");

// CPU specialization of actual computation.
template <typename T>
struct ExampleFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d,int batch_num,int batch_size,int depth,int H,int W, const T* in, T* out) {
	int size =depth*H*W;
    for (int i = 0; i < size; ++i) {
      out[i] = 3 * in[i];
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class ExampleOp : public OpKernel {
 public:
  explicit ExampleOp(OpKernelConstruction* context) : OpKernel(context) {
	    // Get the index of the value to preserve
//	    OP_REQUIRES_OK(context,
//	                   context->GetAttr("stream_num", &stream_num_));
//	    // Check that preserve_index is positive
//	    OP_REQUIRES(context, stream_num_ >= 1,
//	                errors::InvalidArgument("Need preserve_index >= 1, got ",
//	                		                                   stream_num_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    OP_REQUIRES(context, input_tensor.dims() == 3,
                       errors::InvalidArgument("data must be 3-dimensional"));

    int depth=input_tensor.dim_size(0);
    int H=input_tensor.dim_size(1);
    int W=input_tensor.dim_size(2);

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    int batch_num=0;
    int batch_size=50;

    ExampleFunctor<Device, T>()(
                context->eigen_device<Device>(),
				batch_num,
				batch_size,
				depth,
				H,
				W,
                input_tensor.flat<T>().data(),
                output_tensor->flat<T>().data());
  }
 private:
   int stream_num_;
};

//Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ExampleOp<CPUDevice, T>);
REGISTER_CPU(float);
//REGISTER_CPU(double);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class ExampleFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ExampleOp<GPUDevice, T>);
REGISTER_GPU(float);
//REGISTER_GPU(double);
#endif  // GOOGLE_CUDA


