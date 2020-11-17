#include "roi_layer.h"

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("RoiLayer")
.Attr("T:{float}")
.Attr("shape_pool:shape = { dim { size: 7 } dim { size: 7 } }") //pooled_H and pool_W
.Attr("spatial_scale:float = 0.0625") //0.0625 # 1/16
.Input("input0:T")       //[1,C,H,W] image_feature,C=512 or 256
.Input("input1:T")       //[pitch,x_left,y_left,x_right,y_right],protocl=type [pitch,4]
.Output("output0:T")     //rercord for Back Propagation=[pitch,C,feature_num,feature_num,4]
                         //4:[index_H,index_W,frac(H),frac(W)]
.Output("output1:T");    //ROI_feature[pitch,C,feature_num,feature_num]

// CPU specialization of actual computation.
template <typename T>
struct ExampleFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d,int batch_num,int batch_size,int roi_num,int pooled_H,int pooled_W,float spatial_scale,int C,int H,int W, const T* in0,const T* in1,volatile T* out0,T* out1) {
	int size =C*H*W;
    for (int i = 0; i < size; ++i) {
      out0[i] = 3 * in0[i];
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class RoiLayerOp : public OpKernel {
 public:
  explicit  RoiLayerOp(OpKernelConstruction* context) : OpKernel(context) {
	    // Get the index of the value to preserve
	    OP_REQUIRES_OK(context,
	                   context->GetAttr("spatial_scale", &spatial_scale_));
	    // Check that preserve_index is positive
	    OP_REQUIRES(context, spatial_scale_ <=1.0,
	                errors::InvalidArgument("Need spatial_scale <=1.0, got ",
	                		                                 spatial_scale_));

	    OP_REQUIRES_OK(context,
	                   context->GetAttr("shape_pool", &shape_pool_));
  }

  void Compute(OpKernelContext* context) override {

    // Grab the input tensor
    const Tensor& input_tensor0 = context->input(0); //[1,C,H,W] image_feature,C=512 or 256
    OP_REQUIRES(context, input_tensor0.dims() == 4,
                       errors::InvalidArgument("input0 must be 4-dimensional=[1,C,H,W]"));

    const Tensor& input_tensor1 = context->input(1); //[pitch,x_left,y_left,x_right,y_right] protocl
    OP_REQUIRES(context, input_tensor1.dims() == 2,
                       errors::InvalidArgument("input1 must be 2-dimensional=[pitch,[x_left,y_left,x_right,y_right]]"));
    OP_REQUIRES(context,input_tensor1.NumElements() <= tensorflow::kint32max,errors::InvalidArgument("Too many elements in input1"));

    //C,H.W
    int C=input_tensor0.dim_size(1);
    int H=input_tensor0.dim_size(2);
    int W=input_tensor0.dim_size(3);
    int pooled_h=(int)shape_pool_.dim_size(0);//pooled_H
    int pooled_w=(int)shape_pool_.dim_size(1);//pooled_W

    //rercord for Back Propagation=[pitch,C,pooled_h,fpooled_w,4]
    //4:[index_H,index_W,frac(H),frac(W)] shape
    int dims0[5];
    dims0[0]=(int)input_tensor1.dim_size(0); //pitch
    int roi_num=dims0[0];
    dims0[1]=C; //C
    dims0[2]=pooled_h; //pooled_h
    dims0[3]=pooled_w; //pooled_w
    dims0[4]=4; //4:[index_H,index_W,frac(H),frac(W)]
    TensorShape output_shape0;
    TensorShapeUtils::MakeShape(dims0, 5, &output_shape0);

    //ROI_feature[pitch,C,feature_num,feature_num] shape
    int dims1[4];
    dims1[0]=(int)input_tensor1.dim_size(0); //pitch
    dims1[1]=C; //C
    dims1[2]=pooled_h; //feature_num
    dims1[3]=pooled_w; //feature_num
    TensorShape output_shape1;
    TensorShapeUtils::MakeShape(dims1, 4, &output_shape1);

    //Create an output tensor=0,1
    Tensor* output_tensor0 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0,output_shape0,&output_tensor0));

    Tensor* output_tensor1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1,output_shape1,&output_tensor1));

    //init tex num
    int batch_num=0;
    int batch_size=1;
//    printf("feature_num_h:%d \n",feature_num_h);
//    printf("feature_num_w:%d \n",feature_num_w);

    ExampleFunctor<Device, T>()(
                context->eigen_device<Device>(),
				batch_num,
				batch_size,
				roi_num,
				pooled_h,
				pooled_w,
				spatial_scale_,
				C,
				H,
				W,
                input_tensor0.flat<T>().data(),  //[1,C,H,W] image_feature,C=512 or 256
			    input_tensor1.flat<T>().data(),  //[pitch,[x_left,y_left,x_right,y_right]],protocl
                output_tensor0->flat<T>().data(),//rercord for Back Propagation=[pitch,C,feature_num,feature_num,4]
                                                 //4:[index_H,index_W,frac(H),frac(W)] for back
				output_tensor1->flat<T>().data() //ROI_feature[pitch,C,feature_num,feature_num]
				);
  }
 private:
   float spatial_scale_;
   TensorShape shape_pool_;
};

//Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("RoiLayer").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
	  RoiLayerOp<CPUDevice, T>);
REGISTER_CPU(float);
//REGISTER_CPU(double);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class ExampleFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("RoiLayer").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
	  RoiLayerOp<GPUDevice, T>);
REGISTER_GPU(float);
//REGISTER_GPU(double);
#endif  // GOOGLE_CUDA


