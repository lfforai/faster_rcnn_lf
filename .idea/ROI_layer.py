import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
print(tf.sysconfig.get_include())
print(tf.sysconfig.get_lib())
# g++ -std=c++11 -shared -o kernel_example.so kernel_example.cc kernel_example.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -l:libtensorflow_framework.so.2

#test how to build and use the self creating ops in tensorflow2.4 with ncc g++ ..crossing building tool
def jx(num):
    return_list=[]
    for  i in range(num):
        lx=np.random.random()*50.0+250
        ly=np.random.random()*60.0+270
        rx=np.random.random()*50.0+1000
        ry=np.random.random()*50.0+1300
        return_list.append([lx,ly,rx,ry])
    return return_list

with tf.device('gpu'):
     roi_layer_module_gpu = tf.load_op_library('/root/eclipse-workspace/ROI_layer/roi_layer.so')
#     a=1*3*100*100
#     b=100
#     b=np.array(jx(b))
#     # print(b)
#     # print(np.reshape(np.arange(a).astype(np.float32),newshape=(1,3,10,10)))
#     # output0
#     output0,output1=roi_layer_module_gpu.roi_layer(np.reshape(np.arange(a).astype(np.float32),newshape=(1,3,100,100)),
#                                                   np.reshape(np.array(b).astype(np.float32),newshape=(100,4)))
#     # print("in gpu:",output1)
#     # print("in gpu",output0)

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

#basetest
@ops.RegisterGradient("RoiLayer")
def _roi_layer_grad(op, grad0,grad1):
    #grad0->out0 ,out0 no need backward only parameter ,so no grad[0]
    backinfo=op.outputs[0]
    #out0::
    #for Back Propagation=[pitch,C,pooled_num,pooled_num,4] 4:[index_H,index_W,frac(H),frac(W)]
    back_dy=grad1;              #[1,C,pooled_num,pooled_num]
    shape_o=op.inputs[0].get_shape()
    # print("op.outputs[0]",op.outputs[0].get_shape())
    # print("back_dy",back_dy.get_shape())
    # print("op.inputs[0]:",shape_o)
    # print("grad1:",grad1.get_shape())
    C=shape_o[1]
    H=shape_o[2]
    W=shape_o[3]
    shape_CHW=(C,H,W)
    # print(shape_CHW)
    with tf.device('gpu'):
         roi_back_module_gpu = tf.load_op_library('/root/eclipse-workspace/Roi_layerback/roi_back.so')
         to_roi_grad=roi_back_module_gpu.roi_back(backinfo,back_dy,CHW=shape_CHW)
    # print("to_roi_grad:",to_roi_grad.get_shape())
    return [to_roi_grad,None]


a=1*3*100*100
x_feature=tf.constant(np.reshape(np.arange(a).astype(np.float32),newshape=(1,3,100,100)))
b=100
b=np.array(jx(b))
x_roi=tf.constant(np.reshape(np.array(b).astype(np.float32),newshape=(100,4)))
with tf.GradientTape() as t:
    t.watch(x_feature)
    z0,z1 =roi_layer_module_gpu.roi_layer(x_feature,x_roi)
    z=tf.reduce_sum(z1)
# Derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x_feature)
print(dz_dx)

