import tensorflow as tf
print(tf.sysconfig.get_include())
print(tf.sysconfig.get_lib())

#test how to build and use the self creating ops in tensorflow2.4 with ncc g++ ..crossing building tool
zero_out_module_gpu = tf.load_op_library('/root/eclipse-workspace_lf/ROI_tf_lf/kernel_example.so')
print("in gpu:",zero_out_module_gpu.example([[1, 2], [3, 4]]))

zero_out_module_cpu = tf.load_op_library('/root/eclipse-workspace_lf/ZeroOut/zero_out.so')
print("in cpu:",zero_out_module_cpu.zero_out([[1.0, 2.0], [3.0, 4.0]]))

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

#basetest
@ops.RegisterGradient("ZeroOut")
def _zero_out_grad(op, grad):
    """The gradients for `zero_out`.
    Args:
      op: The `zero_out` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
      grad: Gradient with respect to the output of the `zero_out` op.
    Returns:
      Gradients with respect to the input of `zero_out`.
    """
    to_zero = op.inputs[0]
    shape = array_ops.shape(to_zero)
    index = array_ops.zeros_like(shape)
    first_grad = array_ops.reshape(grad, [-1])[0]
    to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)
    return [to_zero_grad]  # List of one Tensor, since we have one input

x = tf.constant([5.0,4.0,2.0,3.0])
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.pow(x,2)
    z =zero_out_module_cpu.zero_out(y)
    z=tf.reduce_sum(z)

print(z)
# Derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x)
print(dz_dx)


