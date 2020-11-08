import tensorflow as tf
print(tf.sysconfig.get_include())
print(tf.sysconfig.get_lib())

#test how to build and use the self creating ops in tensorflow2.4 with ncc g++ ..crossing building tool
zero_out_module = tf.load_op_library('/root/eclipse-workspace_lf/ROI_tf_lf/kernel_example.so')
print(zero_out_module.example([[1, 2], [3, 4]]))

zero_out_module = tf.load_op_library('/root/eclipse-workspace_lf/ZeroOut/zero_out.so')
print(zero_out_module.zero_out([[1, 2], [3, 4]]))