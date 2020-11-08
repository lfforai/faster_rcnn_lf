一、github 代理设置 
git clone  --recursive

//自动配置，将网络设置为手动
export http_proxy=http://127.0.0.1:12333    
export https_proxy=https://127.0.0.1:12333
export HTTP_PROXY=http://127.0.0.1:12333
export HTTPS_PROXY=https://127.0.0.1:12333
export all_proxy=socks://127.0.0.1:1080
export ALL_PROXY=socks://127.0.0.1:1080

人工配置，才能起到加速作用//不要使用 用上面的
git config --global https.proxy https://127.0.0.1:12333
git config --global http.proxy http://127.0.0.1:12333

git config --global --unset http.proxy
git config --global --unset https.proxy

git config --global http.proxy 'socks5://127.0.0.1:1080'
git config --global https.proxy 'socks5://127.0.0.1:1080'

env|grep -i proxy 

二、conda 代理设置
gedit ~/.condarc
proxy_servers:
  http:http://127.0.0.1:12333
  https:https://127.0.0.1:12333
ssl_verify: false

0.15  cuda11.0
conda install -c rapidsai -c nvidia -c conda-forge \
    -c defaults rapids=0.15 python=3.7 cudatoolkit=11.0

0.14  cuda10.0
conda install -c rapidsai -c nvidia -c conda-forge \
    -c defaults cudf=0.14 python=3.7 cudatoolkit=10.0

pip 设置代理
--proxy="127.0.0.1:12333" 

三、编译opencv
sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
https://github.com/TheKobiThirdParty/benchmark

1、在opencv之前需要预装一些依赖库，其中有一项为libjasper-dev 。
大部分会出现这种错误
errorE: unable to locate libjasper-dev
解决方法：
sudo add-apt-repository "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ xenial main multiverse restricted universe"
sudo apt update
sudo apt install libjasper1 libjasper-dev
如果出现
apt-get install software-properties-common
则
apt-get install software-properties-common

//四、编译tensorflow 自定义so文件
编译中使用了预先安装在anaconda中的tensorflow的源文件，所以只要使用相同的gcc编译出的自定义op一定可以运行
#在以下文件中创建一个软连接（官网中没有，如果不创建编译会出现问题）
cd /root/anaconda3/envs/pyp3/lib/python3.7/site-packages/tensorflow/include/third_party/
mkdir gpus
cd gpus
mkdir cuda
cd cuda
mkdir include 
cd include
ln -s /usr/local/cuda/include/* ./

#编译前的环境变量设置
可以在/etc/profile 或在 ./bashac
export TF_CFLAGS=-I/root/anaconda3/envs/pyp3/lib/python3.7/site-packages/tensorflow/include
export TF_LFLAGS=-L/root/anaconda3/envs/pyp3/lib/python3.7/site-packages/tensorflow

直接运行
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

#编译cuda版本
nvcc -D_MWAITXINTRIN_H_INCLUDED -std=c++11 -c -o kernel_example.cu.o kernel_example.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

#nvcc -D_MWAITXINTRIN_H_INCLUDED -std=c++11 -c -o kernel_example.cu.o kernel_example.cu.cc ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC  --expt-relaxed-constexpr -l:libtensorflow_framework.so.2

g++ -std=c++11 -shared -o kernel_example.so kernel_example.cc kernel_example.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -l:libtensorflow_framework.so.2

#gcc>5时候 -D_GLIBCXX_USE_CXX11_ABI=0,这个不要加上去,最好编译前删除中间文件.o重新生成

#编译非cuda版本
g++ -std=c++11 -shared zero_out.cc -o zero_out.so -l:libtensorflow_framework.so.2  -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
#-D_GLIBCXX_USE_CXX11_ABI=0 事实证明不要添加这个玩意 

//五、编译tensorflow
bazel build --config=cuda --config=noaws --config=nogcp  //tensorflow/tools/pip_package:build_pip_package
#编译源文件，编译时候一定使用github代理国外服务器代理，不然无法load
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg  
#安装wlk包







