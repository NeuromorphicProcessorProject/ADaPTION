# ADaPTION: A reduced precision trainings interface for Nullhop
Deep neural networks (DNN) and Convolutional Networks (CNNs) offer a great opportunity for lots of classification and recognition tasks in machine learning, such as handwritten digit recognition (based on MNIST dataset) or image classification (based on Imagenet dataset). The weight between two consecutive neurons within the deep network is stored with high-resolution, i.e. 32 bit floating point are used to represent the weights. Furthermore, while training such a network the weight needs to be updated based on a gradient descent technique, such as error backpropagation. The weight update is then stored and propagated through the network. However, storage capacity and memory access are two limiting factors when it comes to an implementation of deep networks on small devices, since the storage capacity is limited and each memory access consumes power.\\\\
We extended the well-known deep learning library Caffe to support training deep CNNs with reduced numerical precision of weights, as well as activations using fixed-point notation.
We are able to quantize VGG16 [Zisserman 2014] down to 16 bit weights and activations with only 0.8 \% drop in top-1 accuracy compared to its high precision counterpart. The quantization especially of the activations lead to increase of up to 50 \% of sparsity in certain layers, which can be exploited to skip multiplications with zero, thus performing fast and computationally cheap inference.
## Low Precision Caffe. Used to convert existing high-precision CNNs to reduced precision of weights & activations
ADaPTION takes high precision pre-trained networks and adapts the parameter to a user specified fixed-point representation. One can also generate new low precision CNN architectures from scratch.
This toolbox is discribed in detail here:
* [ADaPTION](ArXiv)
* [Nullhop, a CNN hardware accelerator](ArXiv)

Notes for installation:
  * Adjust your paths!  You probably already have a copy of Caffe.  Make sure to use this one: check your .bash_profile or .bashrc file to make sure you are using the correct version.
  * Ubuntu 16.04: Needs a few fixes.  CuDNN and cmake seem to be incompatible right now, so you'll need to do Makefile.config fixes.  Basically gcc and nvcc are currently incompatible, but it's fixable.  Also, the location of the hdf5 header files has changed.  Google around if you get issues.  Overall, this branch should install the same as Caffe.  Here's a few important lines from my Makefile.config:

  ```
  INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial /home/dneil/Dropbox/tools/cudnn/7.5/cuda/include/
  LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial /home/dneil/Dropbox/tools/cudnn/7.5/cuda/lib64

  # .....

  # Somewhere at the end
  COMMON_FLAGS += -D_FORCE_INLINES
  ```

## Quick at-a-glance features:
  * New low-precision layer types.  Currently, we have the following:
    * **LPInnerProduct** - low precision inner product layer.  Rounds weights and optionally biases.  Available for CPU and GPU.
    * **LPConvolution** - low precision convolution layer.  Rounds weights and optionally biases.  Available for CPU, GPU, and CuDNN.
    * **LPAct** - low precision on activations.  Available for CPU and GPU.
  * Check out [examples/low_precision](examples/low_precision) for an example network with all of these layers.  There's also a [reduced-precision iPython notebook](examples/low_precision/Examine_LP_Net.ipynb) that you can preview on Github directly.
  * There is a new parameter type added: lpfp_param (low-precision fixed-point).  It has three elements: **bd** for how many bits before the decimal, **ad** for how many bits after the decimal, and **round_bias** if the bias should also be rounded.  One can also specify if the rounding should be stochastic or deterministic using the **rounding_scheme** flag. Here's a prototxt example:

```protobuf
layer {
  name: "ip1"
  type: "LPInnerProduct"
  bottom: "data"
  top: "ip1"
  lpfp_param {
    bd: 2
    ad: 2
    round_bias: false
    rounding_scheme: STOCHASTIC
  }
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "gaussian"
      std: 0.1
    }
    bias_filler {
      type: "constant"
    }
  }
}
```
This gives a Q2.2 weight representation.  Don't forget, if you're playing with serious quantization like this, your initial conditions matter.  **Make sure they don't all round to zero**!

## How does it work?

Basically, we wrote two new functions, which just do the standard fixed-point quantization: data = clip(round(data*2^f)/2^f, minval, maxval), where we have _f_ bits of representation after the decimal point.  Basically, think of it as a shift, truncating the number, and shifting back.

Here's the implementation of the rounding function, for CPU:
```cpp
template <>
void caffe_cpu_round_fp<float>(const int N, const int bd, const int ad, const int rounding_scheme,
                        const float *w, float *wr){
  const int bdshift = bd - 1;
  const int adshift = ad;
  const int rounding_mode = rounding_scheme;
  const float MAXVAL = ((float) (1 << bdshift)) - 1.0/(1<<adshift);
  switch (rounding_mode){
    case LowPrecisionFPParameter_RoundingScheme_DETERMINISTIC:
      for (int i = 0; i < N; ++i) {
        wr[i] = std::max(-MAXVAL, std::min( ((float)round(w[i]*
          (1<<adshift)))/(1<<adshift), MAXVAL));
      }
      break;
    case LowPrecisionFPParameter_RoundingScheme_STOCHASTIC:
      for (int i = 0; i < N; ++i) {
        wr[i] = std::max(-MAXVAL, std::min( ((float)floor(w[i]*
          (1<<adshift) + randomNumber()))/(1<<adshift), MAXVAL));
      }
      break;
  }
}
```

For GPU:
```cpp
template <typename Dtype>
__global__ void round_fp_kernel(const int N, const int bd, const int ad, const int rounding_scheme,
                        const Dtype *w, Dtype *wr){
  const int bdshift = bd - 1;
  const int adshift = ad;
  const float MAXVAL = ((float) (1 << bdshift)) - 1.0/(1<<adshift);
  switch (rounding_scheme){
      case LowPrecisionFPParameter_RoundingScheme_DETERMINISTIC:
          CUDA_KERNEL_LOOP(index, N) {
          wr[index] = max(-MAXVAL, min( ((Dtype)round(w[index]*(1<<adshift)))/(1<<adshift), MAXVAL));
          }
      case LowPrecisionFPParameter_RoundingScheme_STOCHASTIC:
          CUDA_KERNEL_LOOP(index, N) {
          wr[index] = max(-MAXVAL, min( ((Dtype)floorf(w[index]*(1<<adshift)+RandUniform_device(index)))/(1<<adshift), MAXVAL));
          }
  }
}

template <>
void caffe_gpu_round_fp(const int N, const int bd, const int ad,
                        const float *w, float *wr){
  round_fp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, bd, ad, w, wr);
}
```
And we call these functions to round Caffe blobs (units of data).  So, for example, in the inner product layer:

```cpp
  // Round weights:
  const int weight_count = this->blobs_[0]->count();
  caffe_cpu_round_fp(weight_count, BD_, AD_, this->blobs_[0]->cpu_data(),
    this->blobs_[0+1]->mutable_cpu_data());
  const Dtype* weight = this->blobs_[0+1]->cpu_data();
```
We fetch the weights (in blobs[0]), round them and store them (in blobs[1]), and then we fetch a handle to that data as the "weights" used in the computation.  In the backward phase, we just use the full precision ones and pretend like nothing happened.

## Extending

Mostly, it's a cut+paste job to add new layers.  Copy them, replace their names, and double their storage capacity - we now need two copies of all data, one for regular precision and one for low precision.  Then just call the appropriate rounding function (cpu or gpu) on the data you want rounded (weights, biases, activations, etc.).

Make sure to index the correct variables now!  The bias is likely to be be set to blobs_[1], which should now be a copy.  By convention, all the even-number blobs correspond to the unrounded version, while all the odd-numbered blobs correspond to the rounded version.

## Quantization

A an example notebook how to convert a high-precision model into a low-precision one at any desired fixed-point bit precision can be found in
[quantization](examples/low_precision/quantization/)

# Caffe

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
