#include <algorithm>
#include <vector>

#include "caffe/layers/lp_act_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LPActLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  caffe_cpu_round_fp(count, this->layer_param_.lpfp_param().bd(), this->layer_param_.lpfp_param().ad(), bottom_data, top_data);
}

template <typename Dtype>
void LPActLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(LPActLayer);
#endif

INSTANTIATE_CLASS(LPActLayer);
REGISTER_LAYER_CLASS(LPAct);
}  // namespace caffe
