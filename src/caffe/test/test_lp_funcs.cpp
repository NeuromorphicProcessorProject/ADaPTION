#include <stdint.h>  // for uint32_t & uint64_t
#include <time.h>
#include <cmath>  // for std::fabs

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class LPMathFunctionsTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LPMathFunctionsTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
  }

  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    this->blob_bottom_->Reshape(11, 17, 19, 23);
    this->blob_top_->Reshape(11, 17, 19, 23);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_top_);
  }

  virtual ~LPMathFunctionsTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
};

template <typename Dtype>
class CPULPMathFunctionsTest
  : public LPMathFunctionsTest<CPUDevice<Dtype> > {
};

TYPED_TEST_CASE(CPULPMathFunctionsTest, TestDtypes);

TYPED_TEST(CPULPMathFunctionsTest, TestNothing) {
  // The first test case of a test suite takes the longest time
  //   due to the set up overhead.
}

TYPED_TEST(CPULPMathFunctionsTest, TestBasicRounding) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  // round determinstic
  caffe_cpu_round_fp<TypeParam>(n, 3, 2, 0, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* rounded = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(rounded[i], round(x[i]*(2<<1)) / (2<<1));
  }
}
TYPED_TEST(CPULPMathFunctionsTest, TestStochasticRounding) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  // round stochastic 
  caffe_cpu_round_fp<TypeParam>(n, 3, 2, 1, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* rounded_1 = this->blob_bottom_->cpu_diff();
  caffe_cpu_round_fp<TypeParam>(n, 3, 2, 1, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* rounded_2 = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(rounded_1[i], round(x[i]*(2<<1)) / (2<<1), 1e0);
    EXPECT_NEAR(rounded_2[i], round(x[i]*(2<<1)) / (2<<1), 1e0);
  }
}

// ALSO SHOULD TEST CLIPPING


#ifndef CPU_ONLY

// Nothing here yet...

#endif


}  // namespace caffe
