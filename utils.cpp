#include "utils.h"

using namespace mxnet::cpp;

static Context ctx = Context::cpu();
static std::mt19937 gen;
static std::normal_distribution<> d{0,1};

void setupRandomDistribution() {
  std::random_device rd;
  gen = std::mt19937(rd());
}

NDArray randNormal(NDArray mean, NDArray std, int len) {
  std::vector<mx_float> vec(len);
  for (int i=0; i < len; i++) {
    d.param(std::normal_distribution<>::param_type(mean.At(0,i),std.At(0,i)));
    vec[i] = d(gen);
  }
  return NDArray(vec, Shape(1,len), ctx);
}
