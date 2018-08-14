#include "probability_distributions.h"

DiagGaussianPd::DiagGaussianPd() {
}
  
DiagGaussianPd::DiagGaussianPd(
    int len, std::vector<double> *mean, std::vector<double> *logstd) :
    len_(len), mean_(*mean), logstd_(*logstd) {
  seed = std::chrono::system_clock::now().time_since_epoch().count();
  randomGenerator = std::default_random_engine(seed);
  sampleDistribution = std::normal_distribution<double>(0.0,1.0);
  //TODO std_
}

std::vector<double> DiagGaussianPd::neglogp(std::vector<DataPoint> *actionSet) {
  std::vector<double> ret = DataPoint_mean_square(actionSet,&mean_),
                      tmp;
  ret = vector_edivide(&ret,&std_);
  ret = vector_scale(&ret,0.5);
  tmp = vector_create(0.5*log(M_2_PI),ret.size());
  ret = vector_add(&ret,&tmp);
  return vector_add(&ret,&logstd_);
}

std::vector<double> DiagGaussianPd::logp(std::vector<DataPoint> *actionSet) {
  std::vector<double> ret = neglogp(actionSet);
  return vector_scale(&ret,-1);
}

std::vector<double> DiagGaussianPd::entropy() {
  std::vector<double> ret = vector_add(&logstd_,0.5*log(M_2_PI*M_E));
  return vector_reduce_sum(&ret);
}

std::vector<double> DiagGaussianPd::sample() {
  std::vector<double> ret;
  for (int i=0; i < len_; i++) {
    ret.push_back(sampleDistribution(randomGenerator));
  }
  return ret;
}

//TODO KL Divergence penalty
