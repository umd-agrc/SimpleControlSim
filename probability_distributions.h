#pragma once

#include <cmath>
#include <chrono>
#include <random>
#include <string.h>

#include "defines.h"
#include "matrix_vector_ops.h"

class DiagGaussianPd {
public:
  int len_;
  std::vector<double> mean_, logstd_, std_;
  unsigned seed;
  std::default_random_engine randomGenerator;
  std::normal_distribution<double> sampleDistribution;

public:
  DiagGaussianPd();

  DiagGaussianPd(
      int len, std::vector<double> *mean, std::vector<double> *logstd);
  
  std::vector<double> neglogp(std::vector<DataPoint> *actionSet);

  std::vector<double> logp(std::vector<DataPoint> *actionSet);

  std::vector<double> entropy();

  std::vector<double> sample();
};


