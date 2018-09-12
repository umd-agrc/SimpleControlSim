/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */


#ifndef NN_CONTROL_UTILS_H_
#define NN_CONTROL_UTILS_H_

#include <string>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <mxnet-cpp/MxNetCpp.h>

#include "defines.h"

inline bool isFileExists(const std::string &filename) {
  std::ifstream fhandle(filename.c_str());
  return fhandle.good();
}

inline bool check_datafiles(const std::vector<std::string> &data_files) {
  for (size_t index=0; index < data_files.size(); index++) {
    if (!(isFileExists(data_files[index]))) {
      LG << "Error: File does not exist: "<< data_files[index];
      return false;
    }
  }
  return true;
  }

inline bool setDataIter(mxnet::cpp::MXDataIter *iter , std::string useType,
              const std::vector<std::string> &data_files, int batch_size) {
    if (!check_datafiles(data_files))
        return false;

    iter->SetParam("batch_size", batch_size);
    iter->SetParam("shuffle", 1);
    iter->SetParam("flat", 1);

    if (useType ==  "Train") {
      iter->SetParam("image", data_files[0]);
      iter->SetParam("label", data_files[1]);
    } else if (useType == "Label") {
      iter->SetParam("image", data_files[2]);
      iter->SetParam("label", data_files[3]);
    }

    iter->CreateDataIter();
    return true;
}

inline mxnet::cpp::Symbol mlp(const std::string prefix, const std::vector<int> &layers) {
	auto x = mxnet::cpp::Symbol::Variable(prefix + "x");
	auto label = mxnet::cpp::Symbol::Variable(prefix + "y");
	
	std::vector<mxnet::cpp::Symbol> weights(layers.size());
	std::vector<mxnet::cpp::Symbol> biases(layers.size());
	std::vector<mxnet::cpp::Symbol> outputs(layers.size()-1);
  mxnet::cpp::Symbol fc;

	for(size_t i=0; i < layers.size(); i++) {
		weights[i] = mxnet::cpp::Symbol::Variable(prefix + "w" + std::to_string(i));
		biases[i] = mxnet::cpp::Symbol::Variable(prefix + "b" + std::to_string(i));
		fc = mxnet::cpp::FullyConnected(
			i == 0? x : outputs[i-1], // data
			weights[i],	
			biases[i],	
			layers[i]);
    if (i+1 < layers.size())
      outputs[i] = mxnet::cpp::Activation(prefix + "a" + std::to_string(i),
          fc, mxnet::cpp::ActivationActType::kSigmoid);
	}
	return fc;
}

inline mxnet::cpp::Symbol max(mxnet::cpp::Symbol lhs, mxnet::cpp::Symbol rhs) {
  return mxnet::cpp::broadcast_greater(lhs,rhs)*lhs
    + mxnet::cpp::broadcast_greater(rhs,lhs)*rhs;
}

inline mxnet::cpp::Symbol min(mxnet::cpp::Symbol lhs, mxnet::cpp::Symbol rhs) {
  return mxnet::cpp::broadcast_lesser(lhs,rhs)*lhs
    + mxnet::cpp::broadcast_lesser(rhs,lhs)*rhs;
}

void setRandomDistributionParams(
    std::normal_distribution<> &d, std::mt19937 &g);

mxnet::cpp::NDArray randNormal(double mean, double std, mxnet::cpp::Shape shape);

inline mx_float clip(mx_float a, mx_float low, mx_float high) {
  return fmax(fmin(a,high),low);
}

inline void copyNDArrayMap(std::map<std::string, mxnet::cpp::NDArray> &dst,
                    const std::map<std::string, mxnet::cpp::NDArray> &src) {
  static auto ctx = mxnet::cpp::Context::cpu();
  for (auto it = src.begin(); it != src.end(); it++) {
    dst[it->first] = it->second.Copy(ctx);
  }
}

inline mxnet::cpp::NDArray square(mxnet::cpp::NDArray x) {
  mxnet::cpp::NDArray ret;
  mxnet::cpp::Operator("square")(x).Invoke(ret);
  return ret;
}

inline mxnet::cpp::NDArray sum(mxnet::cpp::NDArray x, size_t axis = 0) {
  mxnet::cpp::NDArray ret;
  mxnet::cpp::Operator("sum")(x,axis).Invoke(ret);
  return ret;
}

inline mxnet::cpp::NDArray negative(mxnet::cpp::NDArray x) {
  mxnet::cpp::NDArray ret;
  mxnet::cpp::Operator("negative")(x).Invoke(ret);
  return ret;
}

inline mxnet::cpp::NDArray log(mxnet::cpp::NDArray x) {
  mxnet::cpp::NDArray ret;
  mxnet::cpp::Operator("log")(x).Invoke(ret);
  return ret;
}

inline mxnet::cpp::NDArray exp(mxnet::cpp::NDArray x) {
  mxnet::cpp::NDArray ret;
  mxnet::cpp::Operator("exp")(x).Invoke(ret);
  return ret;
}

inline mxnet::cpp::NDArray mean(mxnet::cpp::NDArray x) {
  mxnet::cpp::NDArray ret;
  mxnet::cpp::Operator("mean")(x).Invoke(ret);
  return ret;
}

inline mxnet::cpp::NDArray dot(mxnet::cpp::NDArray x1, mxnet::cpp::NDArray x2) {
  mxnet::cpp::NDArray ret;
  mxnet::cpp::Operator("dot")(x1,x2).Invoke(ret);
  return ret;
}

inline mxnet::cpp::NDArray max(mxnet::cpp::NDArray x) {
  mxnet::cpp::NDArray ret;
  mxnet::cpp::Operator("max")(x).Invoke(ret);
  return ret;
}

inline mxnet::cpp::NDArray Concat(
    const std::vector<mxnet::cpp::NDArray> &arrayVec,
    mxnet::cpp::Shape s) {
  mxnet::cpp::NDArray ret;
  std::vector<mx_float> v(s.Size());
  size_t idx = 0;
  for (auto arr : arrayVec) {
    memcpy(&v[idx], arr.GetData(), arr.Size()*sizeof(mx_float));
    idx += arr.Size();
  }

  ret = mxnet::cpp::NDArray(v,s,mxnet::cpp::Context::cpu());

  return ret;
}

inline mxnet::cpp::NDArray Concat(const mxnet::cpp::NDArray &a1,
                                  const mxnet::cpp::NDArray &a2,
                                  mxnet::cpp::Shape s) {
  mxnet::cpp::NDArray ret;

  std::vector<mx_float> v(s.Size());
  memcpy(&v[0], a1.GetData(), a1.Size()*sizeof(mx_float));
  memcpy(&v[a1.Size()], a2.GetData(), a2.Size()*sizeof(mx_float));
  ret = mxnet::cpp::NDArray(v,s,mxnet::cpp::Context::cpu());

  return ret;
}

inline mxnet::cpp::NDArray zeros(mxnet::cpp::Shape s) {
  mxnet::cpp::NDArray ret;
  std::vector<mx_float> v(s.Size(),0.0);
  ret = mxnet::cpp::NDArray(v,s,mxnet::cpp::Context::cpu());
  return ret;
}

inline mxnet::cpp::NDArray ones(mxnet::cpp::Shape s) {
  mxnet::cpp::NDArray ret;
  std::vector<mx_float> v(s.Size(),1.0);
  ret = mxnet::cpp::NDArray(v,s,mxnet::cpp::Context::cpu());
  return ret;
}

template<typename Function, typename... Args>
inline int testFunction(std::string testname, Function f, Args... args) {
  std::cout << "TEST: " << testname << std::endl << std::endl;

  f(args...);

  std::cout << std::endl << "------------------" << std::endl;
  std::cout << std::endl;

  mxnet::cpp::NDArray::WaitAll();
  return TEST_SUCCESS;
}

// Forward execution over rows of inArr
void vecExecForward(
    mxnet::cpp::Executor *exec, 
    mxnet::cpp::Symbol &sym,
    std::map<std::string,mxnet::cpp::NDArray> &inMap,
    std::string arrName,
    const std::vector<mxnet::cpp::NDArray> &inVec,
    std::vector<mxnet::cpp::NDArray> &outVec);

void vecExecForwardBackward(
    mxnet::cpp::Executor *exec, 
    mxnet::cpp::Symbol &sym,
    std::map<std::string,mxnet::cpp::NDArray> &inMap,
    std::vector<mxnet::cpp::NDArray> &labelVec,
    std::string arrName,
    const std::vector<mxnet::cpp::NDArray> &inVec,
    std::vector<mxnet::cpp::NDArray> &outVec);

#endif // NN_CONTROL_UTILS_H_
