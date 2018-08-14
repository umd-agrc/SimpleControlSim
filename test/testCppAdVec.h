#pragma once

#include <CppAD/cppad/cppad.hpp>

namespace {

template<class Type>
CppAD::vector<Type> operator-(const CppAD::vector<Type> &v) {
  size_t k=v.size();
  CppAD::vector<Type> ret(k);
  for(size_t i=0; i < k; i++) {
    ret[i] = -v[i];
  }
  return ret;
}

} // anonymous namespace
