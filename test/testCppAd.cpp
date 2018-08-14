#include <iostream>
#include <vector>
#include <cassert>
#include <CppAD/cppad/cppad.hpp>

#include "test/testCppAdVec.h"

namespace {

  typedef std::vector<double> vec_t;
  typedef std::vector<vec_t> tensor_t;
  typedef CppAD::vector<double> advec_t;

  struct Policy {
    double mult = 1.0;

    template <class Type>
    Type logp(const CppAD::vector<Type> &y) const {
      size_t k = y.size();
      Type ret = 0.;
      size_t i;
      for(i=0; i < k; i++) {
        ret += mult*y[i]*y[i];
      }
      return ret;
    }
  };

  template <class Type>
  Type fn(const vec_t &a, const Type &x) {
    size_t k = a.size();
    Type y = 0.;
    Type x_i = 1.;
    size_t i;
    for(i=0; i<k; i++) {
      y += a[i]*x_i;
      x_i *= x;
    }
    return y;
  }

  template <class Type>
  CppAD::vector<Type> fn2(const vec_t &a, const CppAD::vector<Type> &v) {
    assert(a.size() == v.size());
    size_t k = v.size();
    CppAD::vector<Type> ret(1);
    size_t i;
    for(i = 0; i < k; i++) {
      ret[0] += a[i]*v[i]*v[i];
    }
    return ret;
  }

  template <class Type>
  Type min(const Type &v1, const Type &v2) {
    return v1 < v2 ? v1 : v2;
  }

  template <class Type>
  Type max(const Type &v1, const Type &v2) {
    return v1 > v2 ? v1 : v2;
  }

  template <class Type>
  Type clip(const Type &val, const Type &lbound, const Type &rbound) {
    return max(min(val,rbound),lbound);
  }

  template <class Type>
  CppAD::vector<Type> clippedLoss(const CppAD::vector<Type> &y, const vec_t &t,
      const Policy &p, const Policy &oldP, const Type &atarg,
      const Type &epsilon) {
    //TODO Ratio returns a vector of ratios
    //     Thus, ratio*atarg is an element-wise vector multiplication
    CppAD::vector<Type> ret(1);
    Type ratio = exp(p.logp(y) - oldP.logp(y));
    Type surr1 = ratio*atarg;
    Type surr2 = clip(ratio,1-epsilon,1+epsilon)*atarg;
    Type polSurr = min(surr1,surr2);
    ret[0] = polSurr;
    return ret;
  }
}

int main(int argc, char **argv) {
  using CppAD::AD;
  using std::vector;

  size_t i;
  
  size_t k=3;
  vec_t a(k);
  for(i=0; i<k; i++)
    a[i] = 1.;

  Policy p,oldP;
  p.mult = 10;
  AD<double> atarg = 5;
  AD<double> epsilon = 0.2;

  size_t n=3;
  CppAD::vector<AD<double>>X(n);
  X[0] = -0.3;
  X[1] = -0.2;
  X[2] = -0.1;

  CppAD::vector<AD<double>>Q(n);
  Q = -X;
  std::cout << "testing negation operator: " << Q[0] << std::endl;

  double r = CppAD::Value(X[0]);
  std::cout << "converted from AD<double> " << r << std::endl;

  CppAD::Independent(X);

  size_t m=1;
  CppAD::vector<AD<double>>Y(m);
  Y = clippedLoss(X,a,p,oldP,atarg,epsilon);

  CppAD::ADFun<double> f(X,Y);

  vector<double> jac(m*n);
  vector<double> x(n);
  x[0] = -0.3;
  x[1] = -0.2;
  x[2] = -0.1;
  jac = f.Jacobian(x);

  std::cout << "f(-3,-2,-1) computed by CppAD =\t\t" << Y[0] << std::endl;
  std::cout << "df/dx1(-3,-2,-1) computed by CppAD =\t" << jac[0] << std::endl;
  std::cout << "df/dx2(-3,-2,-1) computed by CppAD =\t" << jac[1] << std::endl;
  std::cout << "df/dx3(-3,-2,-1) computed by CppAD =\t" << jac[2] << std::endl;

  int error_code;
  if(jac[0] == -7.)
    error_code = 0;
  else error_code = 1;

  return error_code;
}
