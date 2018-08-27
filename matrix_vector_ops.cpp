#include "matrix_vector_ops.h"

using namespace mxnet::cpp;

std::vector<mx_float> vector_stack(const std::vector<mx_float> *a, const std::vector<mx_float> *b) {
  std::vector<mx_float> c;
  for (auto it=a->begin(); it!=a->end(); it++) {
    c.push_back(*it);
  }

  for (auto it=b->begin(); it != b->end(); it++) {
    c.push_back(*it);
  }
  return c;
}

std::vector<mx_float> vector_sub(const std::vector<mx_float> *a, const std::vector<mx_float> *b) {
  std::vector<mx_float> c;
  auto itb = b->begin();
  auto ita = a->begin();
  for (; ita != a->end() && itb != b->end(); ita++) {
    c.push_back(*ita-*itb);
    itb++;
  }
  return c;
}


std::vector<mx_float> vector_add(const std::vector<mx_float> *a, const std::vector<mx_float> *b) {
  std::vector<mx_float> c;
  auto itb = b->begin();
  auto ita = a->begin();
  for (; ita != a->end() && itb != b->end(); ita++) {
    c.push_back(*ita+*itb);
    itb++;
  }
  return c;
}

std::vector<mx_float> vector_add(const std::vector<mx_float> *a, mx_float b) {
  std::vector<mx_float> c;
  auto ita = a->begin();
  for (; ita != a->end(); ita++) {
    c.push_back(*ita+b);
  }
  return c;
}

std::vector<mx_float> vector_scale(const std::vector<mx_float> *a, mx_float k) {
  std::vector<mx_float> ret;
  for (auto it=a->begin(); it!=a->end(); it++) {
    ret.push_back((*it)*k); 
  }

  return ret;
}

mx_float vector_infnorm(const std::vector<mx_float> *v) {
  mx_float absmax = 0,absCurr;
  for (auto it=v->begin(); it != v->end(); it++) {
    absCurr = fabs(*it);
    if (absCurr > absmax) {
      absmax = absCurr;
    }
  }
  return absmax;
}

void vector_print(const std::vector<mx_float> *v, char *name) {
  printf("%s: ",name);
  for (auto it=v->begin(); it!=v->end(); it++ ) {
    printf("%g ", *it);
  }
  printf("\n");
}

std::vector<mx_float> vector_reduce_sum(const std::vector<mx_float> *a) {
  std::vector<mx_float> ret;
  auto ita = a->begin();
  ret.resize(1);
  ret[0] = 0;
  for (; ita != a->end(); ita++) {
    ret[0] += *ita;
  }
  return ret;
}

std::vector<mx_float> DataPoint_reduce_sum(const std::vector<DataPoint> *a) {
  return std::vector<mx_float>();
}

std::vector<mx_float> DataPoint_mean_square(const std::vector<DataPoint> *a,
                                          std::vector<mx_float> *mean) {
  return std::vector<mx_float>();
}

std::vector<mx_float> vector_create(mx_float d,int size) {
  std::vector<mx_float> ret;
  ret.resize(size);
  return ret;
}

std::vector<mx_float> vector_edivide(const std::vector<mx_float> *a,
                                   const std::vector<mx_float> *b) {
  std::vector<mx_float> c;
  auto itb = b->begin();
  auto ita = a->begin();
  for (; ita != a->end() && itb != b->end(); ita++) {
    c.push_back(*ita / *itb);
    itb++;
  }
  return c;
}

