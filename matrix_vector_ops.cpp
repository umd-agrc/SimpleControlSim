#include "matrix_vector_ops.h"

std::vector<double> vector_stack(const std::vector<double> *a, const std::vector<double> *b) {
  std::vector<double> c;
  for (auto it=a->begin(); it!=a->end(); it++) {
    c.push_back(*it);
  }

  for (auto it=b->begin(); it != b->end(); it++) {
    c.push_back(*it);
  }
  return c;
}

std::vector<double> vector_sub(const std::vector<double> *a, const std::vector<double> *b) {
  std::vector<double> c;
  auto itb = b->begin();
  auto ita = a->begin();
  for (; ita != a->end() && itb != b->end(); ita++) {
    c.push_back(*ita-*itb);
    itb++;
  }
  return c;
}


std::vector<double> vector_add(const std::vector<double> *a, const std::vector<double> *b) {
  std::vector<double> c;
  auto itb = b->begin();
  auto ita = a->begin();
  for (; ita != a->end() && itb != b->end(); ita++) {
    c.push_back(*ita+*itb);
    itb++;
  }
  return c;
}

std::vector<double> vector_add(const std::vector<double> *a, double b) {
  std::vector<double> c;
  auto ita = a->begin();
  for (; ita != a->end(); ita++) {
    c.push_back(*ita+b);
  }
  return c;
}

std::vector<double> vector_scale(const std::vector<double> *a, double k) {
  std::vector<double> ret;
  for (auto it=a->begin(); it!=a->end(); it++) {
    ret.push_back((*it)*k); 
  }

  return ret;
}

double vector_infnorm(const std::vector<double> *v) {
  double absmax = 0,absCurr;
  for (auto it=v->begin(); it != v->end(); it++) {
    absCurr = fabs(*it);
    if (absCurr > absmax) {
      absmax = absCurr;
    }
  }
  return absmax;
}

void vector_print(const std::vector<double> *v, char *name) {
  printf("%s: ",name);
  for (auto it=v->begin(); it!=v->end(); it++ ) {
    printf("%g ", *it);
  }
  printf("\n");
}

std::vector<double> vector_reduce_sum(const std::vector<double> *a) {
  std::vector<double> ret;
  auto ita = a->begin();
  ret.resize(1);
  ret[0] = 0;
  for (; ita != a->end(); ita++) {
    ret[0] += *ita;
  }
  return ret;
}

std::vector<double> DataPoint_reduce_sum(const std::vector<DataPoint> *a) {
  return std::vector<double>();
}

std::vector<double> DataPoint_mean_square(const std::vector<DataPoint> *a,
                                          std::vector<double> *mean) {
  return std::vector<double>();
}

std::vector<double> vector_create(double d,int size) {
  std::vector<double> ret;
  ret.resize(size);
  return ret;
}

std::vector<double> vector_edivide(const std::vector<double> *a,
                                   const std::vector<double> *b) {
  std::vector<double> c;
  auto itb = b->begin();
  auto ita = a->begin();
  for (; ita != a->end() && itb != b->end(); ita++) {
    c.push_back(*ita / *itb);
    itb++;
  }
  return c;
}

