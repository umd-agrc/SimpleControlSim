#include <stdio.h>
#include <iostream>
#include <FADBAD++/fadiff.h>

using namespace fadbad;

F<double> fn(const F<double>& x, const F<double>&y) {
  F<double> z=fabs(sqrt(x));
  return y*z+sin(z);
}

int main(int argc, char **argv) {

  F<double> x,y,f;

  x=1;
  x.diff(0,2);
  y=2;
  y.diff(1,2);
  f=fn(x,y);
  
  double fval=f.x();
  double dfdx=f.d(0);
  double dfdy=f.d(1);

  std::cout << "f(x,y)=" << fval << std::endl;
  std::cout << "df/dx(x,y)=" << dfdx << std::endl;
  std::cout << "df/dy(x,y)=" << dfdy << std::endl;

  return 0;
}
