#include "data.h"

int seekChar(const char *s, int c) {
  char *inst = strchr((char *)s,c);
  if (inst != NULL) {
    return inst - s; 
  } else return SIM_FAIL;
}

int seekCharSequence(const char *s, const char *c, int *idx) {
  char *substr = (char *)s;
  char *currChar = (char *)c;
int substrIdx = 0;
  int currIdx = 0;
  while (substr != NULL && *currChar != 0) {
    substrIdx = seekChar(substr,*currChar); 
    if (*idx == SIM_FAIL) return SIM_FAIL;

    substr += substrIdx;
    currIdx += substrIdx;
    *idx = currIdx; 

    ++currChar;
    ++idx;
  }

  return SIM_SUCCESS;
}

void printDoubleArray(double *arr, int len, char *name) {
  printf("%s:\t",name);
  for (int i=0; i < len; i++) {
    printf("%g  ",*(arr+i)); 
  }
  printf("\n");
}

void logNDArrayMap(std::string fileprefix,
                   std::string filesuffix,
                   std::map<std::string,mxnet::cpp::NDArray> m) {
  for (auto el : m) {
    std::ofstream outfile(fileprefix + el.first + filesuffix);
    outfile << el.first << std::endl;
    outfile << el.second;
    outfile.close();
  }
}
