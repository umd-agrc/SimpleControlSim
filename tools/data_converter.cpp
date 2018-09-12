#include <iostream>

#include <mxnet-cpp/MxNetCpp.h>

#include "../utils.h"

void convertNDArrayFile(std::string infilename,std::string outfilename) {
  // Skip if file doesn't exist
  if (!isFileExists(infilename)) {
    std::cout << "skipping file " << infilename << ". does not exist" << std::endl;
    return;
  }


}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << "<ndarrayfile1> <ndarrayfile2>...";
    return 1;
  }
  for (int i=0; i < argc; i++) {
    std::string infilename(argv[1]);
    std::string outfilename(infilename + "_.csv");
    convertNDArrayFile(infilename,outfilename);
  }
}
