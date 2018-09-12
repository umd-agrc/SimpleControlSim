#include "testSim.h"

int main(int argc, char **argv) {
  testLog();
  testFunction("Debug build", testDebug);
  return 0;
}

int testLog() {
  for (int i=0; i < 3; i++) {
    log("hi my name ", i, "");
  }

  return TEST_SUCCESS;
}

void testDebug() {
#ifdef DEBUG
  log("DEBUG on");
#else
  log("DEBUG off");
#endif
}
