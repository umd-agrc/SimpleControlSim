#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_odeiv2.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/types.h>
#include <deque>

#include "genann.h"
#include "dynamics.h"
#include "diff.h"
#include "data.h"
#include "simulator.h"

//TODO turn nn stuff into a thread
int main(int argc, char *argv[]) {
  printf("NN controller trainer\n");
  printf("Train neural network to function as PID controller\n");

  // Start simulator environment
  std::deque<char *> sendQueue;
  pthread_t simulatorThreadId;
  pthread_create(&simulatorThreadId, NULL, simulate, static_cast<void*>(&sendQueue));

  // Start simulator server
  int listenfd = 0, connfd = 0;
  struct sockaddr_in serv_addr;

  char sendBuff[1025];

  listenfd = socket(AF_INET,SOCK_STREAM,0);
  SIM_INFO("Socket retrieve success\n");  

  memset(&serv_addr, '0', sizeof(serv_addr));
  memset(sendBuff, '0', sizeof(sendBuff));

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  serv_addr.sin_port = htons(5000);

  bind(listenfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));

  if (listen(listenfd, 10) == -1) {
    SIM_ERROR("Failed to listen");
    return SIM_FAILURE;
  }

  bool shouldExit = false;
  connfd = accept(listenfd, (struct sockaddr*)NULL, NULL); // Accept awaiting response   
  while(!shouldExit) {
    if (!sendQueue.empty()) {
      strcpy(sendBuff, sendQueue.front());
      sendQueue.pop_front();
      write(connfd, sendBuff, strlen(sendBuff));
    }
    if (!strcmp(sendBuff,SHUTDOWN_MESSAGE)) {
      shouldExit = true;
    }
    usleep(5000);
  }
  close(connfd);

  pthread_join(simulatorThreadId, NULL);
  return 0;
}
