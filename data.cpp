#include "data.h"

int seekChar(const char *s, int c) {
  char *inst = strchr((char *)s,c);
  if (inst != NULL) {
    return inst - s; 
  } else return SIM_FAILURE;
}

int seekCharSequence(const char *s, const char *c, int *idx) {
  char *substr = (char *)s;
  char *currChar = (char *)c;

  int substrIdx = 0;
  int currIdx = 0;
  while (substr != NULL && *currChar != 0) {
    substrIdx = seekChar(substr,*currChar); 
    if (*idx == SIM_FAILURE) return SIM_FAILURE;

    substr += substrIdx;
    currIdx += substrIdx;
    *idx = currIdx; 

    ++currChar;
    ++idx;
  }

  return SIM_SUCCESS;
}

int load_data(const char *training_data,
              int *datalen,
              double **input,
              double **output,
              int *samples) {
  /* Load the training data-set. */
  FILE *in = fopen(training_data, "r");
  if (!in) {
    printf("Could not open file: %s\n", training_data);
    exit(1);
  }

  /* Loop through the data to get a count. */
  char line[1024];

  // Read first line
  int numInputs = 0;
  int numOutputs = 0;
  char dataSetWrapperChars[5];
  int dataSetIdx[4];
  strcpy(dataSetWrapperChars,"[][]");

  // Determine size of input and output sets
  if (!feof(in) && fgets(line,1024,in)) {
    seekCharSequence(line,dataSetWrapperChars,dataSetIdx); 
    int currIdx = dataSetIdx[0];
    while (currIdx < dataSetIdx[1]) {
      currIdx += seekChar(&(line[currIdx]),' ') + 1;
      ++numInputs;
    }
    printf("num inputs: %d\n", numInputs);

    currIdx = dataSetIdx[2];
    int idxAdd = 0;
    while (currIdx < dataSetIdx[3]) {
      idxAdd = seekChar(&(line[currIdx]),' ');
      if (idxAdd == SIM_FAILURE) break;
      currIdx += idxAdd + 1;
      ++numOutputs;
    }
    if (currIdx != dataSetIdx[2]) ++numOutputs;
    printf("num outputs: %d\n", numOutputs);
  }

  while (!feof(in) && fgets(line, 1024, in)) {
    ++(*samples);
  }
  
  fseek(in, 0, SEEK_SET);

  printf("Loading %d data points from %s\n", *samples, training_data);

  /* Allocate memory for input and output data. */
  *input = (double*)malloc(sizeof(double) * (*samples) * numInputs);
  *output = (double*)malloc(sizeof(double) * (*samples) * numOutputs);

  // Read in data
  char inputSubstr[1024], outputSubstr[1024];
  char *split;
  if (!feof(in) && fgets(line,1024,in)) {
    for (int i=0; i < (*samples) && fgets(line,1024,in); ++i) {
      double *p = *input + i * numInputs;
      double *o = *output + i * numOutputs;
      seekCharSequence(line,dataSetWrapperChars,dataSetIdx); 
      int inputStrlen = dataSetIdx[1]-dataSetIdx[0]-1;
      strncpy(inputSubstr,&(line[dataSetIdx[0]+1]),inputStrlen);
      inputSubstr[inputStrlen] = 0;
      split = strtok(inputSubstr, " ");
      for (int j = 0; j < numInputs; ++j) {
        p[j] = atof(split);
        split = strtok(0, " ");
      }

      int outputStrlen = dataSetIdx[3]-dataSetIdx[2]-1;
      strncpy(outputSubstr,&(line[dataSetIdx[2]+1]),outputStrlen);
      outputSubstr[outputStrlen] = 0;
      split = strtok(outputSubstr, " ");
      for (int j = 0; j < numOutputs; ++j) {
        o[j] = atof(split);
        split = strtok(0, " ");
      }
    }
  }

  *datalen = numInputs;
  *(datalen+1) = numOutputs;

  fclose(in);

  return SIM_SUCCESS;
}

int teardownData(double *input, double *output) {
  free(input);
  free(output);
  
  return SIM_SUCCESS;
}

int logVector(FILE *logfile,gsl_vector *v, char *dlm, char *mode, bool endDlm) {
  char line[1024], finalChar[2];
  line[0]=0;finalChar[0]=0;

  if (!strcmp(mode,APPEND_LINE)) {
  } else if (!strcmp(mode,NEW_LINE)) {
    strcat(finalChar,"\n");
  } else {
    // Unrecognized log mode
    return SIM_FAILURE;
  }

  int len = snprintf(NULL,0,"%f",gsl_vector_get(v,0));
  char *str = (char*)malloc(len+1);
  for (unsigned int i=0; i < v->size; i++) {
    double el = gsl_vector_get(v,i);
    snprintf(str,len,"%f",el);
    strcat(line,str);
    if (i < v->size - 1 || endDlm) {
      strcat(line,dlm);
    }
  }
  free(str);

  strcat(line,finalChar);

  fputs(line,logfile);

  return SIM_SUCCESS;
}

int logHeader(FILE *logfile,char *header) {
  fputs(header,logfile);

  return SIM_SUCCESS;
}

int logTime(FILE *logfile,double t, char *dlm, char *mode) {
  char  finalChar[2];
  finalChar[0]=0;

  if (!strcmp(mode,APPEND_LINE)) {
  } else if (!strcmp(mode,NEW_LINE)) {
    strcat(finalChar,"\n");
  } else {
    // Unrecognized log mode
    return SIM_FAILURE;
  }

  int len = snprintf(NULL,0,"%f",t);
  char *str = (char*)malloc(len+1);
  snprintf(str,len,"%f",t);
  fputs(str,logfile);
  fputs(dlm,logfile);
  fputs(finalChar,logfile);

  return SIM_SUCCESS;
}

void printDoubleArray(double *arr, int len, char *name) {
  printf("%s:\t",name);
  for (int i=0; i < len; i++) {
    printf("%g  ",*(arr+i)); 
  }
  printf("\n");
}
