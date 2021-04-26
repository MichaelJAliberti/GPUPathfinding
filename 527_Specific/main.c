/****************************************************************************


   gcc -O1 -std=gnu11 test_SOR_Task4.c -lpthread -lrt -lm -o test_SOR_4

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "LoadMap.h"
#include <pthread.h>

#define CPNS 3.0    /* Cycles per nanosecond -- Adjust to your computer, for example a 3.2 GhZ GPU, this would be 3.2 */



/* A, B, and C needs to be a multiple of your BLOCK_SIZE,
   total array size will be (GHOST + Ax^2 + Bx + C) */

#define BLOCK_SIZE 30
#define NUM_THREADS 60     // TO BE DETERMINED

#define OPTIONS 3

#define MINVAL   0.0
#define MAXVAL  10.0

#define TOL 0.00001
#define OMEGA 1.9       // TO BE DETERMINED

long int x;

typedef double data_t;

typedef struct {
  long int rowlen;
  data_t *data;
} arr_rec, *arr_ptr;

struct Node {
    int x;
    int y;
    struct Node* next;
};

/* Prototypes */
void diffuse(arr_ptr dataGrid, arr_ptr obstacleGrid);
bool checkDiffusion(arr_ptr dataGrid, arr_ptr obstacleGrid);
void traversePath(arr_ptr dataGrid, int srcX, int srcY);

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
/*
  As described in the clock_gettime manpage (type "man clock_gettime" at the
  shell prompt), a "timespec" is a structure that looks like this:

        struct timespec {
          time_t   tv_sec;   // seconds
          long     tv_nsec;  // and nanoseconds
        };
 */

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}
/*
     This method does not require adjusting a #define constant

  How to use this method:

      struct timespec time_start, time_stop;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
      measurement = interval(time_start, time_stop);

 */


/* -=-=-=-=- End of time measurement declarations =-=-=-=- */

/*****************************************************************************/
int main(int argc, char *argv[])
{
    struct timespec time_start, time_stop;
    double time_stamp[OPTIONS][NUM_TESTS];
    int convergence[OPTIONS][NUM_TESTS];

    long int n;
    long int alloc_size;

    x = NUM_TESTS-1;
    alloc_size = GHOST + A*x*x + B*x + C;

    printf("SOR serial variations \n");

    printf("OMEGA = %0.2f\n", OMEGA);

    /* declare and initialize the array */
    arr_ptr v0 = new_array(alloc_size);

    printf("  iter %d rowlen = %d\n", x, GHOST+n);
    init_array_rand(v0, GHOST+n);
    set_arr_rowlen(v0, GHOST+n);
    clock_gettime(CLOCK_REALTIME, &time_start);
    SOR(v0);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);


} /* end main */

/************************************/

/* SOR */
void diffuse(arr_ptr dataGrid, arr_ptr obstacleGrid)
{

  long int i, j;
  long int rowlen = get_arr_rowlen(dataGrid);
  data_t *data = get_array_start(dataGrid);
  data_t *obstacles = get_array_start(obstacleGrid);
  double newD, total_change = 1.0e10;   /* start w/ something big */
  int destX, destY;

  while (!checkDiffusion(dataGrid, obstacleGrid)) {
    data[destX*rowlen+destY] = rowlen;
    total_change = 0;
    for (i = 1; i < rowlen-1; i++) {
      for (j = 1; j < rowlen-1; j++) {
        newD = .25 * (data[(i-1)*rowlen+j] + data[(i+1)*rowlen+j] + data[i*rowlen+j+1] + data[i*rowlen+j-1]) * obstacles[destX*rowlen+destY];
        data[i*rowlen+j] = newD;
      }
    }
  }
}

bool checkDiffusion(arr_ptr dataGrid, arr_ptr obstacleGrid){
    int i,j;
    long int rowlen = get_arr_rowlen(dataGrid);
    data_t *data = get_array_start(dataGrid);
    data_t *obs = get_array_start(obstacleGrid);

    for(i = 0; i < rowlen; i++){
        for(j = 0; j < rowlen; j++){
            if(obs[i*rowlen+j] || (data[i*rowlen+j] <= 0)){return false;}
        }
    }
    return true;
}

void traversePath(arr_ptr dataGrid, int srcX, int srcY){
    long int rowlen = get_arr_rowlen(dataGrid);
    int i, x = srcX, y = srcY, arrsize = rowlen*rowlen;
    data_t *data = get_array_start(dataGrid);
    float currentSpot;
    float neighborMax;
    int pathLength = 0;

    currentSpot = data[x*rowlen+y]
    while(currentSpot != arrsize){
        neighborMax = 0;
        if(data[(x-1)*rowlen+y] > neighborMax){x--];}
        if(data[(x+1)*rowlen+y] > neighborMax){x++];}
        if(data[x*rowlen+y+1] > neighborMax){neighborMax=data[y++];}
        if(data[x*rowlen+y-1] > neighborMax){neighborMax=data[y--];}

        currentSpot = data[x*rowlen+y];
        outfile << "- [" << (x)%rowlen << ", " << (y)/rowlen << "] | D = " << currentSpot << endl;
        pathLength++;



    }
    outfile << "Path Length: " << pathLength << endl;
}
