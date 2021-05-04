/****************************************************************************

   gcc -O1 -lrt -lm main.c -o main
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "LoadMap.h"

/* structures */
struct point{
    int x;
    int y;
    int length;
    struct point* next;
};

/* Prototypes */
void diffuse(grid* g);
int checkDiffusion(grid* g);
struct point** traversePath(grid* g);
int printPath(int num_agents, struct point** paths, int args, char* filein);
double interval(struct timespec start, struct timespec end);
double wakeup_delay();

/*****************************************************************************/
int main(int argc, char *argv[])
{
    struct timespec time_start, time_stop;
    char* name;
    
    /* retrieve input file */
    grid* myGrid;
    if (argc == 3){
        name = argv[2];
        myGrid = LoadGrid(name); //this creates the grid to use
    }
    else if (argc == 2){
        printf("DEFAULTING\n");
        name = "null.yaml";
        myGrid = MakeGrid(atoi(argv[1]));
    }
    else{
        printf("ERROR: No input\n");
        exit(1);
    }

    /* setup for timing */
    //double final_answer = wakeup_delay();

    /* this runs the entire diffusion process and will run until the graph is fully diffused */
    clock_gettime(CLOCK_REALTIME, &time_start);
    diffuse(myGrid);
    struct point** paths = traversePath(myGrid);
    clock_gettime(CLOCK_REALTIME, &time_stop);

    /* print runtime */
    printf("TIME: %.10f\n", interval(time_start, time_stop));
    //printf("\n%.0f\n", final_answer);

    /* print path to file */
    printPath(myGrid->num_agents, paths, argc, name);
} /* end main */

/************************************/

/* Diffuse Function */
void diffuse(grid* g)
{
  long int i, j;
  int rowlen = g->size;
  int row_bound = rowlen + 1;
  int mod_size = rowlen + 2;
  data_t *data = g->diff_matrix;
  data_t *obstacles = g->obs_matrix;
  data_t newD;
  int destX = g->dx, destY = g->dy;

  data_t large = (data_t) rowlen*rowlen*10000000000;
  int it = 0;
  int fullsize = rowlen * rowlen;

  while (!checkDiffusion(g) && it < fullsize){ /**/
    it++;
    PrintGrid(g);
    for (i = 1; i < row_bound; i++) {
      for (j = 1; j < row_bound; j++) {
        data[destY * mod_size + destX] = large;
        newD = (data_t) .25 * (data[(i-1)*mod_size + j] + data[(i+1)*mod_size + j] + data[i*mod_size + j+1] + data[i*mod_size + j-1]) * obstacles[i*mod_size + j];
        data[i*mod_size + j] = newD;
      }
    }
  }
}

int checkDiffusion(grid* g){
    int i,j;
    int debug1=1,debug2=2,debug3=3;
    int rowlen = g->size;
    int row_bound = rowlen + 1;
    int mod_size = rowlen + 2;
    data_t *data = g->diff_matrix;
    data_t *obstacles = g->obs_matrix;

    for(i = 1; i < row_bound; i++){
        for(j = 1; j < row_bound; j++){
            if(obstacles[i*mod_size+j] && (data[i*mod_size+j] == 0)){return 0;}
        }
    }

    return 1;
}


struct point** traversePath(grid* g){
    
    int rowlen = g->size;
    int row_bound = rowlen + 1;
    int mod_size = rowlen + 2;
    int num_agents = g->num_agents;
    data_t *data = g->diff_matrix;
    struct agent* a = g->agents;
    struct point* pt_it;
    int i;
    int x, y, arrsize = rowlen*rowlen;
    data_t currentSpot, target;
    data_t neighborMax;
    int neighborIter;
    int pathLength;
    target = g->dy*mod_size + g->dx;

    struct point** paths = (struct point**) calloc(num_agents, sizeof(struct point*));

    for (i = 0; i < num_agents; i++){

        x = a->sx;
        y = a->sy;
        pathLength = 0;
        paths[i] = (struct point*) calloc(1, sizeof(struct point));
        paths[i]->x = x;
        paths[i]->y = y;
        paths[i]->next = NULL;
        pt_it = paths[i];
        currentSpot = y*mod_size + x;

        printf("AGENT%d: [%d, %d]\n", i, x, y);

        while(currentSpot != target){

            pt_it->next = (struct point*) calloc(1, sizeof(struct point));

            neighborMax = 0;
            if(data[(y-1)*mod_size + x] > neighborMax){
                neighborMax = data[(y-1)*mod_size+x]; 
                neighborIter = 1;
            }
            if(data[(y+1)*mod_size + x] > neighborMax){
                neighborMax = data[(y+1)*mod_size+x]; 
                neighborIter = 2;
            }
            if(data[y*mod_size+x+1] > neighborMax){
                neighborMax = data[y*mod_size+x+1]; 
                neighborIter = 3;
            }
            if(data[y*mod_size+x-1] > neighborMax){
                neighborMax = data[y*mod_size+x-1]; 
                neighborIter = 4;
            }
            
            if(neighborIter == 1) y--;
            else if(neighborIter == 2) y++;
            else if(neighborIter == 3) x++;
            else if(neighborIter == 4) x--;

            // setup for next iteration
            currentSpot = y*mod_size+x;
            pt_it = pt_it->next;
            pt_it->x = x;
            pt_it->y = y;
            pt_it->next = NULL;

            pathLength++;
        }

        paths[i]->length = pathLength;
        a = a->next;
    }

    return paths;
}

int printPath(int num_agents, struct point** paths, int args, char* filein)
{
    int i;
    struct point* pt_it;
    FILE *fp;
    char* new_name;

    // get new file name
    if (args == 3){
        new_name = strtok(filein, ".");
        new_name = strcat(new_name, "_serial_path.yaml");
    }
    else
        new_name = "default_paths.yaml";

    fp = fopen(new_name,"w+");
    fprintf(fp, "paths:\n");

    for (i = 0; i < num_agents; i++){
        fprintf(fp, "-   name: agent%d\n", i);
        pt_it = paths[i];
        fprintf(fp, "    path:\n");
        while (pt_it != NULL){
            fprintf(fp, "    - [%d,%d]\n", pt_it->x -1, pt_it->y -1);
            pt_it = pt_it->next;
        }
        fprintf(fp, "    length: %d\n", paths[i]->length);
    }

    fclose(fp);
}

double wakeup_delay()
{
  double meas = 0; int i, j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
  j = 100;
  while (meas < 1.0) {
    for (i=1; i<j; i++) {
      /* This iterative calculation uses a chaotic map function, specifically
         the complex quadratic map (as in Julia and Mandelbrot sets), which is
         unpredictable enough to prevent compiler optimisation. */
      quasi_random = quasi_random*quasi_random - 1.923432;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    meas = interval(time_start, time_stop);
    j *= 2; /* Twice as much delay next time, until we've taken 1 second */
  }
  return quasi_random;
}

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