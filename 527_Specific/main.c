<<<<<<< Updated upstream
/****************************************************************************

   gcc -O1 main.c -o main

*/

#include "LoadMap.h"

/* Prototypes */
void diffuse(grid* g);
int checkDiffusion(grid* g);
void traversePath(grid* g);
int getNeighborAvg(grid* g, int ax, int ay);
/*****************************************************************************/
int main(int argc, char *argv[])
{
    /* declare and initialize the array */
    grid* myGrid = LoadGrid("map_8by8_obst12_agents1_ex0.yaml"); //this creates the grid to use

    /*this runs the entire diffusion process and will run until the graph is fully diffused.*/
    diffuse(myGrid);
    PrintGrid(myGrid);
    traversePath(myGrid);

} /* end main */

/************************************/

/* Diffuse Function */
void diffuse(grid* g)
{

  long int i, j;
  int rowlen = g->size;
  data_t *data = g->diff_matrix;
  data_t *obstacles = g->obs_matrix;
  data_t newD;
  int destX = g->dx, destY = g->dy;
  int sum;

  int large = rowlen*rowlen*rowlen*rowlen*rowlen;

  while (!checkDiffusion(g)) {
    PrintGrid(g);
    printf("\n");
    for (i = 1; i < rowlen-1; i++) {
      for (j = 1; j < rowlen-1; j++) {
        data[destY*rowlen+destX] = large;
        //newD = .25 * (data[(i-1)*rowlen+j] + data[(i+1)*rowlen+j] + data[i*rowlen+j+1] + data[i*rowlen+j-1]) * obstacles[destX*rowlen+destY];
        sum = getNeighborAvg(g, i, j);
        newD = sum * obstacles[j*rowlen+i];
        //printf("Old value: %d, New Value: %d\n\n", data[j*rowlen+i], newD);
        data[j*rowlen+i] = newD;
      }
    }
  }
}

int checkDiffusion(grid* g){
    int i,j;
    int debug1=1,debug2=2,debug3=3;
    int rowlen = g->size;
    data_t *data = g->diff_matrix;
    data_t *obstacles = g->obs_matrix;
    for(i = 1; i < rowlen-1; i++){
        for(j = 1; j < rowlen-1; j++){
            if(obstacles[i*rowlen+j] && (data[i*rowlen+j] <= 0)){return 0;}
        }
    }
    return 1;
}

void traversePath(grid* g){
    int rowlen = g->size;
    data_t *data = g->diff_matrix;
    struct agent* a = g->agents;
    int i, x = a->sx, y = a->sy, arrsize = rowlen*rowlen;
    float currentSpot, target;
    float neighborMax;
    int neighborIter;
    int pathLength = 0;

    FILE *fp;
    fp = fopen("diff_output.txt","w+");
    target = data[a->dy*rowlen+a->dx];
    currentSpot = data[y*rowlen+x];
    printf("Source Value: [%d,%d] : %f\n", x, y, currentSpot);
    printf("Target Value: [%d,%d] : %f\n", a->dx, a->dy, target);
    while(currentSpot != target){
        neighborMax = 0;
        if(data[(x-1)*rowlen+y] > neighborMax){neighborMax = data[(x-1)*rowlen+y]; neighborIter = 1;} //x--
        if(data[(x+1)*rowlen+y] > neighborMax){neighborMax = data[(x+1)*rowlen+y]; neighborIter = 2;} //x++
        if(data[x*rowlen+y+1] > neighborMax){neighborMax = data[x*rowlen+y+1]; neighborIter = 3;}
        if(data[x*rowlen+y-1] > neighborMax){neighborMax = data[x*rowlen+y-1]; neighborIter = 4;}
        printf("%d", neighborIter);
        if(neighborIter == 1){x--;}
        else if(neighborIter == 2){x++;}
        else if(neighborIter == 3){y++;}
        else if(neighborIter == 4){y--;}
        currentSpot = data[x*rowlen+y];
        //fp << "- [" << (x)%rowlen << ", " << (y)/rowlen << "] | D = " << currentSpot << endl;
        printf("- [%d,%d] | D = %f\n", x, y, currentSpot);
        fprintf(fp, "- [%d,%d] | D = %f\n", x, y, currentSpot);
        pathLength++;

    }
    //fp << "Path Length: " << pathLength << endl;
    fprintf(fp, "Path Length: %d", pathLength);
    fclose(fp);
}

int getNeighborAvg(grid* g, int ax, int ay){

    data_t runsum = 0;
    int rowlen = g->size;
    int numNeighbors = 0;
    data_t *data = g->diff_matrix;
    //printf("Analyzing point [%d, %d]:\n", ax, ay);
    runsum += (data_t)data[ay*rowlen+ax-1]; ///*printf("\tAdding Left neighbor %d to runsum\n", data[ax*rowlen+ay-1]);*/ numNeighbors++;}
    runsum += (data_t)data[ay*rowlen+ax+1]; ///*printf("\tAdding Right neighbor %d to runsum\n", data[ax*rowlen+ay+1]);*/ numNeighbors++;}
    runsum += (data_t)data[(ay-1)*rowlen+ax]; ///*printf("\tAdding Top neighbor %d to runsum\n", data[(ax-1)*rowlen+ay]);*/ numNeighbors++;}
    runsum += (data_t)data[(ay+1)*rowlen+ax]; ///*printf("\tAdding Bottom neighbor %d to runsum\n", data[(ax+1)*rowlen+ay]);*/ numNeighbors++;}

    return runsum/4;
}
=======
/****************************************************************************

   gcc -O1 main.c -o main

*/

#include "LoadMap.h"

/* Prototypes */
void diffuse(grid* g);
int checkDiffusion(grid* g);
void traversePath(grid* g);
int getNeighborAvg(grid* g, int ax, int ay);
/*****************************************************************************/
int main(int argc, char *argv[])
{
    /* declare and initialize the array */
    grid* myGrid = LoadGrid("map_8by8_obst12_agents1_ex0.yaml"); //this creates the grid to use

    /*this runs the entire diffusion process and will run until the graph is fully diffused.*/
    diffuse(myGrid);
    PrintGrid(myGrid);
    traversePath(myGrid);

} /* end main */

/************************************/

/* Diffuse Function */
void diffuse(grid* g)
{

  long int i, j;
  int rowlen = g->size;
  data_t *data = g->diff_matrix;
  data_t *obstacles = g->obs_matrix;
  data_t newD;
  int destX = g->dx, destY = g->dy;
  int sum;

  int large = rowlen*rowlen*rowlen*rowlen*rowlen;

  while (!checkDiffusion(g)) {
    PrintGrid(g);
    printf("\n");
    for (i = 1; i < rowlen-1; i++) {
      for (j = 1; j < rowlen-1; j++) {
        data[destY*rowlen+destX] = large;
        //newD = .25 * (data[(i-1)*rowlen+j] + data[(i+1)*rowlen+j] + data[i*rowlen+j+1] + data[i*rowlen+j-1]) * obstacles[destX*rowlen+destY];
        sum = getNeighborAvg(g, i, j);
        newD = sum * obstacles[j*rowlen+i];
        //printf("Old value: %d, New Value: %d\n\n", data[j*rowlen+i], newD);
        data[j*rowlen+i] = newD;
      }
    }
  }
}

int checkDiffusion(grid* g){
    int i,j;
    int debug1=1,debug2=2,debug3=3;
    int rowlen = g->size;
    data_t *data = g->diff_matrix;
    data_t *obstacles = g->obs_matrix;
    for(i = 1; i < rowlen-1; i++){
        for(j = 1; j < rowlen-1; j++){
            if(obstacles[i*rowlen+j] && (data[i*rowlen+j] <= 0)){return 0;}
        }
    }
    return 1;
}

void traversePath(grid* g){
    int rowlen = g->size;
    data_t *data = g->diff_matrix;
    struct agent* a = g->agents;
    int i, x = a->sx, y = a->sy, arrsize = rowlen*rowlen;
    float currentSpot, target;
    float neighborMax;
    int neighborIter;
    int pathLength = 0;

    FILE *fp;
    fp = fopen("diff_output.txt","w+");
    target = data[a->dy*rowlen+a->dx];
    currentSpot = data[y*rowlen+x];
    printf("Source Value: [%d,%d] : %f\n", x, y, currentSpot);
    printf("Target Value: [%d,%d] : %f\n", a->dx, a->dy, target);
    while(currentSpot != target){
        neighborMax = 0;
        if(data[(x-1)*rowlen+y] > neighborMax){neighborMax = data[(x-1)*rowlen+y]; neighborIter = 1;} //x--
        if(data[(x+1)*rowlen+y] > neighborMax){neighborMax = data[(x+1)*rowlen+y]; neighborIter = 2;} //x++
        if(data[x*rowlen+y+1] > neighborMax){neighborMax = data[x*rowlen+y+1]; neighborIter = 3;}
        if(data[x*rowlen+y-1] > neighborMax){neighborMax = data[x*rowlen+y-1]; neighborIter = 4;}
        printf("%d", neighborIter);
        if(neighborIter == 1){x--;}
        else if(neighborIter == 2){x++;}
        else if(neighborIter == 3){y++;}
        else if(neighborIter == 4){y--;}
        currentSpot = data[x*rowlen+y];
        //fp << "- [" << (x)%rowlen << ", " << (y)/rowlen << "] | D = " << currentSpot << endl;
        printf("- [%d,%d] | D = %f\n", x, y, currentSpot);
        fprintf(fp, "- [%d,%d] | D = %f\n", x, y, currentSpot);
        pathLength++;

    }
    //fp << "Path Length: " << pathLength << endl;
    fprintf(fp, "Path Length: %d", pathLength);
    fclose(fp);
}

int getNeighborAvg(grid* g, int ax, int ay){

    data_t runsum = 0;
    int rowlen = g->size;
    int numNeighbors = 0;
    data_t *data = g->diff_matrix;
    //printf("Analyzing point [%d, %d]:\n", ax, ay);
    runsum += (data_t)data[ay*rowlen+ax-1]; ///*printf("\tAdding Left neighbor %d to runsum\n", data[ax*rowlen+ay-1]);*/ numNeighbors++;}
    runsum += (data_t)data[ay*rowlen+ax+1]; ///*printf("\tAdding Right neighbor %d to runsum\n", data[ax*rowlen+ay+1]);*/ numNeighbors++;}
    runsum += (data_t)data[(ay-1)*rowlen+ax]; ///*printf("\tAdding Top neighbor %d to runsum\n", data[(ax-1)*rowlen+ay]);*/ numNeighbors++;}
    runsum += (data_t)data[(ay+1)*rowlen+ax]; ///*printf("\tAdding Bottom neighbor %d to runsum\n", data[(ax+1)*rowlen+ay]);*/ numNeighbors++;}

    return runsum/4;
}
>>>>>>> Stashed changes
