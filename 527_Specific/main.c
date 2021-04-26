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
    //traversePath(myGrid);

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
    for (i = 0; i < rowlen; i++) {
      for (j = 0; j < rowlen; j++) {
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
    for(i = 0; i < rowlen; i++){
        for(j = 0; j < rowlen; j++){
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
    float currentSpot;
    float neighborMax;
    int pathLength = 0;

    FILE *fp;
    fp = fopen("diff_output.txt","w+");

    currentSpot = data[y*rowlen+x];
    while(currentSpot != arrsize){
        neighborMax = 0;
        if(data[(x-1)*rowlen+y] > neighborMax){x--;}
        if(data[(x+1)*rowlen+y] > neighborMax){x++;}
        if(data[x*rowlen+y+1] > neighborMax){y++;}
        if(data[x*rowlen+y-1] > neighborMax){y--;}

        currentSpot = data[y*rowlen+x];
        //fp << "- [" << (x)%rowlen << ", " << (y)/rowlen << "] | D = " << currentSpot << endl;
        fprintf(fp, "- [%d,%d] | D = %f", (x)%rowlen, (y)/rowlen, currentSpot);
        pathLength++;

    }
    //fp << "Path Length: " << pathLength << endl;
    fprintf(fp, "Path Length: %d", pathLength);
    fclose(fp);
}

int getNeighborAvg(grid* g, int ax, int ay){

    int runsum = 0;
    int rowlen = g->size;
    int numNeighbors = 0;
    data_t *data = g->diff_matrix;
    //printf("Analyzing point [%d, %d]:\n", ax, ay);
    if(ax > 0){runsum += data[ay*rowlen+ax-1]; /*printf("\tAdding Left neighbor %d to runsum\n", data[ax*rowlen+ay-1]);*/ numNeighbors++;}
    if(ax < rowlen-1){runsum += data[ay*rowlen+ax+1]; /*printf("\tAdding Right neighbor %d to runsum\n", data[ax*rowlen+ay+1]);*/ numNeighbors++;}
    if(ay > 0){runsum += data[(ay-1)*rowlen+ax]; /*printf("\tAdding Top neighbor %d to runsum\n", data[(ax-1)*rowlen+ay]);*/ numNeighbors++;}
    if(ay < rowlen-1){runsum += data[(ay+1)*rowlen+ax]; /*printf("\tAdding Bottom neighbor %d to runsum\n", data[(ax+1)*rowlen+ay]);*/ numNeighbors++;}

    return runsum/numNeighbors;
}
