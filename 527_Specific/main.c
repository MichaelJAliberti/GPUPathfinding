/****************************************************************************


   gcc -O1 main.c -o main "map_8by8_obst12_agents1_ex0.yaml"

*/

#include "LoadMap.h"

/* Prototypes */
void diffuse(grid* g);
int checkDiffusion(grid* g);
void traversePath(grid* g);
float getNeighborSum(grid* g, int ax, int ay);

/*****************************************************************************/
int main(int argc, char *argv[])
{
    /* declare and initialize the array */
    grid* myGrid = LoadGrid("map_8by8_obst12_agents1_ex0.yaml"); //this creates the grid to use

    /*this runs the entire diffusion process and will run until the graph is fully diffused.*/
    PrintGrid(myGrid);
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
  double newD;
  int destX = g->dx, destY = g->dy;
  float sum;

  while (!checkDiffusion(g)) {
    data[destX*rowlen+destY] = rowlen;
    for (i = 0; i < rowlen; i++) {
      for (j = 0; j < rowlen; j++) {
        //newD = .25 * (data[(i-1)*rowlen+j] + data[(i+1)*rowlen+j] + data[i*rowlen+j+1] + data[i*rowlen+j-1]) * obstacles[destX*rowlen+destY];
        sum = getNeighborSum(g, i, j);
        newD = .25 * sum * obstacles[destX*rowlen+destY];
        data[i*rowlen+j] = newD;
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
            if(!obstacles[i*rowlen+j] && (data[i*rowlen+j] <= 0)){return 0;}
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

    currentSpot = data[x*rowlen+y];
    while(currentSpot != arrsize){
        neighborMax = 0;
        if(data[(x-1)*rowlen+y] > neighborMax){x--;}
        if(data[(x+1)*rowlen+y] > neighborMax){x++;}
        if(data[x*rowlen+y+1] > neighborMax){y++;}
        if(data[x*rowlen+y-1] > neighborMax){y--;}

        currentSpot = data[x*rowlen+y];
        //fp << "- [" << (x)%rowlen << ", " << (y)/rowlen << "] | D = " << currentSpot << endl;
        fprintf(fp, "- [%d,%d] | D = %f", (x)%rowlen, (y)/rowlen, currentSpot);
        pathLength++;



    }
    //fp << "Path Length: " << pathLength << endl;
    fprintf(fp, "Path Length: %d", pathLength);
    fclose(fp);
}

float getNeighborSum(grid* g, int ax, int ay){

    float runsum = 0;
    int rowlen = g->size;
    printf("Rowlen: %f\n", g->size);
    data_t *data = g->diff_matrix;
    printf("Top: %f\n",data[ax*rowlen+ay-1]);
    printf("Bottom: %f\n",data[ax*rowlen+ay+1]);
    printf("Left: %f\n",data[(ax-1)*rowlen+ay]);
    printf("Right: %f\n",data[(ax+1)*rowlen+ay]);
    if(ax > 0){runsum += data[ax*rowlen+ay-1]; printf("Adding top neighbor %f to runsum\n", data[ax*rowlen+ay-1]);}
    if(ax < rowlen){runsum += data[ax*rowlen+ay+1]; printf("Adding bottom neighbor %f to runsum\n", data[ax*rowlen+ay+1]);}
    if(ay > 0){runsum += data[(ax-1)*rowlen+ay]; printf("Adding left neighbor %f to runsum\n", data[(ax-1)*rowlen+ay]);}
    if(ay < rowlen){runsum += data[(ax+1)*rowlen+ay]; printf("Adding right neighbor %f to runsum\n", data[(ax+1)*rowlen+ay]);}
    printf("[%d,%d] Runsum: %2f\n\n",ax,ay,runsum);
    return runsum;
}
