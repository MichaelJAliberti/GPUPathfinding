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
    grid* myGrid = LoadGrid("map_8by8_obst24_agents1_ex0.yaml"); //this creates the grid to use

    /*this runs the entire diffusion process and will run until the graph is fully diffused.*/
    diffuse(myGrid);
    PrintGrid(myGrid);
    //traversePath(myGrid);

} /* end main */

/************************************/

/* Diffuse Function */
void diffuse(grid* g)
{

  long int i, j, k;
  int rowlen = g->size;
  int row_bound = rowlen + 1;
  int mod_size = rowlen + 2;
  data_t *data = g->diff_matrix;
  data_t *obstacles = g->obs_matrix;
  data_t newD;
  int destX = g->dx, destY = g->dy;

  data_t large = (data_t) rowlen*rowlen*10000000000;

  k = 0;

  while (!checkDiffusion(g)){ /**/
    k++;
    PrintGrid(g);
    printf("\n");
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
