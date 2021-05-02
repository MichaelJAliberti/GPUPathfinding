/****************************************************************************

   gcc -O1 main.c -o main

*/

#include "LoadMap.h"

/* Prototypes */
void diffuse(grid* g);
int checkDiffusion(grid* g);
void traversePath(grid* g);
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
  int row_bound = rowlen + 1;
  int mod_size = rowlen + 2;
  data_t *data = g->diff_matrix;
  data_t *obstacles = g->obs_matrix;
  data_t newD;
  int destX = g->dx, destY = g->dy;

  data_t large = (data_t) rowlen*rowlen*10000000000;

  while (!checkDiffusion(g)){ /**/
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
    int row_bound = rowlen + 1;
    int mod_size = rowlen + 2;
    data_t *data = g->diff_matrix;
    struct agent* a = g->agents;
    int i, x = a->sx, y = a->sy, arrsize = rowlen*rowlen;
    float currentSpot, target;
    float neighborMax;
    int neighborIter;
    int pathLength = 0;

    FILE *fp;
    fp = fopen("diff_output.txt","w+");
    target = data[a->dy*mod_size+a->dx];
    currentSpot = data[y*mod_size+x];
    printf("Source Value: [%d,%d] : %f\n", y, x, currentSpot);
    printf("Target Value: [%d,%d] : %f\n", a->dy, a->dx, target);
    while(currentSpot != target){
        neighborMax = 0;
        if(data[(y-1)*mod_size+x] > neighborMax){neighborMax = data[(y-1)*mod_size+x]; neighborIter = 1;} //y--
        if(data[(y+1)*mod_size+x] > neighborMax){neighborMax = data[(y+1)*mod_size+x]; neighborIter = 2;} //y++
        if(data[y*mod_size+x+1] > neighborMax){neighborMax = data[y*mod_size+x+1]; neighborIter = 3;}
        if(data[y*mod_size+x-1] > neighborMax){neighborMax = data[y*mod_size+x-1]; neighborIter = 4;}
        if(neighborIter == 1){y--;}
        else if(neighborIter == 2){y++;}
        else if(neighborIter == 3){x++;}
        else if(neighborIter == 4){x--;}
        currentSpot = data[y*mod_size+x];
        //fp << "- [" << (x)%rowlen << ", " << (y)/rowlen << "] | D = " << currentSpot << endl;
        printf("- [%d,%d] | D = %f\n", y, x, currentSpot);
        fprintf(fp, "- [%d,%d] | D = %f\n", y, x, currentSpot);
        pathLength++;

    }
    //fp << "Path Length: " << pathLength << endl;
    fprintf(fp, "Path Length: %d", pathLength);
    fclose(fp);
}
