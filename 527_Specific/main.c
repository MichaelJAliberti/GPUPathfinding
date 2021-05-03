/****************************************************************************

   gcc -O1 main.c -o main
*/

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
int printPath(int num_agents, struct point** paths, char* filein);

/*****************************************************************************/
int main(int argc, char *argv[])
{
    /* retrieve input file */
    grid* myGrid;
    if (argc == 2){
        myGrid = LoadGrid(argv[1]); //this creates the grid to use
    }
    else{
        printf("Requires input: filepath\n");
        exit(1);
    }

    /*this runs the entire diffusion process and will run until the graph is fully diffused.*/
    diffuse(myGrid);
    struct point** paths = traversePath(myGrid);

    printPath(myGrid->num_agents, paths, argv[1]);
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
    //PrintGrid(g);
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

            //fp << "- [" << (x)%rowlen << ", " << (y)/rowlen << "] | D = " << currentSpot << endl;
            //fprintf(fp, "- [%d,%d] | D = %f\n", x, y, currentSpot);
            pathLength++;
        }

        paths[i]->length = pathLength;
        a = a->next;
    }

    return paths;
    //FILE *fp;
    //fprintf(fp, "Path Length: %d", pathLength);
    //fclose(fp);
}

int printPath(int num_agents, struct point** paths, char* filein)
{
    int i;
    struct point* pt_it;
    FILE *fp;
    char* new_name;

    // get new file name
    new_name = strtok(filein, ".");
    new_name = strcat(new_name, "_serial_path.yaml");

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