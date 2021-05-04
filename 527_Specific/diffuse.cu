/*


     nvcc -arch compute_70 -code sm_70 diffuse.cu -o diffuse
     ./diffuse /map_8by8_obst12_agents1_ex0.yaml


*/

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include <fstream>

#include "LoadMap.h"
using namespace std;

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true){
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
#define NUM_THREADS_PER_BLOCK   256
#define NUM_BLOCKS         1
#define PRINT_TIME         1
#define SM_ARR_LEN        2048
#define TOL            1e6

#define IMUL(a, b) __mul24(a, b)

/* structures */
struct point{
    int x;
    int y;
    int length;
    struct point* next;
};

__global__ void gpu_diffuse (int size, data_t* diffuseMap, data_t* obstacleMap, int dx, int dy);
void cpu_diffuse (int size, data_t* diffuseMap, data_t* obstacleMap, int dx, int dy);
__global__ void gpu_checkDiffusion (int size, data_t* diffuseMap, data_t* obstacleMap, int* checkMap);
int cpu_checkDiffusion(int size, data_t* diffuseMap, data_t* obstacleMap);
int gpu_checkDiffusionHost(int size, int* map);
struct point** traversePath(grid* g);
int printPath(int num_agents, struct point** paths, char* filein);
int PrintIntArray(int size, int* arr);
int PrintArray(int size, data_t* arr);




int main(int argc, char *argv[]){
    printf("Start\n");
    int flag = 0;

    /* retrieve input file */
    grid* cpuGrid;
    grid* gpuGrid;
    if (argc == 2){
      cpuGrid = LoadGrid(argv[1]);
      gpuGrid = LoadGrid(argv[1]); //this creates the grid to use
    }
    else{
      printf("Requires input: filepath\n");
      exit(1);
    }
    // Essential data from grid
    int size = cpuGrid->size;
    int mod_size = size + 2;
    int dx = cpuGrid->dx;
    int dy = cpuGrid->dy;


    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

    size_t allocSize = mod_size * mod_size * sizeof(data_t);
    size_t intAllocSize = mod_size * mod_size * sizeof(int);

    // Allocate host memory
    data_t *h_dMap;
    data_t *h_oMap;
    int *h_dCheck;
    data_t* diffuseMap = cpuGrid->diff_matrix;
    data_t* obstacleMap = cpuGrid->obs_matrix;
    CUDA_SAFE_CALL(cudaMallocHost((void **)&h_dMap, allocSize));
    CUDA_SAFE_CALL(cudaMallocHost((void **)&h_oMap, allocSize));
    CUDA_SAFE_CALL(cudaMallocHost((void **)&h_dCheck, intAllocSize));
    // h_dMap = (data_t*)malloc(allocSize);
    // h_oMap = (data_t*)malloc(allocSize);
    // h_dCheck = (int*)malloc(intAllocSize);
    memcpy(h_dMap, diffuseMap, allocSize);
    memcpy(h_oMap, obstacleMap, allocSize);
    memset(h_dCheck, 0, mod_size*mod_size*sizeof(int));

    // Allocate GPU memory

    data_t *d_dMap;
    data_t *d_oMap;
    int *d_dCheck;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_dMap, allocSize));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_oMap, allocSize));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_dCheck, intAllocSize));

    // Transfer the arrays to the GPU memory
    CUDA_SAFE_CALL(cudaMemcpy(d_dMap, h_dMap, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_oMap, h_oMap, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_dCheck, h_dCheck, intAllocSize, cudaMemcpyHostToDevice));
    // Defining single block dimensions
    dim3 dimBlock(8,8);
    dim3 dimGrid(size/dimBlock.x, size/dimBlock.y);
    printf("Running GPU diffusion...\n");
    // Run diffusion on GPU
    int it = 0;
    int fullsize = size * size;
    while(!flag && it < fullsize){
        it++;
        for(int i = 0; i < size; i++){
            //printf("Diffusing...\n");
            gpu_diffuse<<<dimGrid, dimBlock>>>(mod_size, d_dMap, d_oMap, dx, dy);
        }

        //printf("Analyzing diffusion...\n");
        gpu_checkDiffusion<<<dimGrid, dimBlock>>>(mod_size, d_dMap, d_oMap, d_dCheck);
        //printf("Transferring diffusion check to host...\n");
        CUDA_SAFE_CALL(cudaMemcpy(h_dCheck, d_dCheck, intAllocSize, cudaMemcpyDeviceToHost));
        //printf("Checking diffusion...\n");
        cudaDeviceSynchronize();
        flag = gpu_checkDiffusionHost(mod_size, h_dCheck);
        //printf("Flag: %d\n",flag);
    }
    printf("GPU diffusion finished.\n");

    // Check for errors during launch
    CUDA_SAFE_CALL(cudaPeekAtLastError());

    // Transfer the results back to the host
    CUDA_SAFE_CALL(cudaMemcpy(h_dMap, d_dMap, allocSize, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    gpuGrid->diff_matrix = h_dMap;
    PrintArray(mod_size,h_dMap);


    // Compute the results on the host //EDIT
    printf("Running CPU diffusion...\n");
    cpu_diffuse(mod_size, diffuseMap, obstacleMap, dx, dy);
    printf("CPU diffusion finished.\n");
    PrintArray(mod_size, diffuseMap);

    // Compare the results //EDIT
    struct point** cpuPaths = traversePath(cpuGrid);
    struct point** gpuPaths = traversePath(gpuGrid);
    printPath(cpuGrid->num_agents, cpuPaths, argv[1]);
    printPath(gpuGrid->num_agents, gpuPaths, argv[1]);

    // Free-up device and host memory
    CUDA_SAFE_CALL(cudaFree(d_dMap));
    CUDA_SAFE_CALL(cudaFree(d_oMap));
    CUDA_SAFE_CALL(cudaFree(d_dCheck));

    CUDA_SAFE_CALL(cudaFreeHost(h_dMap));
    CUDA_SAFE_CALL(cudaFreeHost(h_oMap));
    CUDA_SAFE_CALL(cudaFreeHost(h_dCheck));


    return 0;
}

__global__ void gpu_diffuse (int size, data_t* diffuseMap, data_t* obstacleMap, int dx, int dy) {
    int i = blockIdx.y*blockDim.y+threadIdx.y+1;
    int j = blockIdx.x*blockDim.x+threadIdx.x+1;

    data_t newD;
    data_t large = (data_t) size*size*100;

    if(i > 0 && j > 0 && i < size-1 && j < size-1){
        diffuseMap[dy * size + dx] = large;
        newD = (data_t) .25 * (diffuseMap[(i-1)*size + j] + diffuseMap[(i+1)*size + j] + diffuseMap[i*size + j+1] + diffuseMap[i*size + j-1]) * obstacleMap[i*size + j];
        diffuseMap[i*size + j] = newD;
    }

}

__global__ void gpu_checkDiffusion (int size, data_t* diffuseMap, data_t* obstacleMap, int* checkMap) {
    int i = blockIdx.y*blockDim.y+threadIdx.y+1;
    int j = blockIdx.x*blockDim.x+threadIdx.x+1;

    checkMap[i*size+j] = !(((int)obstacleMap[i*size+j] && (diffuseMap[i*size+j] > 0)) || (!(int)obstacleMap[i*size+j] && !(diffuseMap[i*size+j] > 0)));

}

int gpu_checkDiffusionHost(int size, int* map){
    long int i, j;
    for (i = 1; i < size-1; i++) {
        for (j = 1; j < size-1; j++) {
            if(map[i*size+j] == 1){return 0;}
        }
    }
    return 1;
}

void cpu_diffuse (int size, data_t* diffuseMap, data_t* obstacleMap, int dx, int dy) {
    long int i, j;
    data_t newD;
    data_t large = (data_t) size*size*100;

    while (!cpu_checkDiffusion(size, diffuseMap, obstacleMap)){
      for (i = 1; i < size-1; i++) {
          for (j = 1; j < size-1; j++) {
              diffuseMap[dy * size + dx] = large;
              newD = (data_t) .25 * (diffuseMap[(i-1)*size + j] + diffuseMap[(i+1)*size + j] + diffuseMap[i*size + j+1] + diffuseMap[i*size + j-1]) * obstacleMap[i*size + j];
              diffuseMap[i*size + j] = newD;
            }
        }
    }
}

int cpu_checkDiffusion(int size, data_t* diffuseMap, data_t* obstacleMap){
    int i,j;
    for(i = 1; i < size-1; i++){
        for(j = 1; j < size-1; j++){
            if(obstacleMap[i*size+j] && (diffuseMap[i*size+j] == 0)){return 0;}
        }
    }
    return 1;
}

int PrintIntArray(int size, int* arr){
	int i, j, mult;

	for (i = 0; i < size; i++){
		mult = i*size;
		for (j = 0; j < size; j++){
			printf("%d,\t", arr[mult + j]);
		}
		printf("\n");
	}
	printf("\n");
	return 0;
}

int PrintArray(int size, data_t* arr){
	int i, j, mult;

	for (i = 0; i < size; i++){
		mult = i*size;
		for (j = 0; j < size; j++){
			printf("%.4f,\t", arr[mult + j]);
		}
		printf("\n");
	}
	printf("\n");
	return 0;
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
