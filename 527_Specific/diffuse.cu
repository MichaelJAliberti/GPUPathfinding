/*


     nvcc -arch compute_70 -code sm_70 diffuse.cu -o diffuse
     ./diffuse /map_8by8_obst12_agents1_ex0.yaml


*/

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

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

#define IMUL(a, b) __mul24(a, b)

/* structures */
struct point{
    int x;
    int y;
    int length;
    struct point* next;
};

__global__ void gpu_diffuse (int size, data_t* diffuseMap, data_t* obstacleMap, int dx, int dy);
__global__ void gpu_checkDiffusion (int size, data_t* diffuseMap, data_t* obstacleMap, int* checkMap);
int gpu_checkDiffusionHost(int size, int* map);
void cpu_diffuse (int size, data_t* diffuseMap, data_t* obstacleMap, int dx, int dy);
int cpu_checkDiffusion(int size, data_t* diffuseMap, data_t* obstacleMap);
void printPath(int num_agents, struct point** paths, int args, char* filein);
int PrintIntArray(int size, int* arr);
int PrintArray(int size, data_t* arr);
double interval(struct timespec start, struct timespec end);
struct point** traversePath(grid* g);




int main(int argc, char *argv[]){
    printf("Start\n");
    int flag = 0;
    char * name = "Null.yaml";

    /* retrieve input file */
    grid* cpuGrid;
    grid* gpuGrid;
    if (argc == 2){
      cpuGrid = MakeGrid(atoi(argv[1]));
      gpuGrid = MakeGrid(atoi(argv[1])); //this creates the grid to use
    }
    else{
      printf("Requires input: filepath\n");
      exit(1);
    }
    // Essential variables
    int size = cpuGrid->size;
    int mod_size = size + 2;
    int dx = cpuGrid->dx;
    int dy = cpuGrid->dy;
    int it = 0;
    int fullsize = size * size;

    // Timing variables
    struct timespec time_start, time_stop;
    cudaEvent_t start, stop;
    cudaEvent_t kernal_start, kernal_stop;
    float elapsed_cpu, elapsed_gpu, elapsed_kernal;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernal_start);
    cudaEventCreate(&kernal_stop);


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

    // Recording timing including data transfers to and from device
    printf("Running GPU diffusion...\n");
    cudaEventRecord(start, 0);

    // Transfer the arrays to the GPU memory
    CUDA_SAFE_CALL(cudaMemcpy(d_dMap, h_dMap, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_oMap, h_oMap, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_dCheck, h_dCheck, intAllocSize, cudaMemcpyHostToDevice));
    // Defining single block dimensions
    dim3 dimBlock(8,8);
    dim3 dimGrid(size/dimBlock.x, size/dimBlock.y);
    // Run diffusion on GPU
    cudaEventRecord(kernal_start, 0);
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
        flag = gpu_checkDiffusionHost(mod_size, h_dCheck);
        //printf("Flag: %d\n",flag);
    }
    cudaEventRecord(kernal_stop, 0);
    cudaEventSynchronize(stop);
    // Transfer the results back to the host
    CUDA_SAFE_CALL(cudaMemcpy(h_dMap, d_dMap, allocSize, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    cudaEventElapsedTime(&elapsed_kernal, kernal_start, kernal_stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(kernal_start);
    cudaEventDestroy(kernal_stop);
    printf("GPU diffusion finished.\n");


    gpuGrid->diff_matrix = h_dMap;


    // Compute the results on the host //EDIT
    printf("Running CPU diffusion...\n");
    clock_gettime(CLOCK_REALTIME, &time_start);
    cpu_diffuse(mod_size, diffuseMap, obstacleMap, dx, dy);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    printf("CPU diffusion finished.\n");
    elapsed_cpu = interval(time_start, time_stop);
    cpuGrid->diff_matrix = diffuseMap;

    // Display timing results
    printf("\nGPU Time:    %f (msec))\n", elapsed_gpu);
    printf("\nKernal Time: %f (msec))\n", elapsed_kernal);
    printf("\nCPU Time:    %f (msec))\n", elapsed_cpu);

    // Compare the results //EDIT
    struct point** cpuPaths = traversePath(cpuGrid);
    struct point** gpuPaths = traversePath(gpuGrid);
    printPath(cpuGrid->num_agents, cpuPaths, argc, name);
    printPath(gpuGrid->num_agents, gpuPaths, argc, name);

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
    diffuseMap[dy * size + dx] = large;
    if(i > 0 && j > 0 && i < size-1 && j < size-1){
        newD = (data_t) .25 * (diffuseMap[(i-1)*size + j] + diffuseMap[(i+1)*size + j] + diffuseMap[i*size + j+1] + diffuseMap[i*size + j-1]) * obstacleMap[i*size + j];
        diffuseMap[i*size + j] = newD;
        diffuseMap[dy * size + dx] = large;
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

            pathLength++;
            //printf("AGENT%d: [%d, %d]\n", i, x, y);
        }
        paths[i]->length = pathLength;
        a = a->next;
    }
    return paths;
}

void printPath(int num_agents, struct point** paths, int args, char* filein)
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
