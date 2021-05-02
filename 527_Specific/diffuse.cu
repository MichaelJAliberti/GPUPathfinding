/*


     nvcc -arch compute_70 -code sm_70 diffuse.cu -o diffuse
     ./diffuse 2048


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

__global__ void gpu_diffuse (int size, data_t* diffuseMap, data_t* obstacleMap, int dx, int dy);
void cpu_diffuse (int size, data_t* diffuseMap, data_t* obstacleMap, int dx, int dy);
__global__ int gpu_checkDiffusion (int size, data_t* diffuseMap, data_t* obstacleMap)
int cpu_checkDiffusion(int size, data_t* diffuseMap, data_t* obstacleMap);
void verify (int size, float* arrGPU, float* arrCPU);
double interval(struct timespec start, struct timespec end);


int main(int argc, char *argv[]){
  //timing variables
  struct timespec time_start, time_stop;
  double elapsed_cpu;
  // GPU Timing variables
  cudaEvent_t start, stop;
  cudaEvent_t kernal_start, kernal_stop;
  float elapsed_gpu, elapsed_kernal;
  // Arrays on GPU global memory
  float *gpu_dMap;
  float *gpu_oMap;
  // Output array on host memory
  float *host_dMap

  /* retrieve input file */
  grid* myGrid;
  if (argc == 2){
      myGrid = LoadGrid(argv[1]); //this creates the grid to use
  }
  else{
      printf("Requires input: filepath\n");
      exit(1);
  }

  // Input vars for gpu_diffuse and cpu_diffuse
  int size = myGrid->size;
  int row_bound = size + 1;
  int mod_size = size + 2;
  data_t* diffuseMap = myGrid->diff_matrix;
  data_t* obstacleMap = myGrid->obs_matrix;
  int dx = myGrid->dx;
  int dy = myGrid->dy;

    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

  // Allocate GPU memory
  size_t allocSize = mod_size * mod_size * sizeof(data_t);
  CUDA_SAFE_CALL(cudaMalloc((void **)&gpu_dMap, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&gpu_oMap, allocSize));

  // Allocate output diffusion map
  host_dMap = (float *) malloc(allocSize);

#if PRINT_TIME
  // Create the cuda events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&kernal_start);
  cudaEventCreate(&kernal_stop);
  // Record event on the default stream
  cudaEventRecord(start, 0);
#endif

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(gpu_dMap, diffuseMap, allocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(gpu_oMap, obstacleMap, allocSize, cudaMemcpyHostToDevice));

  // Defining single block dimensions
  dim3 dimBlock(16,16);
  dim3 dimGrid(size/dimBlock.x, size/dimBlock.y);

  // Launch the kernel
  cudaEventRecord(kernal_start, 0);
  gpu_diffuse<<<dimGrid, dimBlock>>>(mod_size, gpu_dMap, gpu_oMap, dx, dy);
  cudaDeviceSynchronize();
  gpu_checkDiffusion<<<dimGrid, dimBlock>>>(mod_size, gpu_dMap, gpu_oMap);
  cudaEventRecord(kernal_stop, 0);
  cudaEventSynchronize(stop);
  // Check for errors during launch
  CUDA_SAFE_CALL(cudaPeekAtLastError());

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(host_dMap, d_c, allocSize, cudaMemcpyDeviceToHost));

#if PRINT_TIME
  // Stop and destroy the timer
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_gpu, start, stop);
  cudaEventElapsedTime(&elapsed_kernal, kernal_start, kernal_stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventDestroy(kernal_start);
  cudaEventDestroy(kernal_stop);
#endif

  // Compute the results on the host //EDIT
 clock_gettime(CLOCK_REALTIME, &time_start);
 cpu_diffuse(mod_size, diffuseMap, obstacleMap, dx, dy);
 clock_gettime(CLOCK_REALTIME, &time_stop);
 elapsed_cpu = interval(time_start, time_stop);

  // Compare the results //EDIT
  verify(arrLen, host_dMap, diffuseMap);

  //Printing runtimes
  printf("\nGPU Time:    %f (msec))\n", elapsed_gpu);
  printf("\nKernal Time: %f (msec))\n", elapsed_kernal);
  printf("\nCPU Time:    %f (msec))\n", elapsed_cpu);

  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(gpu_dMap));
  CUDA_SAFE_CALL(cudaFree(gpu_oMap));

  free(host_dMap);
  free(diffuseMap);
  free(obstacleMap);

  return 0;
}

__global__ void gpu_diffuse (int size, data_t* diffuseMap, data_t* obstacleMap, int dx, int dy) {
    const int bId_x= blockIdx.x;
    const int bId_y= blockIdx.y;
    // Local thread index
    const int local_tid_x= threadIdx.x;
    const int local_tid_y= threadIdx.y;
    // Number of rows and columns of the result matrix to be evaluated by each block
    const int rows_per_block= size/ gridDim.x;
    const int cols_per_block= size/ gridDim.y;
    const int rows_per_thread= rows_per_block/ blockDim.x;
    const int cols_per_thread= cols_per_block/ blockDim.y;
    // Row and column indices of the result matrix that the current block has to compute
    const int blockStartId_row= bId_x* rows_per_block;
    const int blockEndId_row= (bId_x+1) * rows_per_block- 1;
    const int blockStartId_col= bId_y* cols_per_block;
    const int blockEndId_col= (bId_y+1) * cols_per_block- 1;

    const int threadStartId_row= blockStartId_row+ local_tid_x* rows_per_thread;
    const int threadEndId_row= blockStartId_row+ (local_tid_x+1)* rows_per_thread- 1;
    const int threadStartId_col= blockStartId_col+ local_tid_y* cols_per_thread;
    const int threadEndId_col= blockStartId_col+ (local_tid_y+1)* cols_per_thread- 1;

    long int i, j;
    data_t newD;
    data_t large = (data_t) size*size*10000000000;

    for (i = threadStartId_row; i < threadEndId_row; i++) {
        for (j = threadStartId_col; j < threadEndId_col; j++) {
            if(i > 0 && j > 0 && i < size-1 && j < size-1){
                diffuseMap[dy * size + dx] = large;
                newD = (data_t) .25 * (diffuseMap[(i-1)*size + j] + diffuseMap[(i+1)*size + j] + diffuseMap[i*size + j+1] + diffuseMap[i*size + j-1]) * obstacleMap[i*size + j];
                diffuseMap[i*size + j] = newD;
            }
        }
    }
}

void cpu_diffuse (int size, data_t* diffuseMap, data_t* obstacleMap, int dx, int dy) {
    long int i, j;
    data_t newD;
    data_t large = (data_t) size*size*10000000000;

    while (!checkDiffusion(g)){ /**/
      for (i = 1; i < size-1; i++) {
          for (j = 1; j < size-1; j++) {
              diffuseMap[dy * size + dx] = large;
              newD = (data_t) .25 * (diffuseMap[(i-1)*size + j] + diffuseMap[(i+1)*size + j] + diffuseMap[i*size + j+1] + diffuseMap[i*size + j-1]) * obstacleMap[i*size + j];
              diffuseMap[i*size + j] = newD;
            }
        }
    }
}

__global__ int gpu_checkDiffusion (int size, data_t* diffuseMap, data_t* obstacleMap) {
    const int bId_x= blockIdx.x;
    const int bId_y= blockIdx.y;
    // Local thread index
    const int local_tid_x= threadIdx.x;
    const int local_tid_y= threadIdx.y;
    // Number of rows and columns of the result matrix to be evaluated by each block
    const int rows_per_block= size/ gridDim.x;
    const int cols_per_block= size/ gridDim.y;
    const int rows_per_thread= rows_per_block/ blockDim.x;
    const int cols_per_thread= cols_per_block/ blockDim.y;
    // Row and column indices of the result matrix that the current block has to compute
    const int blockStartId_row= bId_x* rows_per_block;
    const int blockEndId_row= (bId_x+1) * rows_per_block- 1;
    const int blockStartId_col= bId_y* cols_per_block;
    const int blockEndId_col= (bId_y+1) * cols_per_block- 1;

    const int threadStartId_row= blockStartId_row+ local_tid_x* rows_per_thread;
    const int threadEndId_row= blockStartId_row+ (local_tid_x+1)* rows_per_thread- 1;
    const int threadStartId_col= blockStartId_col+ local_tid_y* cols_per_thread;
    const int threadEndId_col= blockStartId_col+ (local_tid_y+1)* cols_per_thread- 1;

    long int i, j;
    data_t newD;
    data_t large = (data_t) size*size*10000000000;

    for (i = threadStartId_row; i < threadEndId_row; i++) {
        for (j = threadStartId_col; j < threadEndId_col; j++) {
            if(obstacleMap[i*size+j] && (diffuseMap[i*size+j] == 0)){return 0;}
        }
    }
    return 1;
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

void verify (int arrLen, float* arrGPU, float* arrCPU){
    fstream file;
    file.open("error_report.txt");
    //printf("\n[i][j]\t\tGPU\t\tCPU\n");
    file << "[i][j]\t\t\t\tGPU\t\t\t\tCPU\n";
    int i, errCount=0;
    int y = arrLen*arrLen;
    for(i = 0; i < y; i++) {
      if (arrGPU[i] != arrCPU[i]) {

        //printf("[%d][%d], \t\t%f, \t\t%f\n", i/arrLen, i%arrLen, arrGPU[i], arrCPU[i]);
        file << "[" << i/arrLen << "][" << i%arrLen << "]\t\t\t" << arrGPU[i] << "\t\t\t" << arrCPU[i] << "\n";
        errCount++;
      }
    }
    printf("\nError Report: %d mismatches were found.\n", errCount);
    file.close();
}

double interval(struct timespec start, struct timespec end){
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9)*1000;
}
