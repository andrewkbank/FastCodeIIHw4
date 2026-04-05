#include <iostream>
#include <omp.h>

#define LAYERS 10

using namespace std;

__device__ float warpReduce(float init_val) {
    // Seed starting value as inverse lane ID
    float value = init_val;

    // Use XOR mode to perform butterfly reduction
    for (int i = 16; i >= 1; i /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, i, 32);
    }

    return value;
}

__global__ void gpu_kernel(float *in_buff, float *weights, float *out_cat) {
    __shared__ float buffer[256];

    int id = threadIdx.x;

    buffer[id] = in_buff[id];

    __syncthreads();

    // 10 layer ANN
    float tmp = 0.0;
    for (int layer = 0; layer < LAYERS; ++layer) {
        for (int i = 0; i < 256; ++i) {
            tmp += buffer[i] * weights[layer * 256 * 256 + id * 256 + i];
        }

        __syncthreads();

        buffer[id] = tmp;

        __syncthreads();
    }

    float min_result = warpReduce(buffer[id]);

    if (id % 32 == 0) {
        atomicAdd(out_cat, min_result);
    }
}

int main(int argc, char *argv[]) {
    float *recv_buffer, *dev_buff, *weight, *dev_weight;
    float category, *dev_cat;
    double st, et;

    if (argc <= 1) {
        exit(-1);
    }

    int runlen = atoi(argv[1]);

    if (!runlen) {
        exit(-1);
    }

    recv_buffer = (float *)malloc(sizeof(float) * 256);
    weight = (float *)malloc(sizeof(float) * 256 * 256 * LAYERS);

    // initialized weights
    for (int i = 0; i < LAYERS; ++i) {
        for (int j = 0; j < 256 * 256; ++j) {
            weight[i * 256 * 256 + j] = (i + 1.0) * 1e-5;
        }
    }

    cudaMalloc(&dev_weight, sizeof(float) * 256 * 256 * LAYERS);
    cudaMalloc(&dev_buff, sizeof(float) * 256);
    cudaMalloc(&dev_cat, sizeof(float));
    cudaMemset(dev_cat, 0, sizeof(float));

    cudaMemcpy(dev_weight, weight, sizeof(float) * 256 * 256 * LAYERS,
               cudaMemcpyHostToDevice);

    // create any streams or parallelization data structure above this line
    st = omp_get_wtime();

    for (int runs = 0; runs != runlen; ++runs) {
        // simulate receiving input from sensor
        for (int i = 0; i < 256; ++i) {
            recv_buffer[i] = (runs + i) * 1e-3;
        }

        // once buffer is full, send it to the GPU for processing
        cudaMemcpy(dev_buff, recv_buffer, sizeof(float) * 256,
                   cudaMemcpyHostToDevice);

        // Call GPU Kernel
        gpu_kernel<<<1, 256>>>(dev_buff, dev_weight, dev_cat);

        // copy result from GPU to CPU
        cudaDeviceSynchronize();
        cudaMemcpy(&category, dev_cat, sizeof(float), cudaMemcpyDeviceToHost);

        printf("%d %e\n", runs, category);
    }

    cudaDeviceSynchronize();

    et = omp_get_wtime();

    // remember to clean up your streams and data structure you have introduced.

    // do not change the code below this line

    cout << (et - st) << " seconds for " << runlen << " runs" << endl;

    cudaFree(dev_weight);
    cudaFree(dev_buff);
    cudaFree(dev_cat);

    free(recv_buffer);
}
