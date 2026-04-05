#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>
#include <atomic>

#define LAYERS 10
#define DEBUG_MODE 0 
#define POOL_SIZE 64 

using namespace std;

/**
 * Most of this code is written by Gemini 3.0 Flash.
 * Gemini 3.0 Pro was needed to help me find the final synchronization bug
 * (It took me an embarassingly long time to find it so pls don't ask for my chat log)
 * Prompted with my Pooling, 2-Stream approach that I formulated almost entirely independently
*/

// Wrapper for warpReduce (defined previously)
__device__ float warpReduce(float init_val) {
    float value = init_val;
    for (int i = 16; i >= 1; i /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, i, 32);
    }
    return value;
}

// SECTION 3: GPU Processing Kernel
__global__ void gpu_kernel(float *in_buff, float *weights, float *out_cat) {
    __shared__ float buffer[256];
    int id = threadIdx.x;
    int block_offset = blockIdx.x * 256;
    
    buffer[id] = in_buff[block_offset + id];
    __syncthreads();

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
        atomicAdd(&out_cat[blockIdx.x], min_result);
    }
}

int main(int argc, char *argv[]) {
    if (argc <= 1) exit(-1);
    int runlen = atoi(argv[1]);
    if (!runlen) exit(-1);

    // Initial Weights Setup
    float *weight, *dev_weight;
    weight = (float *)malloc(sizeof(float) * 256 * 256 * LAYERS);
    for (int i = 0; i < LAYERS; ++i) {
        for (int j = 0; j < 256 * 256; ++j) {
            weight[i * 256 * 256 + j] = (i + 1.0) * 1e-5;
        }
    }
    cudaMalloc(&dev_weight, sizeof(float) * 256 * 256 * LAYERS);
    cudaMemcpy(dev_weight, weight, sizeof(float) * 256 * 256 * LAYERS, cudaMemcpyHostToDevice);

    // Pipeline Resources (Double Buffering)
    cudaStream_t streams[2];
    float *h_pools[2], *d_pools[2], *h_results[2], *d_results[2];
    atomic<bool> buffer_ready[2]; // Flags to sync Producer and Consumer

    for (int i = 0; i < 2; ++i) {
        cudaStreamCreate(&streams[i]);
        cudaHostAlloc(&h_pools[i], sizeof(float) * 256 * POOL_SIZE, cudaHostAllocDefault);
        cudaHostAlloc(&h_results[i], sizeof(float) * POOL_SIZE, cudaHostAllocDefault);
        cudaMalloc(&d_pools[i], sizeof(float) * 256 * POOL_SIZE);
        cudaMalloc(&d_results[i], sizeof(float) * POOL_SIZE);
        buffer_ready[i] = false;
    }

    double st, et;
    st = omp_get_wtime();

    int total_pools = (runlen + POOL_SIZE - 1) / POOL_SIZE;

    // Use a single parallel region
    #pragma omp parallel
    {
        // One thread acts as the "Manager" to spawn the tasks
        #pragma omp single
        {
            // --- TASK 1: THE PRODUCER (Sensor) ---
            #pragma omp task
            {
                for (int p = 0; p < total_pools; ++p) {
                    int b_idx = p % 2;
                    int current_size = (p == total_pools - 1) ? (runlen - p * POOL_SIZE) : POOL_SIZE;

                    // Wait for Consumer to finish with this specific buffer
                    while (buffer_ready[b_idx].load(std::memory_order_acquire)) { 
                        #pragma omp taskyield 
                    }

                    // Sequential sensor collection
                    for (int s = 0; s < current_size; ++s) {
                        int run_id = p * POOL_SIZE + s;
                        for (int i = 0; i < 256; ++i) {
                            h_pools[b_idx][s * 256 + i] = (run_id + i) * 1e-3;
                        }
                    }

                    // Mark buffer as ready for GPU
                    buffer_ready[b_idx].store(true, std::memory_order_release);
                }
            }

            // --- TASK 2: THE CONSUMER (GPU Manager) ---
            #pragma omp task
            {
                for (int p = 0; p < total_pools; ++p) {
                    int b_idx = p % 2;
                    int current_size = (p == total_pools - 1) ? (runlen - p * POOL_SIZE) : POOL_SIZE;

                    // 1. Wait for Producer to fill the CURRENT buffer
                    while (!buffer_ready[b_idx].load(std::memory_order_acquire)) { 
                        #pragma omp taskyield 
                    }

                    // 2. LAUNCH CURRENT POOL ASYNC
                    // Submit work to the GPU's internal queue immediately
                    cudaMemsetAsync(d_results[b_idx], 0, sizeof(float) * POOL_SIZE, streams[b_idx]);
                    cudaMemcpyAsync(d_pools[b_idx], h_pools[b_idx], sizeof(float) * 256 * current_size, 
                                    cudaMemcpyHostToDevice, streams[b_idx]);
                    
                    gpu_kernel<<<current_size, 256, 0, streams[b_idx]>>>(d_pools[b_idx], dev_weight, d_results[b_idx]);
                    
                    cudaMemcpyAsync(h_results[b_idx], d_results[b_idx], sizeof(float) * current_size, 
                                    cudaMemcpyDeviceToHost, streams[b_idx]);

                    // 3. SYNCHRONIZE AND PRINT THE *PREVIOUS* POOL
                    // This is the "Software Pipeline" fix. We queued pool `p`, now we wait for `p-1`.
                    if (p > 0) {
                        int prev_p = p - 1;
                        int prev_b_idx = prev_p % 2;
                        int prev_size = (prev_p == total_pools - 1) ? (runlen - prev_p * POOL_SIZE) : POOL_SIZE;

                        cudaStreamSynchronize(streams[prev_b_idx]);
                        
                        for (int i = 0; i < prev_size; ++i) {
                            printf("%d %e\n", prev_p * POOL_SIZE + i, h_results[prev_b_idx][i]);
                        }

                        #if DEBUG_MODE
                        printf(">> Pool %d finished on Stream %d\n", prev_p, prev_b_idx);
                        #endif

                        // Release PREVIOUS buffer back to Producer
                        buffer_ready[prev_b_idx].store(false, std::memory_order_release);
                    }
                }

                // 4. CLEANUP: SYNCHRONIZE AND PRINT THE VERY LAST POOL
                // The loop finishes after launching the last pool, so we must sync/print it here outside the loop.
                if (total_pools > 0) {
                    int last_p = total_pools - 1;
                    int last_b_idx = last_p % 2;
                    int last_size = (last_p == total_pools - 1) ? (runlen - last_p * POOL_SIZE) : POOL_SIZE;

                    cudaStreamSynchronize(streams[last_b_idx]);
                    
                    for (int i = 0; i < last_size; ++i) {
                        printf("%d %e\n", last_p * POOL_SIZE + i, h_results[last_b_idx][i]);
                    }

                    #if DEBUG_MODE
                    printf(">> Pool %d finished on Stream %d\n", last_p, last_b_idx);
                    #endif

                    buffer_ready[last_b_idx].store(false, std::memory_order_release);
                }
            }
        } // End Single
    } // End Parallel

    cudaDeviceSynchronize();
    et = omp_get_wtime();

    cout << (et - st) << " seconds for " << runlen << " runs" << endl;

    // Cleanup
    for (int i = 0; i < 2; ++i) {
        cudaStreamDestroy(streams[i]);
        cudaFreeHost(h_pools[i]);
        cudaFreeHost(h_results[i]);
        cudaFree(d_pools[i]);
        cudaFree(d_results[i]);
    }
    cudaFree(dev_weight);
    free(weight);

    return 0;
}