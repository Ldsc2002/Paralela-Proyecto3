#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <vector>
#include <jpeglib.h>
#include <string>
#include "pgm.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

/* 
Function: CPU_HoughTran
Purpose: CPU implementation of Hough Transform
Parameters:
    pic: pointer to input image
    w: width of input image
    h: height of input image
    acc: pointer to accumulator array
Returns: 
    None
 */
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc) {
    // Calculate the maximum possible radius
    float rMax = sqrt(w * w + h * h) / 2;

    // Initialize accumulator array
    *acc = new int[rBins * degreeBins];
    memset(*acc, 0, sizeof(int) * rBins * degreeBins);
    
    // Calculate the center of the image
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    // Perform Hough Transform
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            int idx = j * w + i;
            
            if (pic[idx] > 0) {
                // Calculate the radius for each degree
                int xCoord = i - xCent;
                int yCoord = yCent - j;
                float theta = 0;

                for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
                    // Calculate the radius
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;

                    // Increment the accumulator
                    (*acc)[rIdx * degreeBins + tIdx]++;
                    theta += radInc;
                }
            }
        }
    }
}

/* 
Function: GPU_HoughTran
Purpose: GPU implementation of Hough Transform
Parameters:
    pic: pointer to input image
    w: width of input image
    h: height of input image
    acc: pointer to accumulator array
    rMax: maximum possible radius
    rScale: scaling factor for radius
    d_Cos: pointer to pre-computed cosine values
    d_Sin: pointer to pre-computed sine values
Returns:
    None
 */
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float* d_Cos, float* d_Sin) {
    // Calculate global ID
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return; // Check if global ID is valid

    // Calculate the center of the image
    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    // Perform Hough Transform
    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            // Calculate the radius
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;

            // Increment the accumulator
            atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
}

/* 
Function: GPU_HoughTranshared
Purpose: GPU implementation of Hough Transform with shared memory
Parameters:
    pic: pointer to input image
    w: width of input image
    h: height of input image
    acc: pointer to accumulator array
    rMax: maximum possible radius
    rScale: scaling factor for radius
    d_Cos: pointer to pre-computed cosine values
    d_Sin: pointer to pre-computed sine values
Returns:
    None
 */
__global__ void GPU_HoughTranshared(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float* d_Cos, float* d_Sin) {
    // Calculate global ID
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    // Calculate the center of the image
    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    // Initialize local accumulator
    __shared__ int localAcc[degreeBins * rBins];
    for (int i = threadIdx.x; i < degreeBins * rBins; i += blockDim.x)
        localAcc[i] = 0;
    
    // Wait for all threads to finish
    __syncthreads();

    // Perform Hough Transform
    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            // Calculate the radius
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;

            // Increment the accumulator
            atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
        }
    }

    // Wait for all threads to finish
    __syncthreads();

    // Add local accumulator to global accumulator
    for (int i = threadIdx.x; i < degreeBins * rBins; i += blockDim.x)
        atomicAdd(&acc[i], localAcc[i]);
}

/* 
Function: main
Purpose: Entry point of the program
Parameters:
    argc: number of command line arguments
    argv: array of command line arguments
Returns:
    0 on success, 1 on failure
 */
int main (int argc, char **argv) {
    int i;
    std::string arg = argv[2];
    std::size_t pos;
    int threshold = std::stoi(arg,&pos);

    // Read input image
    PGMImage* inImg = new PGMImage(argv[1], 2);

    int *cpuht;

    // Get image dimensions
    int w = inImg->getXDim();
    int h = inImg->getYDim();

    float* d_Cos;
    float* d_Sin;

    // Create CUDA events for timing purposes
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory for pre-computed values
    cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
    cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

    // CPU calculation
    CPU_HoughTran(inImg->getPixels(), w, h, &cpuht);

    // Pre-compute values to be stored
    float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
    float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
    float rad = 0;

    // Calculate cosine and sine values
    for (i = 0; i < degreeBins; i++) {
        pcCos[i] = cos (rad);
        pcSin[i] = sin (rad);
        rad += radInc;
    }

    // Calculate maximum possible radius and scaling factor
    float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    // Allocate memory
    cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

    // GPU calculation
    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    // Get input image pixels
    h_in = inImg->getPixels();
    h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));
    
    // Allocate memory on device
    cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
    cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
    cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

    // Perform Hough Transform
    int blockNum = ceil (w * h / 256);
    cudaEventRecord(start);
    GPU_HoughTranshared <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

    // Get results
    cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    
    // Compare CPU and GPU results
    for (i = 0; i < degreeBins * rBins; i++) {
        if (cpuht[i] != h_hough[i]) {
            printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
        }
    }
    
    printf("Done!\n");
    
    // Save output image
    std::vector<std::pair<int, int>> lines;
    for (i = 0; i < degreeBins * rBins; i++){
        if (h_hough[i] > threshold) {
            int my_r = i / degreeBins;
            int my_th = i % degreeBins;
            std::pair<int, int> line = {my_r, my_th};
            lines.push_back(line);
        }
    }
    inImg->write("SharedOutput.jpeg", lines, radInc, rBins);

    // Print timing results
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Milliseconds: %.3f ms\n" ,milliseconds);
    printf("Seconds: %d.%.3d s\n", (int)milliseconds/1000, (int)milliseconds%1000);

    // Free memory
    cudaFree((void *) d_Cos);
    cudaFree((void *) d_Sin);
    cudaFree((void *) d_in);
    cudaFree((void *) d_hough);

    delete[] pcCos;
    delete[] pcSin;
    delete inImg;
    cudaDeviceReset();

    return 0;
}
