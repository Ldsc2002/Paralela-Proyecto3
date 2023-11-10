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
Purpose: Perform Hough Transform on the input image using CPU
Parameters:
    pic - input image
    w - width of the image
    h - height of the image
    acc - accumulator array
Returns:
    None
 */
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc) {
    // Calculate the maximum radius
    float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
    
    // Initialize the accumulator array
    *acc = new int[rBins * degreeBins];
    memset (*acc, 0, sizeof (int) * rBins * degreeBins);
    
    // Calculate the center of the image
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    // Perform Hough Transform
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            int idx = j * w + i;
            
            if (pic[idx] > 0) {
                // Calculate the coordinates of the pixel
                int xCoord = i - xCent;
                int yCoord = yCent - j;
                float theta = 0;
                
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
                    // Calculate the radius
                    float r = xCoord * cos (theta) + yCoord * sin (theta); 
                    int rIdx = (r + rMax) / rScale;

                    // Increment the accumulator array
                    (*acc)[rIdx * degreeBins + tIdx]++;
                    theta += radInc;
                }
            }
        }
    }
}

/* 
Function: GPU_HoughTran
Purpose: Perform Hough Transform on the input image using GPU
Parameters:
    pic - input image
    w - width of the image
    h - height of the image
    acc - accumulator array
    rMax - maximum radius
    rScale - scaling factor for radius
    d_Cos - cosine values
    d_Sin - sine values
Returns:
    None
 */
__global__ void GPU_HoughTran(
    unsigned char *pic, 
    int w, 
    int h, 
    int *acc, 
    float rMax, 
    float rScale, 
    float* d_Cos, 
    float* d_Sin
) {
    // Calculate the global ID
    int gloID = ( blockIdx.x ) * blockDim.x + threadIdx.x;
    if (gloID > w * h) return;

    // Calculate the center of the image
    int xCent = w / 2;
    int yCent = h / 2;

    // Calculate the coordinates of the pixel
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    // Perform Hough Transform
    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            // Calculate the radius
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;

            // Increment the accumulator array
            atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
}

/* 
Function: main
Purpose: Main function of the program
Parameters:
    argc - number of arguments
    argv - array of arguments
Returns:
    0 - success
 */
int main (int argc, char **argv) {
    // Initialize variables
    int i;
    std::string arg = argv[2]; 
    std::size_t pos; 
    int threshold = std::stoi(arg,&pos);

    // Read the input image
    PGMImage* inImg = new PGMImage(argv[1], 0);

    int *cpuht;

    // Get the width and height of the image
    int w = inImg->getXDim();
    int h = inImg->getYDim();

    float* d_Cos;
    float* d_Sin; 

    // Initialize CUDA events
    cudaEvent_t start, stop; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory on the GPU
    cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
    cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

    // Perform Hough Transform on the CPU
    CPU_HoughTran(inImg->getPixels(), w, h, &cpuht); 

    // Allocate memory on the CPU
    float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
    float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
    float rad = 0;  
    
    // Calculate the cosine and sine values
    for (i = 0; i < degreeBins; i++) {
        pcCos[i] = cos (rad);  
        pcSin[i] = sin (rad); 
        rad += radInc;
    }

    // Calculate the maximum radius and scaling factor
    float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins; 

    // Copy the cosine and sine values to the GPU
    cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    // Allocate memory on the CPU
    h_in = inImg->getPixels();
    h_hough = (int *) malloc (degreeBins * rBins * sizeof (int)); 

    // Copy the input image to the GPU
    cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h); 
    cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
    cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice); 
    cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins); 

    int blockNum = ceil (w * h / 256);

    // Start the timer
    cudaEventRecord(start);                                           

    // Perform Hough Transform on the GPU
    GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
    cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // Stop the timer
    cudaEventRecord(stop);
    
    // Check for calculation mismatches
    for (i = 0; i < degreeBins * rBins; i++) {
        if (cpuht[i] != h_hough[i]) {
            printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]); 
        }
    }

    printf("Done!\n");

    // Write the output image 
    std::vector<std::pair<int, int>> lines;
    for (i = 0; i < degreeBins * rBins; i++) {  
        if (h_hough[i] > threshold) { 
            int my_r = i / degreeBins;
            int my_th = i % degreeBins;
            std::pair<int, int> line = {my_r, my_th}; 
            lines.push_back(line); 
        }
    }

    // Write the output image to a file
    inImg->write("BaseOutput.jpeg", lines, radInc, rBins); 

    // Print the time taken
    cudaEventSynchronize(stop);  
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Milliseconds: %.3f ms\n" ,milliseconds);
    printf("Seconds: %d.%.3d s\n", (int)milliseconds/1000, (int)milliseconds%1000);

    // Free the memory on the GPU
    cudaFree((void *) d_Cos); 
    cudaFree((void *) d_Sin);
    cudaFree((void *) d_in); 
    cudaFree((void *) d_hough); 
    
    // Free the memory on the CPU
    delete[] pcCos;
    delete[] pcSin;  
    delete inImg;  
    
    // Reset the GPU
    cudaDeviceReset();

    return 0;
}