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

void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc) {
    float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
    
    *acc = new int[rBins * degreeBins];
    memset (*acc, 0, sizeof (int) * rBins * degreeBins);
    
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            int idx = j * w + i;
            
            if (pic[idx] > 0) {
                int xCoord = i - xCent;
                int yCoord = yCent - j;
                float theta = 0;
                
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
                    float r = xCoord * cos (theta) + yCoord * sin (theta); 
                    int rIdx = (r + rMax) / rScale;
                    (*acc)[rIdx * degreeBins + tIdx]++;
                    theta += radInc;
                }
            }
        }
    }
}

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

    int gloID = ( blockIdx.x ) * blockDim.x + threadIdx.x;
    if (gloID > w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;

            atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
}

int main (int argc, char **argv) {
    int i;
    std::string arg = argv[2]; 
    std::size_t pos; 
    int threshold = std::stoi(arg,&pos);

    PGMImage* inImg = new PGMImage(argv[1], 0);

    int *cpuht;
    int w = inImg->getXDim();
    int h = inImg->getYDim();

    float* d_Cos;
    float* d_Sin; 

    cudaEvent_t start, stop; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
    cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

    CPU_HoughTran(inImg->getPixels(), w, h, &cpuht); 

    float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
    float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
    float rad = 0;  
    for (i = 0; i < degreeBins; i++) {
        pcCos[i] = cos (rad);  
        pcSin[i] = sin (rad); 
        rad += radInc;
    }

    float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins; 

    cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = inImg->getPixels();
    h_hough = (int *) malloc (degreeBins * rBins * sizeof (int)); 

    cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h); 
    cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
    cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice); 
    cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins); 

    int blockNum = ceil (w * h / 256);

    cudaEventRecord(start);                                           

    GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
    cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    
    for (i = 0; i < degreeBins * rBins; i++) {
        if (cpuht[i] != h_hough[i]) {
            printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]); 
        }
    }

    printf("Done!\n");

    std::vector<std::pair<int, int>> lines;
    for (i = 0; i < degreeBins * rBins; i++) {  
        if (h_hough[i] > threshold) { 
            int my_r = i / degreeBins;
            int my_th = i % degreeBins;
            std::pair<int, int> line = {my_r, my_th}; 
            lines.push_back(line); 
        }
    }
    inImg->write("BaseOutput.jpeg", lines, radInc, rBins); 

    cudaEventSynchronize(stop);  
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Milliseconds: %.3f ms\n" ,milliseconds);
    printf("Seconds: %d.%.3d s\n", (int)milliseconds/1000, (int)milliseconds%1000);

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