#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>

#include "image.h"

#include <xmmintrin.h>
typedef __v4sf v4sf;

__global__
void test_parallel_kernel(image_t *du){
    printf("The size of du is %lu",sizeof(int));
}

//void someFunction(){
//    image_t *du = image_new(10,10);
//    test_parallel_kernel<<<1,1>>>(du);
//}

__global__ void cube(float * d_out, float * d_in){
	// Todo: Fill in this function
	int index = threadIdx.x;
	float f = d_in[index];
	d_out[index] = f * f * f;
}

int somefn(int argc, char ** argv) {
	const int ARRAY_SIZE = 96;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	
	// generate the input array on the host
	float h_in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_in[i] = float(i);
	}
	float h_out[ARRAY_SIZE];
	
	// declare GPU memory pointers
	float * d_in;
	float * d_out;
	
	// allocate GPU memory
	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);
	
	// transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	
	// launch the kernel
	cube<<<1, ARRAY_SIZE>>>(d_out, d_in);
	
	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	// print out the resulting array
	for (int i =0; i < ARRAY_SIZE; i++) {
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}
	
	cudaFree(d_in);
	cudaFree(d_out);
	
	    image_t *du = image_new(10,10);
//	    test_parallel_kernel<<<1,1>>>(du);
	
	return 0;
}