#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <malloc.h>

#include "variational.h"
#include "variational_aux.h"
#include "solver.h"


#include <xmmintrin.h>
typedef __v4sf v4sf;

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.h"

typedef __v4sf v4sf;

void parallel_sor(float *d_du, float *d_dv, float *d_a11, float *d_a12, float *d_a22, float *d_b1, float *d_b2, float *d_dpsis_horiz, float *d_dpsis_vert, int width, int height, int stride, const int iterations, const float omega);

void sor_coupled_slow_but_readable(image_t *du, image_t *dv, image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *dpsis_horiz, image_t *dpsis_vert, const int iterations, const float omega);

void sor_coupled_cuda(image_t *du, image_t *dv, image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *dpsis_horiz, image_t *dpsis_vert, const int iterations, const float omega);

void checkCudaMemoryErrors(cudaError_t status){
	if(status != cudaSuccess){
		printf("%s",cudaGetErrorString(status));
	}
}

__global__
void red_sor_reordered(float *du_red, float *dv_red, float *dpsis_horiz_red, float *dpsis_vert_red, float *d_a11_red, float *d_a12_red, float *d_a22_red, float *d_b1_red, float *d_b2_red, float *du_black, float *dv_black, float *dpsis_horiz_black, float *dpsis_vert_black,float *d_a11_black, float *d_a12_black, float *d_a22_black, float *d_b1_black, float *d_b2_black, float omega, int width, int height, int stride){
	
	int numRows = height;
	int numCols = width;
	
	float sigma_u,sigma_v,sum_dpsis,A11,A22,A12,B1,B2,det;
	
	int block_linear_index = blockIdx.y * gridDim.x + blockIdx.x;
	int thread_linear_index = threadIdx.y * blockDim.x + threadIdx.x;
	int total_number_of_threads_per_block = blockDim.x * blockDim.y;
	
	int element_linear_index = (block_linear_index * total_number_of_threads_per_block) + (thread_linear_index);
	
	element_linear_index = (2 * element_linear_index) + ((2 * element_linear_index/numCols) % 2);
	
	if(element_linear_index < 0 || element_linear_index >= numRows * numCols){
		return;
	}

	
	int i = element_linear_index % numCols;
	int j = (element_linear_index - i) / numCols;
	
	float dpsis_horiz_left;
	float dpsis_horiz_center;
	
	float dpsis_vert_top;
	float dpsis_vert_center;
	
//	if ((i+j) % 2 == 0) {
		sigma_u = 0.0f;
		sigma_v = 0.0f;
		sum_dpsis = 0.0f;
		
		//If not the first row
		if(j>0){
			//Top neighbor
			dpsis_vert_top = dpsis_vert_black[((j-1)*stride/2) + i/2];
			sigma_u -= dpsis_vert_top * du_black[((j-1)*stride/2) + i/2];
			sigma_v -= dpsis_vert_top * dv_black[((j-1)*stride/2) + i/2];
			sum_dpsis += dpsis_vert_top;
		}
		
		//If not first column
		if(i>0){
			if(j % 2 == 0 ){
				//RED NODES: If even row, left neighbor is left neighbor
				dpsis_horiz_left = dpsis_horiz_black[(j*stride/2) + (i/2)-1];
				sigma_u -= dpsis_horiz_left * du_black[((j)*stride/2) + (i/2)-1];
				sigma_v -= dpsis_horiz_left * dv_black[((j)*stride/2) + (i/2)-1];
				sum_dpsis += dpsis_horiz_left;
			}
			else{
				//RED NODES: If odd row, left neighbor is center neighbor
				dpsis_horiz_left = dpsis_horiz_black[(j*stride/2)+(i/2)];
				sigma_u -= dpsis_horiz_left * du_black[((j)*stride/2) + (i/2)];
				sigma_v -= dpsis_horiz_left * dv_black[((j)*stride/2) + (i/2)];
				sum_dpsis += dpsis_horiz_left;
			}
		}
		
		//If not last row
		if(j<height - 1){
			//Bottom neighbor
			dpsis_vert_center = dpsis_vert_red[((j)*stride/2) + (i/2)];
			sigma_u -= dpsis_vert_center * du_black[((j+1)*stride/2) + i/2];
			sigma_v -= dpsis_vert_center * dv_black[((j+1)*stride/2) + i/2];
			sum_dpsis += dpsis_vert_center;
		}
		
		//If not last column
		if(i<width-1){
			if(j % 2 == 0){
				//RED NODES: If even row, right neighbor is center neighbor
				dpsis_horiz_center = dpsis_horiz_red[(j*stride/2) + (i/2)];
				sigma_u -= dpsis_horiz_center * du_black[((j)*stride/2) + (i/2)];
				sigma_v -= dpsis_horiz_center * dv_black[((j)*stride/2) + (i/2)];
				sum_dpsis += dpsis_horiz_center;
			}
			else{
				//RED NODES: If even row, right neighbor is right neighbor
				dpsis_horiz_center = dpsis_horiz_red[(j*stride/2) + (i/2)];
				sigma_u -= dpsis_horiz_center * du_black[((j)*stride/2) + (i/2) + 1];
				sigma_v -= dpsis_horiz_center * dv_black[((j)*stride/2) + (i/2) + 1];
				sum_dpsis += dpsis_horiz_center;
			}
		}
		
		A11 = d_a11_red[(j*stride/2) + (i/2)] + sum_dpsis;
		A12 = d_a12_red[(j*stride/2) + (i/2)];
		A22 = d_a22_red[(j*stride/2) + (i/2)] + sum_dpsis;
		
		det = A11*A22-A12*A12;
		
		B1 = d_b1_red[(j*stride/2) + (i/2)] - sigma_u;
		B2 = d_b2_red[(j*stride/2) + (i/2)] - sigma_v;
		
		du_red[(j*stride/2) + (i/2)] = (1.0f-omega) * du_red[(j*stride/2) + (i/2)] + omega*( A22*B1-A12*B2)/det;
		dv_red[(j*stride/2) + (i/2)] = (1.0f-omega) * dv_red[(j*stride/2) + (i/2)] + omega*(-A12*B1+A11*B2)/det;
//	}
}

__global__
void black_sor_reordered(float *du_red, float *dv_red, float *dpsis_horiz_red, float *dpsis_vert_red, float *d_a11_red, float *d_a12_red, float *d_a22_red, float *d_b1_red, float *d_b2_red, float *du_black, float *dv_black, float *dpsis_horiz_black, float *dpsis_vert_black,float *d_a11_black, float *d_a12_black, float *d_a22_black, float *d_b1_black, float *d_b2_black, float omega, int width, int height, int stride){
	int numRows = height;
	int numCols = width;
	
	float sigma_u,sigma_v,sum_dpsis,A11,A22,A12,B1,B2,det;
	
	int block_linear_index = blockIdx.y * gridDim.x + blockIdx.x;
	int thread_linear_index = threadIdx.y * blockDim.x + threadIdx.x;
	int total_number_of_threads_per_block = blockDim.x * blockDim.y;
	
	int element_linear_index = (block_linear_index * total_number_of_threads_per_block) + (thread_linear_index);
	
	element_linear_index = ((2*element_linear_index)+1) - ((2 * element_linear_index/numCols) % 2);
	
	if(element_linear_index < 0 || element_linear_index >= numRows * numCols){
		return;
	}
	
	int i = element_linear_index % numCols;
	int j = (element_linear_index - i) / numCols;

	float dpsis_horiz_left;
	float dpsis_horiz_center;
	
	float dpsis_vert_top;
	float dpsis_vert_center;
	
//	if ((i+j) % 2 != 0) {
		sigma_u = 0.0f;
		sigma_v = 0.0f;
		sum_dpsis = 0.0f;
		
		//If not the first row
		if(j>0){
			//Top neighbor
			dpsis_vert_top = dpsis_vert_red[((j-1)*stride/2) + i/2];
			sigma_u -= dpsis_vert_top * du_red[((j-1)*stride/2) + i/2];
			sigma_v -= dpsis_vert_top * dv_red[((j-1)*stride/2) + i/2];
			sum_dpsis += dpsis_vert_top;
		}
		
		//If not first column
		if(i>0){
			if(j % 2 == 0 ){
				//BLACK NODES: If even row, left neighbor is center neighbor
				dpsis_horiz_left = dpsis_horiz_red[(j*stride/2) + (i/2)];
				sigma_u -= dpsis_horiz_left * du_red[((j)*stride/2) + (i/2)];
				sigma_v -= dpsis_horiz_left * dv_red[((j)*stride/2) + (i/2)];
				sum_dpsis += dpsis_horiz_left;
			}
			else{
				//BLACK NODES: If odd row, left neighbor is left neighbor
				dpsis_horiz_left = dpsis_horiz_red[(j*stride/2)+(i/2) - 1];
				sigma_u -= dpsis_horiz_left * du_red[((j)*stride/2) + (i/2) - 1];
				sigma_v -= dpsis_horiz_left * dv_red[((j)*stride/2) + (i/2) - 1];
				sum_dpsis += dpsis_horiz_left;
			}
		}
		
		//If not last row
		if(j<height - 1){
			//Bottom neighbor
			dpsis_vert_center = dpsis_vert_black[((j)*stride/2) + (i/2)];
			sigma_u -= dpsis_vert_center * du_red[((j+1)*stride/2) + i/2];
			sigma_v -= dpsis_vert_center * dv_red[((j+1)*stride/2) + i/2];
			sum_dpsis += dpsis_vert_center;
		}
		
		//If not last column
		if(i<width-1){
			if(j % 2 == 0){
				//BLACK NODES: If even row, right neighbor is right neighbor
				dpsis_horiz_center = dpsis_horiz_black[(j*stride/2) + (i/2)];
				sigma_u -= dpsis_horiz_center * du_red[((j)*stride/2) + (i/2) + 1];
				sigma_v -= dpsis_horiz_center * dv_red[((j)*stride/2) + (i/2) + 1];
				sum_dpsis += dpsis_horiz_center;
			}
			else{
				//BLACK NODES: If even row, right neighbor is center neighbor
				dpsis_horiz_center = dpsis_horiz_black[(j*stride/2) + (i/2)];
				sigma_u -= dpsis_horiz_center * du_red[((j)*stride/2) + (i/2)];
				sigma_v -= dpsis_horiz_center * dv_red[((j)*stride/2) + (i/2)];
				sum_dpsis += dpsis_horiz_center;
			}
		}
		
		A11 = d_a11_black[(j*stride/2) + (i/2)] + sum_dpsis;
		A12 = d_a12_black[(j*stride/2) + (i/2)];
		A22 = d_a22_black[(j*stride/2) + (i/2)] + sum_dpsis;
		
		det = A11*A22-A12*A12;
		
		B1 = d_b1_black[(j*stride/2) + (i/2)] - sigma_u;
		B2 = d_b2_black[(j*stride/2) + (i/2)] - sigma_v;
		
		du_black[(j*stride/2) + (i/2)] = (1.0f-omega) * du_black[(j*stride/2) + (i/2)] + omega*( A22*B1-A12*B2)/det;
		dv_black[(j*stride/2) + (i/2)] = (1.0f-omega) * dv_black[(j*stride/2) + (i/2)] + omega*(-A12*B1+A11*B2)/det;

//	}
}


__global__
void reorder_split(float *d_du, float *d_dv, float *d_a11, float *d_a12, float *d_a22, float *d_b1, float *d_b2, float *d_dpsis_horiz, float *d_dpsis_vert,
		   float *d_du_red, float *d_dv_red, float *d_dpsis_horiz_red, float *d_dpsis_vert_red, float*d_a11_red, float*d_a12_red, float*d_a22_red, float*d_b1_red, float*d_b2_red,
		   float *d_du_black, float *d_dv_black, float *d_dpsis_horiz_black, float *d_dpsis_vert_black, float*d_a11_black, float*d_a12_black, float*d_a22_black, float*d_b1_black, float*d_b2_black,
		   int width, int height, int stride){
	
	int numRows = height;
	int numCols = width;
	
	int block_linear_index = blockIdx.y * gridDim.x + blockIdx.x;
	int thread_linear_index = threadIdx.y * blockDim.x + threadIdx.x;
	int total_number_of_threads_per_block = blockDim.x * blockDim.y;
	
	int element_linear_index = (block_linear_index * total_number_of_threads_per_block) + (thread_linear_index);
	
	if(element_linear_index < 0 || element_linear_index >= numRows * numCols){
		return;
	}
	
	int i = element_linear_index % numCols;
	int j = (element_linear_index - i) / numCols;
	
	int new_i = i / 2;
	
	if((i + j) % 2 == 0){
		d_du_red[j*(stride/2) + new_i] = d_du[j*stride + i];
		d_dv_red[j*(stride/2) + new_i] = d_dv[j*stride + i];
		d_a11_red[j*(stride/2) + new_i] = d_a11[j*stride + i];
		d_a12_red[j*(stride/2) + new_i] = d_a12[j*stride + i];
		d_a22_red[j*(stride/2) + new_i] = d_a22[j*stride + i];
		d_b1_red[j*(stride/2) + new_i] = d_b1[j*stride + i];
		d_b2_red[j*(stride/2) + new_i] = d_b2[j*stride + i];
		d_dpsis_vert_red[j*(stride/2) + new_i] = d_dpsis_vert[j*stride + i];
		d_dpsis_horiz_red[j*(stride/2) + new_i] = d_dpsis_horiz[j*stride + i];
	}
	else{
		d_du_black[j*(stride/2) + new_i] = d_du[j*stride + i];
		d_dv_black[j*(stride/2) + new_i] = d_dv[j*stride + i];
		d_a11_black[j*(stride/2) + new_i] = d_a11[j*stride + i];
		d_a12_black[j*(stride/2) + new_i] = d_a12[j*stride + i];
		d_a22_black[j*(stride/2) + new_i] = d_a22[j*stride + i];
		d_b1_black[j*(stride/2) + new_i] = d_b1[j*stride + i];
		d_b2_black[j*(stride/2) + new_i] = d_b2[j*stride + i];
		d_dpsis_vert_black[j*(stride/2) + new_i] = d_dpsis_vert[j*stride + i];
		d_dpsis_horiz_black[j*(stride/2) + new_i] = d_dpsis_horiz[j*stride + i];
	}
	__syncthreads();
}

__global__
void reorder_combine(float *d_du, float *d_dv, float *d_a11, float *d_a12, float *d_a22, float *d_b1, float *d_b2, float *d_dpsis_horiz, float *d_dpsis_vert,
					 float *d_du_red, float *d_dv_red, float *d_dpsis_horiz_red, float *d_dpsis_vert_red, float*d_a11_red, float*d_a12_red, float*d_a22_red, float*d_b1_red, float*d_b2_red,
					 float *d_du_black, float *d_dv_black, float *d_dpsis_horiz_black, float *d_dpsis_vert_black, float*d_a11_black, float*d_a12_black, float*d_a22_black, float*d_b1_black, float*d_b2_black,
					 int width, int height, int stride){
	
	int numRows = height;
	int numCols = width;
	
	int block_linear_index = blockIdx.y * gridDim.x + blockIdx.x;
	int thread_linear_index = threadIdx.y * blockDim.x + threadIdx.x;
	int total_number_of_threads_per_block = blockDim.x * blockDim.y;
	
	int element_linear_index = (block_linear_index * total_number_of_threads_per_block) + (thread_linear_index);
	
	if(element_linear_index < 0 || element_linear_index >= numRows * numCols){
		return;
	}
	
	int i = element_linear_index % numCols;
	int j = (element_linear_index - i) / numCols;
	
	int new_i = i/2;
	
	if((i+j)%2 == 0){
		d_du[j*stride + i] = d_du_red[(j*stride/2) + new_i];
		d_dv[j*stride + i] = d_dv_red[(j*stride/2) + new_i];
		d_dpsis_vert[j*stride + i] = d_dpsis_vert_red[(j*stride/2) + new_i];
		d_dpsis_horiz[j*stride + i] = d_dpsis_horiz_red[(j*stride/2) + new_i];
	}
	else{
		d_du[j*stride + i] = d_du_black[(j*stride/2) + new_i];
		d_dv[j*stride + i] = d_dv_black[(j*stride/2) + new_i];
		d_dpsis_vert[j*stride + i] = d_dpsis_vert_black[(j*stride/2) + new_i];
		d_dpsis_horiz[j*stride + i] = d_dpsis_horiz_black[(j*stride/2) + new_i];
	}
}

image_t *image_new_cuda(const int width, const int height){
	image_t *image = (image_t*) malloc(sizeof(image_t));
	if(image == NULL){
		fprintf(stderr, "Error: image_new() - not enough memory !\n");
		exit(1);
	}
	image->width = width;
	image->height = height;
	image->stride = ( (width+3) / 4 ) * 4;
	cudaMallocHost(&image->data, ((image->stride*height*sizeof(float))));
//	image->data = (float *)malloc(image->stride*height*sizeof(float));
	if(image->data == NULL){
		fprintf(stderr, "Error: image_new_cuda() - not enough memory !\n");
		exit(1);
	}
	return image;
}

/* free memory of an image */
void image_delete_cuda(image_t *image){
	if(image == NULL){
		//fprintf(stderr, "Warning: Delete image --> Ignore action (image not allocated)\n");
	}else{
		cudaFreeHost(image->data);
		free(image);
	}
}

convolution_t *deriv, *deriv_flow;

float half_alpha, half_delta_over3, half_gamma_over3;


/* perform flow computation at one level of the pyramid */
void compute_one_level(image_t *wx, image_t *wy, color_image_t *im1, color_image_t *im2, const variational_params_t *params){
	
    const int width = wx->width, height = wx->height, stride=wx->stride;

    image_t *du = image_new_cuda(width,height), *dv = image_new_cuda(width,height), // the flow increment
        *mask = image_new(width,height), // mask containing 0 if a point goes outside image boundary, 1 otherwise
        *smooth_horiz = image_new_cuda(width,height), *smooth_vert = image_new_cuda(width,height), // horiz: (i,j) contains the diffusivity coeff from (i,j) to (i+1,j)
        *uu = image_new(width,height), *vv = image_new(width,height), // flow plus flow increment
        *a11 = image_new_cuda(width,height), *a12 = image_new_cuda(width,height), *a22 = image_new_cuda(width,height), // system matrix A of Ax=b for each pixel
        *b1 = image_new_cuda(width,height), *b2 = image_new_cuda(width,height); // system matrix b of Ax=b for each pixel

    color_image_t *w_im2 = color_image_new(width,height), // warped second image
        *Ix = color_image_new(width,height), *Iy = color_image_new(width,height), *Iz = color_image_new(width,height), // first order derivatives
        *Ixx = color_image_new(width,height), *Ixy = color_image_new(width,height), *Iyy = color_image_new(width,height), *Ixz = color_image_new(width,height), *Iyz = color_image_new(width,height); // second order derivatives
  
  
    image_t *dpsis_weight = compute_dpsis_weight(im1, 5.0f, deriv);
	
	float *d_du; float *d_dv;
	float *d_a11; float *d_a12; float *d_a22;
	float *d_b1; float *d_b2;
	float *d_dpsis_horiz; float *d_dpsis_vert;
	
	float *d_du_red; float *d_dv_red;
	float *d_a11_red; float *d_a12_red; float *d_a22_red;
	float *d_b1_red; float *d_b2_red;
	float *d_dpsis_horiz_red; float *d_dpsis_vert_red;
	
	float *d_du_black; float *d_dv_black;
	float *d_a11_black; float *d_a12_black; float *d_a22_black;
	float *d_b1_black; float *d_b2_black;
	float *d_dpsis_horiz_black; float *d_dpsis_vert_black;
	
	int data_size = (du->stride*du->height*sizeof(float));
	
	if(params->use_gpu){
		
		checkCudaMemoryErrors(cudaMalloc(&d_du,data_size));
		checkCudaMemoryErrors(cudaMalloc(&d_dv,data_size));
		checkCudaMemoryErrors(cudaMalloc(&d_a11,data_size));
		checkCudaMemoryErrors(cudaMalloc(&d_a22,data_size));
		checkCudaMemoryErrors(cudaMalloc(&d_a12,data_size));
		checkCudaMemoryErrors(cudaMalloc(&d_b1,data_size));
		checkCudaMemoryErrors(cudaMalloc(&d_b2,data_size));
		checkCudaMemoryErrors(cudaMalloc(&d_dpsis_horiz,data_size));
		checkCudaMemoryErrors(cudaMalloc(&d_dpsis_vert,data_size));
		
		checkCudaMemoryErrors(cudaMalloc(&d_du_red,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_dv_red,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_a11_red,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_a22_red,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_a12_red,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_b1_red,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_b2_red,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_dpsis_horiz_red,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_dpsis_vert_red,data_size/2));
		
		checkCudaMemoryErrors(cudaMalloc(&d_du_black,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_dv_black,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_a11_black,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_a22_black,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_a12_black,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_b1_black,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_b2_black,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_dpsis_horiz_black,data_size/2));
		checkCudaMemoryErrors(cudaMalloc(&d_dpsis_vert_black,data_size/2));
		
	}
	
    int i_outer_iteration;
    //Default case, this is done for 5 outer iterations and 30 solver iterations
    for(i_outer_iteration = 0 ; i_outer_iteration < params->niter_outer ; i_outer_iteration++){
        int i_inner_iteration;
        // warp second image
        image_warp(w_im2, mask, im2, wx, wy);
        // compute derivatives
        get_derivatives(im1, w_im2, deriv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz);
        // erase du and dv
        image_erase(du);
        image_erase(dv);
        // initialize uu and vv
        memcpy(uu->data,wx->data,wx->stride*wx->height*sizeof(float));
        memcpy(vv->data,wy->data,wy->stride*wy->height*sizeof(float));
        // inner fixed point iterations
        for(i_inner_iteration = 0 ; i_inner_iteration < params->niter_inner ; i_inner_iteration++){
            //  compute robust function and system
            compute_smoothness(smooth_horiz, smooth_vert, uu, vv, dpsis_weight, deriv_flow, half_alpha );
            compute_data_and_match(a11, a12, a22, b1, b2, mask, du, dv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, half_delta_over3, half_gamma_over3);
            sub_laplacian(b1, wx, smooth_horiz, smooth_vert);
            sub_laplacian(b2, wy, smooth_horiz, smooth_vert);
			
			// solve system
			if(params->skip_epic){
				
			}
			else if (params->use_gpu) {
				
				checkCudaMemoryErrors(cudaMemcpy(d_du,du->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpy(d_dv,dv->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpy(d_a11,a11->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpy(d_a12,a12->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpy(d_a22,a22->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpy(d_b1,b1->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpy(d_b2,b2->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpy(d_dpsis_horiz,smooth_horiz->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpy(d_dpsis_vert,smooth_vert->data,data_size,cudaMemcpyHostToDevice));
				
				int threadCountX = params->threadX;
				int threadCountY = params->threadY;
				const dim3 blockSize(threadCountX,threadCountY,1);
				int gridSizeX = (width % blockSize.x == 0) ? width/blockSize.x : (width/blockSize.x) + 1;
				int gridSizeY = (height % blockSize.y == 0) ? height/blockSize.y : (height/blockSize.y) + 1;
				const dim3 gridSize(gridSizeX,gridSizeY,1);
				
				reorder_split<<<gridSize,blockSize>>>(d_du, d_dv, d_a11, d_a12, d_a22, d_b1, d_b2, d_dpsis_horiz, d_dpsis_vert, d_du_red, d_dv_red, d_dpsis_horiz_red, d_dpsis_vert_red, d_a11_red, d_a12_red, d_a22_red, d_b1_red, d_b2_red, d_du_black, d_dv_black, d_dpsis_horiz_black, d_dpsis_vert_black, d_a11_black, d_a12_black, d_a22_black, d_b1_black, d_b2_black, width, height, stride);
				
				for(int iter = 0 ; iter<params->niter_solver ; iter++){
					
					red_sor_reordered<<<gridSize,blockSize>>>(d_du_red, d_dv_red, d_dpsis_horiz_red, d_dpsis_vert_red, d_a11_red, d_a12_red, d_a22_red, d_b1_red, d_b2_red, d_du_black, d_dv_black, d_dpsis_horiz_black, d_dpsis_vert_black, d_a11_black, d_a12_black, d_a22_black, d_b1_black, d_b2_black, params->sor_omega, width, height, stride);
			
					black_sor_reordered<<<gridSize,blockSize>>>(d_du_red, d_dv_red, d_dpsis_horiz_red, d_dpsis_vert_red, d_a11_red, d_a12_red, d_a22_red, d_b1_red, d_b2_red, d_du_black, d_dv_black, d_dpsis_horiz_black, d_dpsis_vert_black, d_a11_black, d_a12_black, d_a22_black, d_b1_black, d_b2_black, params->sor_omega, width, height, stride);

				}
				
				reorder_combine<<<gridSize,blockSize>>>(d_du, d_dv, d_a11, d_a12, d_a22, d_b1, d_b2, d_dpsis_horiz, d_dpsis_vert, d_du_red, d_dv_red, d_dpsis_horiz_red, d_dpsis_vert_red, d_a11_red, d_a12_red, d_a22_red, d_b1_red, d_b2_red, d_du_black, d_dv_black, d_dpsis_horiz_black, d_dpsis_vert_black, d_a11_black, d_a12_black, d_a22_black, d_b1_black, d_b2_black, width, height, stride);
			}
			else{
				sor_coupled_slow_but_readable(du, dv, a11, a12, a22, b1, b2, smooth_horiz, smooth_vert, params->niter_solver, params->sor_omega);
			}
			
			
			checkCudaMemoryErrors(cudaDeviceSynchronize());
			
			// copy flow from GPU to CPU
			if (params->use_gpu) {
				checkCudaMemoryErrors(cudaMemcpy(du->data,d_du,data_size,cudaMemcpyDeviceToHost));
				checkCudaMemoryErrors(cudaMemcpy(dv->data,d_dv,data_size,cudaMemcpyDeviceToHost));
			}
			
			checkCudaMemoryErrors(cudaDeviceSynchronize());
			
			// update flow plus flow increment
            int i;
            v4sf *uup = (v4sf*) uu->data, *vvp = (v4sf*) vv->data, *wxp = (v4sf*) wx->data, *wyp = (v4sf*) wy->data, *dup = (v4sf*) du->data, *dvp = (v4sf*) dv->data;
            for( i=0 ; i<height*stride/4 ; i++){
                (*uup) = (*wxp) + (*dup);
                (*vvp) = (*wyp) + (*dvp);
                uup+=1; vvp+=1; wxp+=1; wyp+=1;dup+=1;dvp+=1;
	        }
        }
        // add flow increment to current flow
        memcpy(wx->data,uu->data,uu->stride*uu->height*sizeof(float));
        memcpy(wy->data,vv->data,vv->stride*vv->height*sizeof(float));
    }   
    // free memory
    image_delete_cuda(du); image_delete_cuda(dv);
    image_delete(mask);
    image_delete_cuda(smooth_horiz); image_delete_cuda(smooth_vert);
    image_delete(uu); image_delete(vv);
    image_delete_cuda(a11); image_delete_cuda(a12); image_delete_cuda(a22);
    image_delete_cuda(b1); image_delete_cuda(b2);
    image_delete(dpsis_weight);
    color_image_delete(w_im2); 
    color_image_delete(Ix); color_image_delete(Iy); color_image_delete(Iz);
    color_image_delete(Ixx); color_image_delete(Ixy); color_image_delete(Iyy); color_image_delete(Ixz); color_image_delete(Iyz);
	
	if(params->use_gpu){
		checkCudaMemoryErrors(cudaFree(d_du));
		checkCudaMemoryErrors(cudaFree(d_dv));
		checkCudaMemoryErrors(cudaFree(d_a11));
		checkCudaMemoryErrors(cudaFree(d_a22));
		checkCudaMemoryErrors(cudaFree(d_a12));
		checkCudaMemoryErrors(cudaFree(d_b1));
		checkCudaMemoryErrors(cudaFree(d_b2));
		checkCudaMemoryErrors(cudaFree(d_dpsis_horiz));
		checkCudaMemoryErrors(cudaFree(d_dpsis_vert));
		
		checkCudaMemoryErrors(cudaFree(d_du_red));
		checkCudaMemoryErrors(cudaFree(d_dv_red));
		checkCudaMemoryErrors(cudaFree(d_a11_red));
		checkCudaMemoryErrors(cudaFree(d_a22_red));
		checkCudaMemoryErrors(cudaFree(d_a12_red));
		checkCudaMemoryErrors(cudaFree(d_b1_red));
		checkCudaMemoryErrors(cudaFree(d_b2_red));
		checkCudaMemoryErrors(cudaFree(d_dpsis_horiz_red));
		checkCudaMemoryErrors(cudaFree(d_dpsis_vert_red));
		
		checkCudaMemoryErrors(cudaFree(d_du_black));
		checkCudaMemoryErrors(cudaFree(d_dv_black));
		checkCudaMemoryErrors(cudaFree(d_a11_black));
		checkCudaMemoryErrors(cudaFree(d_a22_black));
		checkCudaMemoryErrors(cudaFree(d_a12_black));
		checkCudaMemoryErrors(cudaFree(d_b1_black));
		checkCudaMemoryErrors(cudaFree(d_b2_black));
		checkCudaMemoryErrors(cudaFree(d_dpsis_horiz_black));
		checkCudaMemoryErrors(cudaFree(d_dpsis_vert_black));
		
//		checkCudaMemoryErrors(cudaFreeArray(cudaArray_du));
//		checkCudaMemoryErrors(cudaFreeArray(cudaArray_dv));
//		checkCudaMemoryErrors(cudaFreeArray(cudaArray_a11));
//		checkCudaMemoryErrors(cudaFreeArray(cudaArray_a12));
//		checkCudaMemoryErrors(cudaFreeArray(cudaArray_a22));
//		checkCudaMemoryErrors(cudaFreeArray(cudaArray_b1));
//		checkCudaMemoryErrors(cudaFreeArray(cudaArray_b2));
//		checkCudaMemoryErrors(cudaFreeArray(cudaArray_dpsis_horiz));
//		checkCudaMemoryErrors(cudaFreeArray(cudaArray_dpsis_vert));
	}
}

/* set flow parameters to default */
void variational_params_default(variational_params_t *params){
    if(!params){
        fprintf(stderr,"Error optical_flow_params_default: argument is null\n");
        exit(1);
    }
    params->alpha = 1.0f;
    params->gamma = 0.71f;
    params->delta = 0.0f;
    params->sigma = 1.00f;
    params->niter_outer = 5;
    params->niter_inner = 1;  
    params->niter_solver = 30;
    params->sor_omega = 1.9f;
	params->use_gpu = false;
	params->skip_epic = false;
}
  
/* Compute a refinement of the optical flow (wx and wy are modified) between im1 and im2 */
void variational(image_t *wx, image_t *wy, const color_image_t *im1, const color_image_t *im2, variational_params_t *params){
  
    // Check parameters
    if(!params){
        params = (variational_params_t*) malloc(sizeof(variational_params_t));
        if(!params){
          fprintf(stderr,"error: not enough memory\n");
          exit(1);
        }
        variational_params_default(params);
    }

    // initialize global variables
    half_alpha = 0.5f*params->alpha;
    half_gamma_over3 = params->gamma*0.5f/3.0f;
    half_delta_over3 = params->delta*0.5f/3.0f;
    
    float deriv_filter[3] = {0.0f, -8.0f/12.0f, 1.0f/12.0f};
    deriv = convolution_new(2, deriv_filter, 0);
    float deriv_filter_flow[2] = {0.0f, -0.5f};
    deriv_flow = convolution_new(1, deriv_filter_flow, 0);


    // presmooth images
    int width = im1->width, height = im1->height, filter_size;
    color_image_t *smooth_im1 = color_image_new(width, height), *smooth_im2 = color_image_new(width, height);
    float *presmooth_filter = gaussian_filter(params->sigma, &filter_size);
    convolution_t *presmoothing = convolution_new(filter_size, presmooth_filter, 1);
    color_image_convolve_hv(smooth_im1, im1, presmoothing, presmoothing);
    color_image_convolve_hv(smooth_im2, im2, presmoothing, presmoothing); 
    convolution_delete(presmoothing);
    free(presmooth_filter);
    
    compute_one_level(wx, wy, smooth_im1, smooth_im2, params);
  
    // free memory
    color_image_delete(smooth_im1);
    color_image_delete(smooth_im2);
    convolution_delete(deriv);
    convolution_delete(deriv_flow);
}

