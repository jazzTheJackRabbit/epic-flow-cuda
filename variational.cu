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

void test_rb_sor(float *d_du, float *d_dv, float *d_a11, float *d_a12, float *d_a22, float *d_b1, float *d_b2, float *d_dpsis_horiz, float *d_dpsis_vert, int width, int height, int stride, const int iterations, const float omega);

void test_rb_sor_black(float *du, float *dv, float *a11, float *a12, float *a22, float *b1, float *b2, float *dpsis_horiz, float *dpsis_vert, int width, int height, int stride, const int iterations, const float omega);

void checkCudaMemoryErrors(cudaError_t status){
	if(status != cudaSuccess){
		printf("%s",cudaGetErrorString(status));
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
	cudaMallocHost(&image->data, (16*(image->stride*height*sizeof(float))));
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
//
	float *d_du; float *d_dv;
	float *d_a11; float *d_a12; float *d_a22;
	float *d_b1; float *d_b2;
	float *d_dpsis_horiz; float *d_dpsis_vert;
	
	int data_size = 16*(du->stride*du->height*sizeof(float));
	
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
				checkCudaMemoryErrors(cudaMemcpyAsync(d_du,du->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpyAsync(d_dv,dv->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpyAsync(d_a11,a11->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpyAsync(d_a12,a12->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpyAsync(d_a22,a22->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpyAsync(d_b1,b1->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpyAsync(d_b2,b2->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpyAsync(d_dpsis_horiz,smooth_horiz->data,data_size,cudaMemcpyHostToDevice));
				checkCudaMemoryErrors(cudaMemcpyAsync(d_dpsis_vert,smooth_vert->data,data_size,cudaMemcpyHostToDevice));
				
				parallel_sor(d_du, d_dv, d_a11, d_a12, d_a22, d_b1, d_b2, d_dpsis_horiz, d_dpsis_vert, du->width, du->height, du->stride, params->niter_solver, params->sor_omega);
			}
			else{
//				sor_coupled_cuda(du, dv, a11, a12, a22, b1, b2, smooth_horiz, smooth_vert, params->niter_solver, params->sor_omega);
				sor_coupled_slow_but_readable(du, dv, a11, a12, a22, b1, b2, smooth_horiz, smooth_vert, params->niter_solver, params->sor_omega);
			}
			
			// copy flow from GPU to CPU
			if (params->use_gpu) {
				checkCudaMemoryErrors(cudaMemcpyAsync(du->data,d_du,data_size,cudaMemcpyDeviceToHost));
				checkCudaMemoryErrors(cudaMemcpyAsync(dv->data,d_dv,data_size,cudaMemcpyDeviceToHost));
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

