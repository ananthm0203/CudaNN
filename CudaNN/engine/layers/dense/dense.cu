#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "../../utils/cuda_utils.cuh"
#include "../../activations/activations.h"
#include "dense.h"

#include <cuda_runtime.h>
#include <device_functions.h>
#include <stdint.h>
#include <memory>
#include <vector>
#include <string>

void denseForward(
	size_t C_in,
	size_t C_out,
	float* X, float* W, float* B, float* Z, float* R, Activation* activation)
{
	dim3 blocksPerGrid(1, TILENO(C_out));
	dim3 threadsPerBlock(1, TILE_WIDTH);
	size_t X_memsize = C_in * sizeof(float);
	size_t W_memsize = C_in * C_out * sizeof(float);
	size_t R_memsize = C_out * sizeof(float);

	// Allocate memory on the device
	float* X_d, * W_d, * R_d;
	checkCuda(cudaMalloc(&X_d, X_memsize));
	checkCuda(cudaMalloc(&W_d, W_memsize));
	checkCuda(cudaMalloc(&R_d, R_memsize));

	// Copy from the host to the device
	checkCuda(cudaMemcpy(X_d, X, X_memsize, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(W_d, W, W_memsize, cudaMemcpyHostToDevice));

	// Perform matrix multiplication on the device
	matMulKernel << <blocksPerGrid, threadsPerBlock >> > (W_d, X_d, R_d, C_out, C_in, 1, 1);

	// Free memory on the device
	checkCuda(cudaFree(W_d));
	checkCuda(cudaFree(X_d));

	// If we need to add bias...
	if (B)
	{
		size_t B_memsize = C_out * sizeof(float);
		// Allocate memory on the device
		float* B_d;
		checkCuda(cudaMalloc(&B_d, B_memsize));

		// Copy from the host to the device
		checkCuda(cudaMemcpy(B_d, B, B_memsize, cudaMemcpyHostToDevice));

		// Perform R (mat.) + Bias (vect.)
		vectMatAddKernel << <blocksPerGrid, threadsPerBlock >> > (B_d, R_d, R_d, C_out, 1, 1);

		// Free memory on the device
		checkCuda(cudaFree(B_d));
	}

	if (activation)
	{
		checkCuda(cudaMemcpy(Z, R_d, R_memsize, cudaMemcpyDeviceToHost));
		activation->forward(R_d);
	}

	// Copy back to device
	checkCuda(cudaMemcpy(R, R_d, R_memsize, cudaMemcpyDeviceToHost));

	// Free result matrix
	checkCuda(cudaFree(R_d));
}

void denseBackprop(
	size_t C_in,
	size_t C_out,
	float* grad,
	float* new_grad,
	float* z,
	float* X, 
	float* W, 
	float* B, 
	float* W_grad, 
	float* B_grad, 
	Activation* activation)
{
	float* grad_d = nullptr;
	size_t grad_memsize = C_in * sizeof(float);
	if (activation)
	{
		if (activation->in_place_grad())
		{
			float* z_d;
			checkCuda(cudaMalloc(&z_d, grad_memsize));
			checkCuda(cudaMemcpy(z_d, z, grad_memsize, cudaMemcpyHostToDevice));
			activation->backprop(z_d, nullptr);
			checkCuda(cudaMalloc(&grad_d, grad_memsize));
			checkCuda(cudaMemcpy(grad_d, grad, grad_memsize, cudaMemcpyHostToDevice));
			dim3 blocksPerGrid(TILENO(C_out));
			dim3 threadsPerBlock(TILE_WIDTH);
			elemwiseOpKernel<MulOp> << <blocksPerGrid, threadsPerBlock >> > (grad_d, z_d, grad_d,
				C_out, 1, 1);
			checkCuda(cudaFree(z_d));
		}
		else
		{
			float* z_d;
			float* grad_z_d;
			float* new_grad_d;
			auto grad_z_memsize = C_out * C_out * sizeof(float); // This has already been asserted in operator()
			checkCuda(cudaMalloc(&z_d, grad_memsize));
			checkCuda(cudaMemcpy(z_d, z, grad_memsize, cudaMemcpyHostToDevice));
			checkCuda(cudaMalloc(&grad_z_d, grad_z_memsize));
			activation->backprop(z_d, grad_z_d);
			checkCuda(cudaFree(z_d));
			checkCuda(cudaMalloc(&new_grad_d, grad_memsize));
			checkCuda(cudaMalloc(&grad_d, grad_memsize));
			checkCuda(cudaMemcpy(grad_d, grad, grad_memsize, cudaMemcpyHostToDevice));
			dim3 blocksPerGrid(TILENO(C_out), TILENO(C_out));
			dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
			matMulKernel << <blocksPerGrid, threadsPerBlock >> > (grad_z_d, grad_d, new_grad_d,
				C_out, C_out, 1, 1);
			checkCuda(cudaFree(grad_z_d));
			checkCuda(cudaFree(grad_d));
			grad_d = new_grad_d;
		}
	}
	else
	{
		checkCuda(cudaMalloc(&grad_d, grad_memsize));
		// Copy memory from host to device
		checkCuda(cudaMemcpy(grad_d, grad, grad_memsize, cudaMemcpyHostToDevice));
	}

	//// Calculate new_grad ////

	// Allocate memory
	auto W_memsize = C_out * C_in * sizeof(float);
	float* W_d;

	// Calculate transpose of the weights
	checkCuda(cudaMalloc(&W_d, W_memsize));

	// Copy weights to device
	checkCuda(cudaMemcpy(W_d, W, W_memsize, cudaMemcpyHostToDevice));

	dim3 blocksPerGrid(TILENO(C_in), TILENO(C_out));
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);

	// Allocate memory
	auto new_grad_memsize = C_in * sizeof(float);
	float* new_grad_d;
	checkCuda(cudaMalloc(&new_grad_d, new_grad_memsize));

	// Calculate new_grad
	matMulKernel << <blocksPerGrid, threadsPerBlock >> > (grad_d, W_d, new_grad_d, 1, C_out, C_in, 1);

	// Copy new_grad back to host
	checkCuda(cudaMemcpy(new_grad, new_grad_d, new_grad_memsize, cudaMemcpyDeviceToHost));

	// Free unneeded vars from device memory
	checkCuda(cudaFree(new_grad_d));
	checkCuda(cudaFree(W_d));

	//// End new_grad calculation ////

	//// Calculate weight gradient ////

	// Calculate the transpose of the input
	auto X_memsize = C_in * sizeof(float);
	float* X_d;
	checkCuda(cudaMalloc(&X_d, X_memsize));
	checkCuda(cudaMemcpy(X_d, X, X_memsize, cudaMemcpyHostToDevice));

	// Allocate memory for the weight gradient
	float* W_grad_d;
	checkCuda(cudaMalloc(&W_grad_d, W_memsize));

	// Calculate the weight gradient
	matMulKernel << <blocksPerGrid, threadsPerBlock >> > (grad_d, X_T, W_grad_d, C_out, 1, C_in, 1);

	// Copy back to host
	checkCuda(cudaMemcpy(W_grad, W_grad_d, W_memsize, cudaMemcpyDeviceToHost));

	//// Calculate bias gradient ////
	// If necessary...
	if (B)
	{
		auto B_memsize = C_out * sizeof(float);
		checkCuda(cudaMemcpy(B_grad, grad_d, B_memsize, cudaMemcpyDeviceToHost));
	}

	// Free all the memory on the device
	checkCuda(cudaFree(W_grad_d));
	checkCuda(cudaFree(X_d));
	checkCuda(cudaFree(grad_d));
}


#endif // KERNEL_CUH
