#include "kernel_nn.cuh"
#include <cuda_runtime.h>
#include <device_functions.h>
#include "device_launch_parameters.h"
#include "activations.h"
#include <memory>
#include <vector>
#include <string>

using std::unique_ptr;
using std::vector;

template<bool use_bias, typename T>
void denseForward(float* X, float* W, float* B, float* R, uint32_t M, uint32_t N, uint32_t L)
{
	dim3 blocksPerGrid((L - 1) / TILE_WIDTH + 1, (N - 1) / TILE_WIDTH + 1);
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	uint32_t X_memsize = M * N * sizeof(float);
	uint32_t W_memsize = L * M * sizeof(float);
	uint32_t R_memsize = L * N * sizeof(float);

	// Allocate memory on the device
	float* X_d, * W_d, * R_d;
	checkCuda(cudaMalloc(&X_d, X_memsize));
	checkCuda(cudaMalloc(&W_d, W_memsize));
	checkCuda(cudaMalloc(&R_d, R_memsize));

	// Copy from the host to the device
	checkCuda(cudaMemcpy(X_d, X, X_memsize, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(W_d, W, W_memsize, cudaMemcpyHostToDevice));

	// Perform matrix multiplication on the device
	matMulKernel << <blocksPerGrid, threadsPerBlock >> > (W_d, X_d, R_d, L, M, N);

	// Free memory on the device
	checkCuda(cudaFree(W_d));
	checkCuda(cudaFree(X_d));

	// If we need to add bias...
	if (use_bias)
	{
		uint32_t B_memsize = L * sizeof(float);
		// Allocate memory on the device
		float* B_d;
		checkCuda(cudaMalloc(&B_d, B_memsize));

		// Copy from the host to the device
		checkCuda(cudaMemcpy(B_d, B, B_memsize, cudaMemcpyHostToDevice));

		// Perform R (mat.) + Bias (vect.)
		vectMatAddKernel << <blocksPerGrid, threadsPerBlock >> > (B, R, L, N);

		// Free memory on the device
		checkCuda(cudaFree(B_d));
	}

	if (activation)
	{
		(*activation)(R_d, L, N);
	}

	// Copy back to device
	checkCuda(cudaMemcpy(R, R_d, R_memsize, cudaMemcpyDeviceToHost);

	// Free result matrix
	checkCuda(cudaFree(R_d));
}

template<bool use_bias, typename Activation>
void denseBackprop(float* grad, float* new_grad, float* z, float* X, float* W, float* B, float* W_grad, float* B_grad, size_t L, size_t M, size_t N)
{
	auto grad_memsize = L * N * sizeof(float);
	float* grad_d;
	if (activation)
	{
		if (T.in_place_grad())
		{
			(T.forward())
		}
		
		if (Activation.in_place_grad())
		{
			float* z_d;
			checkCuda(cudaMalloc(&z_d, grad_memsize));
			checkCuda(cudaMemcpy(z_d, z, grad_memsize, cudaMemcpyHostToDevice));
			Activation.backprop(z_d, L, N);
			checkCuda(cudaMalloc(&grad_d, grad_memsize));
			checkCuda(cudaMemcpy(grad_d, grad, grad_memsize, cudaMemcpyHostToDevice));
			dim3 blocksPerGrid((L - 1) / TILE_WIDTH + 1, (N - 1) / TILE_WIDTH + 1);
			dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
			hadamardKernel << <blocksPerGrid, threadsPerBlock >> > (z_d, grad_d, L, N);
			checkCuda(cudaFree(z_d));
			// We might need to copy back to the CPU for memory savings, but I don't think
			// we need to do that for now;
			// TODO: wrap other declarations of grad_d in if-blocks to prevent double mallocs
		}
		else
		{

		}
		checkCuda(cudaMalloc(&z_grad_d, grad_memsize));
	}
	else
	{
		checkCuda(cudaMalloc(&grad_d, grad_memsize));
		// Copy memory from host to device
		checkCuda(cudaMemcpy(grad_d, grad, grad_memsize, cudaMemcpyHostToDevice));
		
	}

	//// Calculate new_grad ////

	// Allocate memory
	auto W_memsize = L * M * sizeof(float);
	float* W_d;
	float* W_T_d;

	// Calculate transpose of the weights
	checkCuda(cudaMalloc(&W_d, W_memsize));
	checkCuda(cudaMalloc(&W_T_d, W_memsize));

	// Copy weights to device
	checkCuda(cudaMemcpy(W_d, W, W_memsize, cudaMemcpyHostToDevice));
	
	dim3 blocksPerGrid((L - 1) / TILE_WIDTH + 1, (M - 1) / TILE_WIDTH + 1);
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	// Run transpose kernel
	transposeKernel << <blocksPerGrid, threadsPerBlock >> > (W, W_d, L, M);

	// Allocate memory
	auto new_grad_memsize = M * N * sizeof(float);
	float* new_grad_d;
	checkCuda(cudaMalloc(&new_grad_d, new_grad_memsize));
	
	// Calculate new_grad
	matMulKernel << <blocksPerGrid, threadsPerBlock >> > (W_T_d, grad_d, new_grad_d, M, L, N);

	// Copy new_grad back to host
	checkCuda(cudaMemcpy(new_grad, new_grad_d, new_grad_memsize, cudaMemcpyDeviceToHost));

	// Free unneeded vars from device memory
	checkCuda(cudaFree(new_grad_d));
	checkCuda(cudaFree(W_T_d));
	checkCuda(cudaFree(W_d));

	//// End new_grad calculation ////
	
	//// Calculate weight gradient ////
	
	// Calculate the transpose of the input
	auto X_memsize = M * N * sizeof(float);
	float* X_d;
	float* X_T_d;
	checkCuda(cudaMalloc(&X_d, X_memsize));
	checkCuda(cudaMalloc(&X_T_d, X_memsize));
	transposeKernel << <blocksPerGrid, threadsPerBlock >> > (X_d, X_T_d, M, N);

	// X is no longer needed
	checkCuda(cudaFree(X_d));

	// Allocate memory for the weight gradient
	float* W_grad_d;
	checkCuda(cudaMalloc(&W_grad_d, W_memsize));

	// Calculate the weight gradient
	matMulKernel << <blocksPerGrid, threadsPerBlock >> > (grad_d, X_T_d, W_grad_d, L, N, M);
	
	// Copy back to host
	checkCuda(cudaMemcpy(W_grad, W_grad_d, W_memsize, cudaMemcpyDeviceToHost));

	// Free all the memory on the device
	checkCuda(cudaFree(W_grad_d));
	checkCuda(cudaFree(X_T_d));
	checkCuda(cudaFree(grad_d));

	//// Calculate bias gradient ////
	// If necessary...
	if (use_bias)
	{
		auto B_memsize = L * N * sizeof(float);
		memcpy(B_grad, grad, B_memsize);
	}
}