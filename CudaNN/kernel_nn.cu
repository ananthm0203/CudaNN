#include "kernel_nn.cuh"
#include <cuda_runtime.h>
#include <device_functions.h>
#include "device_launch_parameters.h"
#include "activations.h"

__global__ void matMulKernel(float* A, float* B, float* C, uint32_t M, uint32_t N, uint32_t L)
{
	// Using shared memory
	__shared__ float _M1[TILE_WIDTH][TILE_WIDTH];
	__shared__ float _M2[TILE_WIDTH][TILE_WIDTH];

	auto tx = threadIdx.x;
	auto ty = threadIdx.y;
	auto bx = blockIdx.x;
	auto by = blockIdx.y;

	uint32_t ROW = by * TILE_WIDTH + ty;
	uint32_t COL = bx * TILE_WIDTH + tx;

	float tmpSum = 0;

	// Populate shared memory
	for (auto i = 0; i < (M - 1) / TILE_WIDTH + 1; ++i)
	{
		// Populate row-wise from A
		if (ROW < M && i * TILE_WIDTH + tx < N)
		{
			_M1[ty][tx] = A[ROW * N + i * TILE_WIDTH + tx];
		}
		else
		{
			_M1[ty][tx] = 0;
		}
		// Populate column-wise from B
		if (COL < L && i * TILE_WIDTH + ty < N)
		{
			_M2[ty][tx] = B[(i * TILE_WIDTH + ty) * L + COL];
		}
		else
		{
			_M2[ty][tx] = 0;
		}
		__syncthreads();
		// Op
		for (uint32_t i = 0; i < TILE_WIDTH; i++)
		{
			tmpSum += _M1[ty][i] * _M2[i][tx];
		}
		__syncthreads();
	}
	// Copy from shared memory to global
	if (ROW < M && COL < L)
	{
		C[ROW * L + COL] = tmpSum;
	}
}

__global__ void transposeKernel(float* A, float* T, uint32_t M, uint32_t N)
{
	__shared__ float _M[TILE_WIDTH][TILE_WIDTH + 1]; // Padded dimension prevents bank conflicts

	auto tx = threadIdx.x;
	auto ty = threadIdx.y;
	auto bx = blockIdx.x;
	auto by = blockIdx.y;

	uint32_t ROW = by * TILE_WIDTH + ty;
	uint32_t COL = bx * TILE_WIDTH + tx;

	for (auto i = 0; i < TILE_WIDTH; i += BLOCK_SIZE)
	{
		if (COL < N && (ROW + i) < M)
		{
			_M[ty + i][tx] = A[(ROW + i) * N + COL];
		}
	}
	__syncthreads();

	ROW = bx * TILE_WIDTH + ty;
	COL = by * TILE_WIDTH + tx;

	for (size_t i = 0; i < TILE_WIDTH; i += BLOCK_SIZE)
	{
		if (COL < M && (ROW + i) < N)
		{
			T[(ROW + i) * M + COL] = _M[tx][ty + i];
		}
	}
}

__global__ void matAddKernel(float* A, float* B, uint32_t M, uint32_t N)
{
	auto tx = threadIdx.x;
	auto ty = threadIdx.y;
	auto bx = blockIdx.x;
	auto by = blockIdx.y;

	uint32_t ROW = by * TILE_WIDTH + ty;
	uint32_t COL = bx * TILE_WIDTH + tx;

	if (ROW < M && COL < N)
	{
		B[ROW * N + COL] += A[ROW * N + COL];
	}
}

// Adds [M x 1] vector A to [M x N] vector B
__global__ void vectMatAddKernel(float* A, float* B, uint32_t M, uint32_t N)
{
	__shared__ float _V[TILE_WIDTH];

	auto tx = threadIdx.x;
	auto ty = threadIdx.y;
	auto bx = blockIdx.x;
	auto by = blockIdx.y;

	uint32_t ROW = by * TILE_WIDTH + ty;
	uint32_t COL = bx * TILE_WIDTH + tx;

	// Populate shared memory
	if (ROW < M && !tx && COL < M)
	{
		_V[ROW] = A[ROW];
	}

	// Synchronize threads
	__syncthreads();

	if (ROW < M && COL < N)
	{
		B[ROW * N + COL] += _V[ROW];
	}
}

void matMul(float* A, float* B, float* C, uint32_t M, uint32_t N, uint32_t L)
{
	dim3 blocksPerGrid((L - 1) / TILE_WIDTH + 1, (M - 1) / TILE_WIDTH + 1);
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	uint32_t A_memsize = M * N * sizeof(float);
	uint32_t B_memsize = N * L * sizeof(float);
	uint32_t C_memsize = M * L * sizeof(float);

	// Allocate memory on the device
	float* A_d, * B_d, * C_d;
	checkCuda(cudaMalloc(&A_d, A_memsize));
	checkCuda(cudaMalloc(&B_d, B_memsize));
	checkCuda(cudaMalloc(&C_d, C_memsize));

	// Copy from the host to the device
	checkCuda(cudaMemcpy(A_d, A, A_memsize, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(B_d, B, B_memsize, cudaMemcpyHostToDevice));

	// Perform matrix multiplication on the device
	matMulKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, C, M, N, L);

	// Copy from the device to the host
	checkCuda(cudaMemcpy(C_d, C, C_memsize, cudaMemcpyDeviceToHost));

	// Free memory on the device
	checkCuda(cudaFree(A_d));
	checkCuda(cudaFree(B_d));
	checkCuda(cudaFree(C_d));
}

void transpose(float* A, float* T, uint32_t M, uint32_t N)
{
	dim3 blocksPerGrid((N - 1) / TILE_WIDTH + 1, (N - 1) / TILE_WIDTH + 1);
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	uint32_t mem_size = M * N * sizeof(float);

	// Allocate memory on the device
	float* A_d, * T_d;
	checkCuda(cudaMalloc(&A_d, mem_size));
	checkCuda(cudaMalloc(&T_d, mem_size)); // Transpose should be the same size

	// Copy from host to the device
	checkCuda(cudaMemcpy(A_d, A, mem_size, cudaMemcpyHostToDevice));

	// Perform the transpose on the device
	transposeKernel << <blocksPerGrid, threadsPerBlock >> > (A, T, M, N);

	// Copy from the device to the host
	checkCuda(cudaMemcpy(T, T_d, mem_size, cudaMemcpyDeviceToHost));

	// Free memory on the device
	checkCuda(cudaFree(A_d));
	checkCuda(cudaFree(T_d));
}

template<bool use_bias, Activation *activation>
void matMulAdd(float* X, float* W, float* B, float* R, uint32_t M, uint32_t N, uint32_t L)
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

	// Free result matrix
	checkCuda(cudaFree(R_d));
}