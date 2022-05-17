#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <stdint.h>
#include "cuda_utils.cuh"
#include "activations.h"


__global__ void hadamardKernel(float* A, float* B, uint32_t M, uint32_t N)
{
	auto tx = threadIdx.x;
	auto ty = threadIdx.y;
	auto bx = blockIdx.x;
	auto by = blockIdx.y;

	uint32_t ROW = by * blockDim.x + ty;
	uint32_t COL = bx * blockDim.y + tx;

	if (ROW < M && COL < N)
	{
		B[ROW * N + COL] *= A[ROW * N + COL];
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

	uint32_t ROW = by * gridDim.y + ty;
	uint32_t COL = bx * gridDim.x + tx;

	if (ROW < M && COL < N)
	{
		B[ROW * N + COL] += A[ROW * N + COL];
	}
}

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

template<bool use_bias, Activation* activation>
void denseForward(float* X, float* W, float* B, float* R, uint32_t M, uint32_t N, uint32_t L);

template<bool use_bias, Activation* activation>
void denseBackprop(float* grad_prop, float* X, float* W, float* B, float* grad_dest, float* W_grad, float* B_grad, uint32_t M, uint32_t N, uint32_t L);

#endif // KERNEL_CUH
