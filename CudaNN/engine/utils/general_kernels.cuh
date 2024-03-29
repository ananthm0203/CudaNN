#ifndef GENERAL_KERNELS_H
#define GENERAL_KERNELS_H

#include "cuda_utils.cuh"

#include <cuda_runtime.h>
#include <device_functions.h>
#include <functional>

template<typename... Args>
__global__ void elemwiseOpKernel(size_t H, size_t W, size_t C,
	std::function<void(size_t, Args...)> f, Args... inputs)
{
	auto tx = threadIdx.x;
	auto ty = threadIdx.y;
	auto tz = threadIdx.z;
	auto bx = blockIdx.x;
	auto by = blockIdx.y;
	auto bz = blockIdx.z;

	size_t COL = bx * blockDim.x + tx;
	size_t ROW = by * blockDim.y + ty;
	size_t AISLE = bz * blockDim.z + bz;

	if (COL < H && ROW < W && AISLE < C)
	{
		auto idx = ROW * W * C + COL * C + AISLE;
		f(idx, inputs...);
	}
}

template<typename T>
__global__ void transposeKernel(T* A, T* A_T, size_t H, size_t W, size_t C)
{
	__shared__ float _M[BLCK_X][BLCK_X + 1][BLCK_Z]; // Padded dimension prevents bank conflicts

	auto tx = threadIdx.x;
	auto ty = threadIdx.y;
	auto tz = threadIdx.z;
	auto bx = blockIdx.x;
	auto by = blockIdx.y;
	auto bz = blockIdx.z;

	size_t ROW = bx * blockDim.x + tx;
	size_t COL = by * blockDim.y + ty;
	size_t AISLE = bz * blockDim.z + tz;

	for (auto i = 0; i < BLCK_X; i += BLOCK_SIZE)
	{
		if (ROW + i < H && COL < W && AISLE < C)
		{
			auto idx = (ROW + i) * W * C + COL * C + AISLE;
			_M[tx + i][ty][tz] = A[idx];
		}
	}
	__syncthreads();

	ROW = by * BLCK_X + tx;
	COL = bx * BLCK_X + ty;

	for (auto i = 0; i < BLCK_X; i += BLOCK_SIZE)
	{
		if (ROW + i < W && COL < H && AISLE < C)
		{
			auto idx = (ROW + i) * H * C + COL * C + AISLE;
			A_T[idx] = _M[ty][tx + i][tz];
		}
	}
}

template<typename T>
__global__ void matMulKernel(T* A, T* B, T* dst, size_t H1, size_t W1, size_t W2, size_t C)
{
	// Using shared memory
	__shared__ float _M1[BLCK_X][BLCK_X][BLCK_Z];
	__shared__ float _M2[BLCK_X][BLCK_X][BLCK_Z];

	auto tx = threadIdx.x;
	auto ty = threadIdx.y;
	auto tz = threadIdx.z;
	auto bx = blockIdx.x;
	auto by = blockIdx.y;
	auto bz = blockIdx.z;

	size_t ROW = bx * blockDim.x + tx;
	size_t COL = by * blockDim.y + ty;
	size_t AISLE = bz * blockDim.z + tz;

	float tmpSum = 0;

	// Populate shared memory
	for (auto i = 0; i < CEILDIV(W1, BLCK_X); ++i)
	{
		// Populate row-wise from A
		if (ROW < H1 && i * BLCK_X + ty < W1 && AISLE < C)
		{
			auto idx = ROW * W * C + (i * BLCK_X + ty) * C + AISLE;
			_M1[tx][ty][tz] = A[idx];
		}
		else
		{
			_M1[tx][ty][tz] = 0;
		}
		// Populate column-wise from B
		if (i * BLCK_X + tx < W1 && COL < W2 && AISLE < C)
		{
			auto idx = (i * BLCK_X + tx) * W * C + COL * C + AISLE;
			_M2[tx][ty][tz] = B[idx];
		}
		else
		{
			_M2[tx][ty][tz] = 0;
		}
		__syncthreads();
		// Op
		for (size_t i = 0; i < BLCK_X; i++)
		{
			tmpSum += _M1[i][ty][tz] * _M2[tx][i][tz];
		}
		__syncthreads();
	}
	// Copy from shared memory to global
	if (ROW < H1 && COL < W2 && AISLE < C)
	{
		auto idx = ROW * W * C + COL * C + AISLE;
		dst[idx] = tmpSum;
	}
}

// Adds [W x C] vector V to [H x W x C] matrix 
// Note: can probably be generalized via templating
template<typename T>
__global__ void vectMatAddKernel(T* V, T* B, T* dst, size_t H, size_t W, size_t C)
{
	auto tx = threadIdx.x;
	auto ty = threadIdx.y;
	auto tz = threadIdx.z;
	auto bx = blockIdx.x;
	auto by = blockIdx.y;
	auto bz = blockIdx.z;

	size_t ROW = bx * blockDim.x + ty;
	size_t COL = by * blockDim.y + tx;
	size_t AISLE = bz * blockDim.z + tz;

	if (ROW < H && COL < W && AISLE < C)
	{
		auto vidx = COL * C + AISLE;
		auto midx = ROW * W * C + COL * C + AISLE;
		dst[midx] = B[midx] + V[vidx];
	}
}

template<typename T>
__global__ void softmax1DKernel1024M(T* X, T* X_DEST, size_t n, size_t cols_per_thread, size_t threads_per_group)
{

	assert(n <= 1024);
	assert(threads_per_group <= kWarpSize && kWarpSize % threads_per_group == 0);

	auto tx = threadIdx.x;
	auto idx = tx * cols_per_thread;

	float x_max = -std::numeric_limits<float>::infinity();

	for (size_t i = 0; i < cols_per_thread; ++i)
	{
		if (idx + i < n)
		{
			x_max = min(x_max, X[idx + i]);
		}
	}

	x_max = WarpAllReduce<MaxOp>(x_max, threads_per_group);

	float x_sum = 0.0;

	for (size_t i = 0; i < cols_per_thread; ++i)
	{
		if (idx + i < n)
		{
			X_DEST[idx + i] = expf(X[idx + i] - x_max);
			x_sum += X_DEST[idx + i];
		}
	}

	x_sum = WarpAllReduce<SumOp>(x_sum, threads_per_group);

	for (size_t i = 0; i < cols_per_thread; ++i)
	{
		if (idx + i < n)
		{
			X_DEST[idx + i] /= x_sum;
		}
	}
}

#endif