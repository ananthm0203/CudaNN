#ifndef ACTIVATION_KERNELS_H
#define ACTIVATION_KERNELS_H

#include <cuda_runtime.h>
#include <device_functions.h>
#include "device_launch_parameters.h"
#include <stdint.h>
#include <limits>
#include "cuda_utils.cuh"

__global__ void reluKernel(float* X, size_t M, size_t N)
{
	auto ROW = blockIdx.y * blockDim.y + threadIdx.y;
	auto COL = blockIdx.x * blockDim.x + threadIdx.x;

	if (ROW < M && COL < N)
	{
		X[ROW * N + COL] = max(X[ROW * N + COL], 0.f);
	}
}

__global__ void reluBackpropKernel(float* X, size_t M, size_t N)
{
	auto ROW = blockIdx.y * blockDim.y + threadIdx.y;
	auto COL = blockIdx.x * blockDim.x + threadIdx.x;

	if (ROW < M && COL < N)
	{
		X[ROW * N + COL] = X[ROW * N + COL] > 0 ? 1.f : 0.f;
	}
}

__global__ void linearKernel(float* X, size_t M, size_t N, float a, float b)
{
	auto ROW = blockIdx.y * blockDim.y + threadIdx.y;
	auto COL = blockIdx.x * blockDim.x + threadIdx.x;

	if (ROW < M && COL < N)
	{
		X[ROW * N + COL] = a * X[ROW * N + COL] + b;
	}
}

__global__ void linearBackpropKernel(float* X, size_t M, size_t N, float a)
{
	auto ROW = blockIdx.y * blockDim.y + threadIdx.y;
	auto COL = blockIdx.x * blockDim.x + threadIdx.x;

	if (ROW < M && COL < N)
	{
		X[ROW * N + COL] = a;
	}
}

__global__ void softmax1DKernel1024M(float* X, size_t M)
{
	__shared__ float _V[M / warpSize];

	auto tx = threadIdx.x;
	auto bx = blockIdx.x;
	auto IDX = blockDim.x * bx + tx;

	auto warp_id = tx / warpSize;
	auto lane_id = tx % warpSize;

	float x = IDX < M ? X[IDX] : -std::numeric_limits<float>::infinity();
	float x_max = x;
	x_max = WarpAllReduce<MaxOp, float>(x_max);
	if (!lane_id && IDX < M)
	{
		_V[warp_id] = x_max;
	}

	__syncthreads();

	if (IDX < M)
	{
		x_max = _V[lane_id];
	}
	x_max = WarpAllReduce<MaxOp, float>(x_max);
	x = IDX < M ? expf(x - x_max) : 0;

	float x_sum = x;
	x_sum = WarpAllReduce<SumOp, float>(x_sum);
	if (!lane_id && IDX < M)
	{
		_V[warp_id] = x_sum;
	}

	__syncthreads();

	if (IDX < M)
	{
		x_sum = _V[lane_id];
	}

	x_sum = WarpAllReduce<SumOp, float>(x_sum);
	if (IDX < M)
	{
		X[IDX] = x / x_sum;
	}
}

__global__ void softmax1DBackpropKernel1024M(float* X, float* X_DEST, size_t M)
{
	__shared__ float _V[M];

	auto tx = threadIdx.x;
	auto bx = blockIdx.x;
	auto IDX = blockDim.x * bx + tx;

	float x;

	if (IDX < M)
	{
		x = X[IDX];
		_V[IDX] = x;
	}

	__syncthreads();

	if (IDX < M)
	{
		for (size_t i = 0; i < M; ++i)
		{
			X_DEST[i * M + IDX] = i != IDX ? -_V[i] * x : (1 - x) * x;
		}
	}
}

#endif // ACTIVATION_KERNELS_H
