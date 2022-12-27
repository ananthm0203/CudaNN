#ifndef ACTIVATION_KERNELS_H
#define ACTIVATION_KERNELS_H

#include "../utils/cuda_utils.cuh"

#include <cuda_runtime.h>
#include <stdint.h>
#include <limits>

__global__ void reluKernel(float* X, size_t H, size_t W, size_t C)
{
	auto ROW = blockIdx.x * blockDim.x + threadIdx.x;
	auto COL = blockIdx.y * blockDim.y + threadIdx.y;
	auto AISLE = blockIdx.z * blockDim.z + threadIdx.z;

	if (COL < H && ROW < W && AISLE < C)
	{
		auto idx = ROW * W * C + COL * C + AISLE;
		X[idx] = max(X[idx], 0.f);
	}
}

__global__ void reluBackpropKernel(float* X, size_t H, size_t W, size_t C)
{
	auto ROW = blockIdx.x * blockDim.x + threadIdx.x;
	auto COL = blockIdx.y * blockDim.y + threadIdx.y;
	auto AISLE = blockIdx.z * blockDim.z + threadIdx.z;

	if (COL < H && ROW < W && AISLE < C)
	{
		auto idx = ROW * W * C + COL * C + AISLE;
		X[idx] = X[idx] > 0 ? 1.f : 0.f;
	}
}

__global__ void softmax1DKernel1024M(float* X, size_t C)
{
	__shared__ float _V[C / warpSize];

	auto tx = threadIdx.x;
	auto bx = blockIdx.x;
	auto IDX = blockDim.x * bx + tx;

	auto warp_id = tx / warpSize;
	auto lane_id = tx % warpSize;

	float x = IDX < C ? X[IDX] : -std::numeric_limits<float>::infinity();
	float x_max = x;
	x_max = WarpAllReduce<MaxOp, float>(x_max);
	if (!lane_id && IDX < C)
	{
		_V[warp_id] = x_max;
	}

	__syncthreads();

	if (IDX < C)
	{
		x_max = _V[lane_id];
	}
	x_max = WarpAllReduce<MaxOp, float>(x_max);
	x = IDX < C ? expf(x - x_max) : 0;

	float x_sum = x;
	x_sum = WarpAllReduce<SumOp, float>(x_sum);
	if (!lane_id && IDX < C)
	{
		_V[warp_id] = x_sum;
	}

	__syncthreads();

	if (IDX < C)
	{
		x_sum = _V[lane_id];
	}

	x_sum = WarpAllReduce<SumOp, float>(x_sum);
	if (IDX < C)
	{
		X[IDX] = x / x_sum;
	}
}

template<size_t C>
__global__ void softmax1DBackpropKernel1024M(float* X, float* X_DEST)
{
	__shared__ float _V[C];

	auto tx = threadIdx.x;
	auto bx = blockIdx.x;
	auto IDX = blockDim.x * bx + tx;

	float x;

	if (IDX < C)
	{
		x = X[IDX];
		_V[IDX] = x;
	}

	__syncthreads();

	if (IDX < C)
	{
		for (size_t i = 0; i < C; ++i)
		{
			X_DEST[i * C + IDX] = i != IDX ? -_V[i] * x : (1 - x) * x;
		}
	}
}

#endif // ACTIVATION_KERNELS_H
