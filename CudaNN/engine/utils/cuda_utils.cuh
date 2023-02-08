#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>   // For cuda types
#include <type_traits>

constexpr auto kWarpSize = 32;

constexpr auto TILE_WIDTH = 32;
constexpr auto BLOCK_SIZE = 8;

constexpr auto BLCK_X = 16;
constexpr auto BLCK_Y = 16;
constexpr auto BLCK_Z = 4;

#define CEILDIV(N, D)		((N - 1) / D + 1)
#define TILENO(N)			(CEILDIV(N, TILE_WIDTH))

#define DEFAULT_CUDA_DIMS(H, W, C)													\
	dim3 blocksPerGrid(CEILDIV(H, BLCK_X), CEILDIV(W, BLCK_Y), CEILDIV(C, BLCK_Z));	\
	dim3 threadsPerBlock(BLCK_X, BLCK_Y, BLCK_Z);									\

#define DEFAULT_CUDA_DIMS_FROM_SHAPE(shape) DEFAULT_CUDA_DIMS(shape.H, shape.W, shape.C)

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		std::cerr << stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result);
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

template<typename T>
struct DivOp
{
	__device__ T operator()(T val1, T val2)
	{
		return val1 / val2;
	}
};

template<typename T>
struct PowOp
{
	__device__ T operator()(T val1, T val2)
	{
		return pow(val2, val1);
	}
};

template<typename T>
struct MulOp
{
	__device__ T operator()(T val1, T val2)
	{
		return val1 * val2;
	}
};

template<typename T>
struct MaxOp
{
	__device__ T operator()(T val1, T val2)
	{
		return max(val1, val2);
	}
};

template<typename T>
struct SumOp
{
	__device__ T operator()(T val1, T val2)
	{
		return val1 + val2;
	}
};

template<template<typename> typename ReductionOp, typename T>
__inline__ __device__ T WarpAllReduce(T val, uint32_t laneWidth = kWarpSize, uint32_t mask = 0xffffffff) {
	for (int laneMask = laneWidth / 2; laneMask > 0; laneMask /= 2) {
		val = ReductionOp<T>()(val, __shfl_xor_sync(mask, val, laneMask));
	}
	return val;
}

#endif // CUDA_UTILS_H
