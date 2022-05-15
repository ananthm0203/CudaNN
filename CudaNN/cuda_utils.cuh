#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>   // For cuda types
#include <type_traits>

#define TILE_WIDTH 32
#define BLOCK_SIZE 8

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
struct ReduceMax
{
	__device__ T operator()(T val1, T val2)
	{
		return max(val1, val2);
	}
};

template<typename T>
struct ReduceSum
{
	__device__ T operator()(T val1, T val2)
	{
		return val1 + val2;
	}
};

template<template<typename> typename ReductionOp, typename T>
__inline__ __device__ T WarpAllReduce(T val, uint32_t mask = 0xffffffff) {
	for (int laneMask = warpSize / 2; laneMask > 0; laneMask /= 2) {
		val = ReductionOp<T>()(val, __shfl_xor_sync(mask, val, laneMask));
	}
	return val;
}

#endif // CUDA_UTILS_H
