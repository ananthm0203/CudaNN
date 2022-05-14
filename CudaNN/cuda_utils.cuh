#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>   // For cuda types

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

#endif // CUDA_UTILS_H
