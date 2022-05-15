#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "cuda_utils.cuh"
#include "activation_kernels.cuh"

typedef struct Activation
{
	//virtual void operator()(float* z, size_t M, size_t N) = 0;
	//virtual void backprop(float* z, size_t M, size_t N) = 0;
} Activation;

typedef struct ReLU : Activation
{
	void operator()(float* z, size_t M, size_t N)
	{
		dim3 blocksPerGrid((M - 1) / TILE_WIDTH + 1, (N - 1) / TILE_WIDTH + 1);
		dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
		reluKernel << <blocksPerGrid, threadsPerBlock >> > (z, M, N);
	}
	void backprop(float* z, size_t M, size_t N)
	{
		dim3 blocksPerGrid((M - 1) / TILE_WIDTH + 1, (N - 1) / TILE_WIDTH + 1);
		dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
		reluBackpropKernel << <blocksPerGrid, threadsPerBlock >> > (z, M, N);
	}
} ReLU;

typedef struct Linear : Activation
{
	void operator()(float* z, size_t M, size_t N)
	{
		dim3 blocksPerGrid((M - 1) / TILE_WIDTH + 1, (N - 1) / TILE_WIDTH + 1);
		dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
		linearKernel << <blocksPerGrid, threadsPerBlock >> > (z, M, N);
	}
	void backprop(float* z, size_t M, size_t N)
	{
		dim3 blocksPerGrid((M - 1) / TILE_WIDTH + 1, (N - 1) / TILE_WIDTH + 1);
		dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
		linearBackpropKernel << <blocksPerGrid, threadsPerBlock >> > (z, M, N);
	}
} Linear;

typedef struct Softmax : Activation
{
	void operator()(float* z, size_t M, size_t N)
	{
		if (N == 1)
		{
			if (M <= 1024)
			{
				dim3 blocksPerGrid;
				dim3 threadsPerBlock(((M - 1) / warpSize + 1) * warpSize);
				softmax1DKernel1024M << <blocksPerGrid, threadsPerBlock >> > (z, M);
			}
			else
			{
				throw cudaErrorNotYetImplemented;
			}
		}
		else
		{
			throw cudaErrorNotYetImplemented;
		}
	}
	void backprop(float* z, float* z_dest, size_t M, size_t N)
	{
		if (N == 1)
		{
			if (M <= 1024)
			{
				dim3 blocksPerGrid;
				dim3 threadsPerBlock(((M - 1) / warpSize + 1) * warpSize);
				softmax1DBackpropKernel1024M << <blocksPerGrid, threadsPerBlock >> > (z, z_dest, M);
			}
			else
			{
				throw cudaErrorNotYetImplemented;
			}
		}
		else
		{
			throw cudaErrorNotYetImplemented;
		}
	}
} Softmax;

#endif // ACTIVATIONS_H
