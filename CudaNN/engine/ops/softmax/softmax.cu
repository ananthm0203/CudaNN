#include "softmax.h"
#include "../../utils/general_kernels.cuh"
#include "../../utils/cuda_utils.cuh"

#include <cuda_runtime.h>

#define MIN(x, y)		(x < y ? x : y)
#define MAX(x, y)		(x > y ? x : y)

void Softmax::forward()
{
	auto n = in->get_shape().W;
	auto pow2_ceil = 1 << (32 - __builtin_clz(static_cast<unsigned int>(n - 1)));

	float* softmax_d;

	checkCuda(cudaMalloc(&softmax_d, sizeof(float) * n));

	checkCuda(cudaMemcpy(softmax_d, in->raw(), sizeof(float) * n, cudaMemcpyHostToDevice));

	dim3 nb(1); // To be extended for multiple rows, currently only supports 1 row
	dim3 nt(MIN(pow2_ceil, 32));

	softmax1DKernel1024M << <nb, nt >> > (softmax_d, softmax_d, n,
		MAX(1, CEILDIV(n, nt.x)), nt.x);

	checkCuda(cudaMemcpy(out.raw(), softmax_d, sizeof(float) * n, cudaMemcpyHostToDevice));

	checkCuda(cudaFree(softmax_d));
}

__global__ void softmaxGradKernel1024M(float* in_grad_d, float* sftmx_d, float* out_grad_d,
	size_t n, size_t cols_per_thread, size_t threads_per_group)
{
	assert(threads_per_group <= kWarpSize);

	auto tx = threadIdx.x;

	auto idx = tx * cols_per_thread;

	float thread_sum = 0.0f;

	for (size_t i = 0; i < cols_per_thread; ++i)
	{
		if (idx + i < n)
		{
			thread_sum += in_grad_d[idx + i] * sftmx_d[idx + i];
		}
	}

	auto warp_sum = WarpAllReduce<SumOp>(thread_sum, threads_per_group);

	for (size_t i = 0; i < cols_per_thread; ++i)
	{
		if (idx + i < n)
		{
			out_grad_d[idx + i] = (in_grad_d[idx + i] - warp_sum) * sftmx_d[idx + i];
		}
	}
}

void Softmax::backwards()
{
	auto n = in->get_shape().W;
	auto pow2_ceil = 1 << (32 - __builtin_clz(static_cast<unsigned int>(n - 1)));

	float* softmax_d, * in_grad_d, * softmax_grad_d;

	checkCuda(cudaMalloc(&softmax_d, sizeof(float) * n));
	checkCuda(cudaMalloc(&in_grad_d, sizeof(float) * n));
	checkCuda(cudaMalloc(&softmax_grad_d, sizeof(float) * n));

	checkCuda(cudaMemcpy(softmax_d, out.raw(), sizeof(float) * n, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(in_grad_d, out.gradient().raw(), sizeof(float) * n, cudaMemcpyHostToDevice));

	dim3 nb(1); // To be extended for multiple rows, currently only supports 1 row
	dim3 nt(MIN(pow2_ceil, 32));

	softmaxGradKernel1024M << <nb, nt >> > (in_grad_d, softmax_d, softmax_grad_d, n,
		MAX(1, CEILDIV(n, nt.x)), nt.x);

	in->update_gradient(this, softmax_grad_d);

	checkCuda(cudaFree(softmax_d));
	checkCuda(cudaFree(softmax_grad_d));
	checkCuda(cudaFree(in_grad_d));
}
