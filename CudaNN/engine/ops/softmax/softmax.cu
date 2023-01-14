#include "softmax.h"
#include "../../utils/general_kernels.cuh"
#include "../../utils/cuda_utils.cuh"

#include <cuda_runtime.h>

void Softmax::forwards()
{
	auto n = in->get_shape().W;
	auto pow2_ceil = 1 << (32 - __builtin_clz(static_cast<unsigned int>(n - 1)));

	float* softmax_d;

	checkCuda(cudaMalloc(&softmax_d, sizeof(float) * n));

	checkCuda(cudaMemcpy(softmax_d, in->raw(), sizeof(float) * n, cudaMemcpyHostToDevice));

	dim3 nb(1);
	dim3 nt(pow2_ceil);
	softmax1DKernel1024M << <nb, nt >> > (softmax_d, softmax_d, n);

	checkCuda(cudaMemcpy(out.raw(), softmax_d, sizeof(float) * n, cudaMemcpyHostToDevice));

	checkCuda(cudaFree(softmax_d));
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

	dim3 nb(CEILDIV(n * n, 32));
	dim3 nt(32);
	
	auto f(static_cast<std::function<void(size_t, float*, float*, float*)>>(
		[&](size_t ind, float* s, float* g_i, float* g_s) {
			float val(ind / n == ind % n ? s[ind / n] * (1 - s[ind / n]) : -s[ind / n] * s[ind % n]);
			atomicAdd(g_s + (ind / n), val * g_i[ind % n]);
		}
	));
	elemwiseOpKernel << <nb, nt >> > (n * n, 1, 1, f, softmax_d, in_grad_d, softmax_d);

	in->update_gradient(this, softmax_grad_d);

	checkCuda(cudaFree(softmax_d));
	checkCuda(cudaFree(softmax_grad_d));
	checkCuda(cudaFree(in_grad_d));
}
