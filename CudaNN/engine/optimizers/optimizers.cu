#include "optimizers.h"
#include "../utils/cuda_utils.cuh"
#include "../utils/general_kernels.cuh"

#include <cuda_runtime.h>

void Optimizer::update_grads()
{

	auto f(static_cast<std::function<void(size_t, float*, float*)>>(
		[&](size_t ind, float* A, float* B) { B[ind] += A[ind] / batch_size; }
	));

	for (auto& wgp : weights)
	{
		float* grad_d;
		float* accum_grad_d;

		checkCuda(cudaMalloc(&grad_d, wgp.first->get_shape().size));
		checkCuda(cudaMalloc(&accum_grad_d, wgp.first->get_shape().size));

		checkCuda(cudaMemcpy(grad_d, wgp.first->gradient().raw(), wgp.first->get_shape().size, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(accum_grad_d, wgp.second.raw(), wgp.first->get_shape().size, cudaMemcpyHostToDevice));

		auto& shape = wgp.first->get_shape();

		DEFAULT_CUDA_DIMS_FROM_SHAPE(shape);

		elemwiseOpKernel << <blocksPerGrid, threadsPerBlock >> > (shape.H, shape.W, shape.C, f, grad_d, accum_grad_d);

		checkCuda(cudaFree(accum_grad_d));
		checkCuda(cudaFree(grad_d));
	}
}
