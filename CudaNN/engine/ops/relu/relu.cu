#include "relu.h"
#include "../../utils/general_kernels.cuh"

#include <cuda_runtime.h>
#include <functional>
#include <cmath>

void Relu::forwards()
{
	float* relu_d;

	checkCuda(cudaMalloc(&relu_d, in->get_shape().size));

	checkCuda(cudaMemcpy(relu_d, in->raw(), in->get_shape().size, cudaMemcpyHostToDevice));

	auto f(static_cast<std::function<void(size_t, float*)>>(
		[](size_t idx, float* relu_d) { relu_d[idx] = std::max(0.0f, relu_d[idx]); }
	));
	elemwiseOpKernel(in->get_shape().H, in->get_shape().W, in->get_shape().W, f, relu_d);

	checkCuda(cudaMemcpy(out.raw(), relu_d, in->get_shape().size, cudaMemcpyDeviceToHost));

	checkCuda(cudaFree(relu_d));
}

void Relu::backwards()
{
	float* relu_grad_d;

	checkCuda(cudaMalloc(&relu_grad_d, in->get_shape().size));

	checkCuda(cudaMemcpy(relu_grad_d, in->raw(), in->get_shape().size, cudaMemcpyHostToDevice));

	auto f(static_cast<std::function<void(size_t, float*)>>(
		[](size_t idx, float* relu_grad_d) { relu_grad_d[idx] = relu_grad_d[idx] > 0.0f ? 1.0f : 0.0f; }
	));
	elemwiseOpKernel(in->get_shape().H, in->get_shape().W, in->get_shape().W, f, relu_grad_d);

	in->update_gradient(this, relu_grad_d);

	checkCuda(cudaFree(relu_grad_d));
}
