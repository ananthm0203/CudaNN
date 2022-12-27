#include "losses.h"

#include <cuda_runtime.h>

__inline__ __device__ void crossentropy_inner(float* y_true, float* y_pred, float* ce_target, size_t n)
{
	auto tx = threadIdx.x;
	auto _ce = tx < n ? y_true[tx] * log2f(y_pred[tx] + 1e-16) : 0.0f;
	auto ce = WarpAllReduce<SumOp, float>(_ce);
	if (tx == 0)
	{
		*ce_target = ce;
	}
}

Tensor CrossEntropy::cuda_crossentropy_grad(Tensor& y_true, Tensor& y_pred)
{
	auto n = y_true.get_shape().C;
	auto pow2_ceil = 1 << (32 - __builtin_clz(static_cast<unsigned int>(n - 1)));

	float* ce_grad_d;
	float* y_true_d;
	float* y_pred_d;
	checkCuda(cudaMalloc(&ce_grad_d, sizeof(float) * n));
	checkCuda(cudaMalloc(&y_true_d, sizeof(float) * n));
	checkCuda(cudaMalloc(&y_pred_d, sizeof(float) * n));

	checkCuda(cudaMemcpy(y_true_d, y_true.raw(), sizeof(float) * n, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(y_pred_d, y_pred.raw(), sizeof(float) * n, cudaMemcpyHostToDevice));

	dim3 nb(1);
	dim3 nt(pow2_ceil);
	elemwiseOpKernel<DivOp> << <nb, nt >> > (y_true_d, y_pred_d, ce_grad_d, 1, 1, n);

	Tensor grad(1, 1, n);
	checkCuda(cudaMemcpy(grad.raw(), ce_grad_d, sizeof(float) * n, cudaMemcpyDeviceToHost));

	checkCuda(cudaFree(y_true_d));
	checkCuda(cudaFree(y_pred_d));

	return grad;
}

float CrossEntropy::cuda_crossentropy(Tensor& y_true, Tensor& y_pred)
{
	auto n = y_true.get_shape().C;
	auto pow2_ceil = 1 << (32 - __builtin_clz(static_cast<unsigned int>(n - 1)));

	float* ce_d;
	checkCuda(cudaMalloc(&ce_d, sizeof(float)));

	dim3 nb(1);
	dim3 nt(pow2_ceil);
	crossentropy_inner <<<nb, nt>>> (y_true.raw(), y_pred.raw(), n);

	auto cost = * ce_d;

	checkCuda(cudaFree(ce_d));

	return -cost;
}

