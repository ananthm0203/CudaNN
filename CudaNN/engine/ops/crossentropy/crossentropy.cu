#include "crossentropy.h"
#include "../../utils/general_kernels.cuh"

#include <cuda_runtime.h>

static constexpr float EPS = 1e-16f;

__device__ void crossentropy_inner(float* y_true, float* y_pred, float* ce_target, size_t n)
{
	auto tx = threadIdx.x;
	float _ce = tx < n ? y_true[tx] * log2f(y_pred[tx] + EPS) : 0.0f;
	auto ce = WarpAllReduce<SumOp>(_ce);
	if (tx == 0)
	{
		*ce_target = -ce;
	}
}

void CategoricalCrossEntropy::forward()
{
	auto n = y_true->get_shape().C;
	auto pow2_ceil = 1 << (32 - __builtin_clz(static_cast<unsigned int>(n - 1)));

	float* ce_d;
	float* y_true_d;
	float* y_pred_d;

	checkCuda(cudaMalloc(&ce_d, sizeof(float) * n));
	checkCuda(cudaMalloc(&y_true_d, sizeof(float) * n));
	checkCuda(cudaMalloc(&y_pred_d, sizeof(float) * n));

	checkCuda(cudaMemcpy(y_true_d, y_true->raw(), sizeof(float) * n, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(y_pred_d, y_pred->raw(), sizeof(float) * n, cudaMemcpyHostToDevice));

	dim3 nb(1);
	dim3 nt(pow2_ceil);
	crossentropy_inner << <nb, nt >> > (y_true->raw(), y_pred->raw(), ce_d, n);

	checkCuda(cudaMemcpy(out.raw(), ce_d, sizeof(float), cudaMemcpyDeviceToHost));

	checkCuda(cudaFree(ce_d));
	checkCuda(cudaFree(y_true_d));
	checkCuda(cudaFree(y_pred_d));
}

void CategoricalCrossEntropy::backwards()
{
	auto n = y_true->get_shape().C;
	auto pow2_ceil = 1 << (32 - __builtin_clz(static_cast<unsigned int>(n - 1)));

	float* ce_grad_d;
	float* y_true_d;
	float* y_pred_d;

	checkCuda(cudaMalloc(&ce_grad_d, sizeof(float) * n));
	checkCuda(cudaMalloc(&y_true_d, sizeof(float) * n));
	checkCuda(cudaMalloc(&y_pred_d, sizeof(float) * n));

	checkCuda(cudaMemcpy(y_true_d, y_true->raw(), sizeof(float) * n, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(y_pred_d, y_pred->raw(), sizeof(float) * n, cudaMemcpyHostToDevice));

	dim3 nb(1);
	dim3 nt(pow2_ceil);
	auto f(static_cast<std::function<void(size_t, float*, float*, float*)>>
		(
			[](size_t i, float* t, float* p, float* d) { d[i] = - t[i] / p[i]; }
			));
	elemwiseOpKernel << <nb, nt >> > (1, 1, n, f, y_true_d, y_pred_d, ce_grad_d);

	y_pred->update_gradient(this, ce_grad_d);

	if (y_true->updateable())
	{
		f = static_cast<std::function<void(size_t, float*, float*, float*)>>
			(
				[](size_t i, float* t, float* p, float* d) { d[i] = -log2f(p[i] + EPS); }
		);
		elemwiseOpKernel << <nb, nt >> > (1, 1, n, f, y_true_d, y_pred_d, ce_grad_d);
		y_true->update_gradient(this, ce_grad_d);
	}

	checkCuda(cudaFree(ce_grad_d));
	checkCuda(cudaFree(y_true_d));
	checkCuda(cudaFree(y_pred_d));
}
