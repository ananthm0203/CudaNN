#include "crossentropy.h"
#include "../../utils/general_kernels.cuh"

#include <cuda_runtime.h>

static constexpr float EPS = 1e-16f;

static __device__ void _crossentropy_inner(float* y_true, float* y_pred, float* ce_target, size_t n)
{
	__shared__ float _V[n / warpSize];

	auto tx = threadIdx.x;
	auto bx = blockIdx.x;
	auto idx = blockDim.x * bx + tx;

	auto warp_id = tx / warpSize;
	auto lane_id = tx % warpSize;

	float ce = tx < n ? y_true[tx] * log2f(y_pred[tx] + EPS) : 0.0f;
	ce = WarpAllReduce<SumOp>(ce);

	if (!lane_id)
	{
		_V[warp_id] = ce;
	}

	ce = lane_id < (n / warpSize) ? _V[lane_id] : 0;
	ce = WarpAllReduce<SumOp>(ce);

	if (idx == 0)
	{
		*ce_target = -ce;
	}
}

void CrossEntropy::forwards()
{
	auto n = y_true->get_shape().W;
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
	_crossentropy_inner << <nb, nt >> > (y_true->raw(), y_pred->raw(), ce_d, n);

	checkCuda(cudaMemcpy(out.raw(), ce_d, sizeof(float), cudaMemcpyDeviceToHost));

	checkCuda(cudaFree(ce_d));
	checkCuda(cudaFree(y_true_d));
	checkCuda(cudaFree(y_pred_d));
}

void CrossEntropy::backwards()
{
	auto n = y_true->get_shape().W;
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
			[](size_t i, float* t, float* p, float* d) { d[i] = -t[i] / p[i]; }
	));
	elemwiseOpKernel << <nb, nt >> > (1, 1, n, f, y_true_d, y_pred_d, ce_grad_d);

	y_pred->update_gradient(this, ce_grad_d);

	if (!y_true->no_grad())
	{
		auto f = static_cast<std::function<void(size_t, float*, float*)>>
			(
				[](size_t i, float* p, float* d) { d[i] = -log2f(p[i] + EPS); }
		);
		elemwiseOpKernel << <nb, nt >> > (1, 1, n, f, y_pred_d, ce_grad_d);
		y_true->update_gradient(this, ce_grad_d);
	}

	checkCuda(cudaFree(ce_grad_d));
	checkCuda(cudaFree(y_true_d));
	checkCuda(cudaFree(y_pred_d));
}

static __device__ void _softmax_crossentropy_inner(float* y_true, float* y_pred, float* softmax, size_t n)
{
	auto tx = threadIdx.x;
	float _ce = tx < n ? y_true[tx] * log2f(y_pred[tx] + EPS) : 0.0f;
	auto ce = WarpAllReduce<SumOp>(_ce);
	softmax[tx] = 0;
}

void SoftmaxCrossEntropyWithLogits::forwards()
{
	auto n = y_true->get_shape().W;
	auto pow2_ceil = 1 << (32 - __builtin_clz(static_cast<unsigned int>(n - 1)));

	float* ce_d;
	float* y_true_d;
	float* y_pred_d;
	float* softmax_d;

	checkCuda(cudaMalloc(&ce_d, sizeof(float) * n));
	checkCuda(cudaMalloc(&y_true_d, sizeof(float) * n));
	checkCuda(cudaMalloc(&y_pred_d, sizeof(float) * n));
	checkCuda(cudaMalloc(&softmax_d, sizeof(float) * n));

	checkCuda(cudaMemcpy(y_true_d, y_true->raw(), sizeof(float) * n, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(y_pred_d, y_pred->raw(), sizeof(float) * n, cudaMemcpyHostToDevice));

	dim3 nb(1);
	dim3 nt(pow2_ceil);
	softmax1DKernel1024M << <nb, nt >> > (y_pred_d, softmax_d, n);
	_crossentropy_inner << <nb, nt >> > (y_true_d, softmax_d, ce_d, n);

	checkCuda(cudaMemcpy(out.raw(), ce_d, sizeof(float), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(softmax.raw(), softmax_d, n, cudaMemcpyDeviceToHost));

	checkCuda(cudaFree(ce_d));
	checkCuda(cudaFree(softmax_d));
	checkCuda(cudaFree(y_true_d));
	checkCuda(cudaFree(y_pred_d));
}

void SoftmaxCrossEntropyWithLogits::backwards()
{
	auto n = y_true->get_shape().W;
	auto pow2_ceil = 1 << (32 - __builtin_clz(static_cast<unsigned int>(n - 1)));

	float* ce_grad_d;
	float* y_true_d;
	float* softmax_d;

	checkCuda(cudaMalloc(&ce_grad_d, sizeof(float) * n));
	checkCuda(cudaMalloc(&y_true_d, sizeof(float) * n));
	checkCuda(cudaMalloc(&softmax_d, sizeof(float) * n));

	checkCuda(cudaMemcpy(y_true_d, y_true->raw(), sizeof(float) * n, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(softmax_d, softmax.raw(), sizeof(float) * n, cudaMemcpyHostToDevice));

	dim3 nb(1);
	dim3 nt(pow2_ceil);

	auto f(static_cast<std::function<void(size_t, float*, float*, float*)>>(
		[](size_t idx, float* t, float* p, float* d) { d[idx] = p[idx] - t[idx]; }
		));
	elemwiseOpKernel << <nb, nt >> > (1, 1, n, f, y_true_d, softmax_d, ce_grad_d);

	y_pred->update_gradient(this, ce_grad_d);

	if (!y_true->no_grad())
	{
		auto f = static_cast<std::function<void(size_t, float*, float*)>>
			(
				[](size_t i, float* p, float* d) { d[i] = -log2f(p[i] + EPS); }
		);
		elemwiseOpKernel << <nb, nt >> > (1, 1, n, f, softmax_d, ce_grad_d);
		y_true->update_gradient(this, ce_grad_d);
	}

	checkCuda(cudaFree(ce_grad_d));
	checkCuda(cudaFree(softmax_d));
	checkCuda(cudaFree(y_true_d));
}
