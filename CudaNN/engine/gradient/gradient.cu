#include "gradient.h"
#include "../utils/cuda_utils.cuh"
#include "../utils/general_kernels.cuh"

#include <cuda_runtime.h>

void CudaGradient::accum_from_wrapped(uintptr_t handler_id, const CudaGradient& other_grad)
{
	assert(grad.get());
	if (handler_id == last_handler_id)
	{
		memcpy(grad.get(), other_grad.grad.get(), shape.size);
		return;
	}

	float* other_grad_d;
	
	checkCuda(cudaMalloc(&other_grad_d, shape.size));
	checkCuda(cudaMemcpy(other_grad_d, grad.get(), shape.size, cudaMemcpyHostToDevice));

	accum_from_raw(handler_id, other_grad_d);

	checkCuda(cudaFree(other_grad_d));
}

void CudaGradient::accum_from_raw(uintptr_t handler_id, float* other_grad_d)
{
	assert(grad.get());
	// TODO (I think it's just an addition?)
	if (handler_id == last_handler_id)
	{
		checkCuda(cudaMemcpy(grad.get(), other_grad_d, shape.size, cudaMemcpyDeviceToHost));
		return;
	}

	float* grad_d;

	checkCuda(cudaMalloc(&grad_d, shape.size));
	checkCuda(cudaMemcpy(grad_d, grad.get(), shape.size, cudaMemcpyHostToDevice));

	dim3 blocksPerGrid(CEILDIV(shape.H, BLCK_X), CEILDIV(shape.W, BLCK_Y), CEILDIV(shape.C, BLCK_Z));
	dim3 threadsPerBlock(BLCK_X, BLCK_Y, BLCK_Z);

	elemwiseOpKernel << <blocksPerGrid, threadsPerBlock >> > (shape.H, shape.W, shape.C,
		std::function<void(size_t, float*, float*)>(
			[](size_t ind, float* G1, float* G2)
			{
				G1[ind] += G2[ind];
			}
			),
		grad_d, other_grad_d);

	checkCuda(cudaMemcpy(grad.get(), grad_d, shape.size, cudaMemcpyDeviceToHost));
	checkCuda(cudaFree(grad_d));
}