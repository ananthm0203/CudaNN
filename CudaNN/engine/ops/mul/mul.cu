#include "mul.h"
#include "../../utils/general_kernels.cuh"
#include "../../utils/cuda_utils.cuh"

#include <cuda_runtime.h>

void MMul::forwards()
{
	float* out_d, * lhs_d, * rhs_d;

	checkCuda(cudaMalloc(&out_d, out.get_shape().size));
	checkCuda(cudaMalloc(&lhs_d, lhs->get_shape().size));
	checkCuda(cudaMalloc(&rhs_d, rhs->get_shape().size));

	checkCuda(cudaMemcpy(out_d, out.raw(), out.get_shape().size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(lhs_d, lhs->raw(), lhs->get_shape().size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(rhs_d, rhs->raw(), rhs->get_shape().size, cudaMemcpyHostToDevice));

	auto& shape = out.get_shape();

	dim3 blocksPerGrid(CEILDIV(shape.H, BLCK_X), CEILDIV(shape.W, BLCK_Y), CEILDIV(shape.W, BLCK_Z));
	dim3 threadsPerBlock(BLCK_X, BLCK_Y, BLCK_Z);

	matMulKernel << <blocksPerGrid, threadsPerBlock >> > (lhs_d, rhs_d, out_d,
		shape.H, lhs->get_shape().H, shape.W, shape.C);

	checkCuda(cudaMemcpy(out.raw(), out_d, shape.size, cudaMemcpyDeviceToHost));

	checkCuda(cudaFree(out_d));
	checkCuda(cudaFree(lhs_d));
	checkCuda(cudaFree(rhs_d));
}

void MMul::backwards()
{

	if (!lhs->from() && !rhs->from())
	{
		return;
	}

	float* grad_d;

	auto& grad = out.gradient();

	checkCuda(cudaMalloc(&grad_d, grad.get_shape().size));
	checkCuda(cudaMemcpy(grad_d, grad.raw(), grad.get_shape().size, cudaMemcpyHostToDevice));
	
	// Calc LHS grad
	if (lhs->from())
	{
		float* rhs_t_d, * rhs_d;

		checkCuda(cudaMalloc(&rhs_d, rhs->get_shape().size));
		checkCuda(cudaMalloc(&rhs_t_d, rhs->get_shape().size));

		checkCuda(cudaMemcpy(rhs_d, rhs->raw(), rhs->get_shape().size, cudaMemcpyHostToDevice));

		{
			DEFAULT_CUDA_DIMS_FROM_SHAPE(rhs->get_shape());
			transposeKernel << <blocksPerGrid, threadsPerBlock >> > (rhs_d, rhs_t_d,
				lhs->get_shape().H, lhs->get_shape().W, lhs->get_shape().C);

			checkCuda(cudaFree(rhs_d));
		}


		float* grad_lhs_d;

		checkCuda(cudaMalloc(&grad_lhs_d, lhs->get_shape().size));

		{
			DEFAULT_CUDA_DIMS_FROM_SHAPE(lhs->get_shape());
			matMulKernel << <blocksPerGrid, threadsPerBlock >> > (grad_d, rhs_t_d, grad_lhs_d,
				lhs->get_shape().H, rhs->get_shape().W, lhs->get_shape().W, lhs->get_shape().C);
		}

		checkCuda(cudaFree(rhs_t_d));

		lhs->update_gradient(this, grad_lhs_d);

		checkCuda(cudaFree(grad_lhs_d));
	}

	// Calc RHS grad
	if (rhs->from())
	{
		float* lhs_d, * lhs_t_d;

		checkCuda(cudaMalloc(&lhs_d, lhs->get_shape().size));
		checkCuda(cudaMalloc(&lhs_t_d, grad.get_shape().size));

		checkCuda(cudaMemcpy(lhs_d, lhs->raw(), lhs->get_shape().size, cudaMemcpyHostToDevice));

		{
			DEFAULT_CUDA_DIMS_FROM_SHAPE(lhs->get_shape());
			transposeKernel << <blocksPerGrid, threadsPerBlock >> > (lhs_d, lhs_t_d,
				lhs->get_shape().H, lhs->get_shape().W, lhs->get_shape().C);
		}

		checkCuda(cudaFree(lhs_d));

		float* grad_rhs_d;

		checkCuda(cudaMalloc(&grad_rhs_d, rhs->get_shape().size));

		{
			DEFAULT_CUDA_DIMS_FROM_SHAPE(rhs->get_shape());
			matMulKernel << <blocksPerGrid, threadsPerBlock >> > (lhs_t_d, grad_d, grad_rhs_d,
				rhs->get_shape().H, lhs->get_shape().H, rhs->get_shape().W, rhs->get_shape().C);
		}

		checkCuda(cudaFree(lhs_t_d));

		rhs->update_gradient(this, grad_rhs_d);

		checkCuda(cudaFree(grad_rhs_d));
	}
	
	checkCuda(cudaFree(grad_d));
}
