#include "optimizers.h"
#include "../utils/cuda_utils.cuh"
#include "../utils/general_kernels.cuh"

#include <cuda_runtime.h>

__device__ void adam_update_inner(float* X, float* M, float* V, float lr, float eps, const Shape& shape)
{
	auto tx = threadIdx.x;
	auto ty = threadIdx.y;
	auto tz = threadIdx.z;
	auto bx = blockIdx.x;
	auto by = blockIdx.y;
	auto bz = blockIdx.z;

	size_t COL = bx * blockDim.x + tx;
	size_t ROW = by * blockDim.y + ty;
	size_t AISLE = bz * blockDim.z + bz;

	if (COL < shape.H && ROW < shape.W && AISLE < shape.C)
	{
		auto idx = ROW * shape.W * shape.C + COL * shape.C + AISLE;
		X[idx] -= lr / sqrtf(V[idx]) * M[idx];
	}
}

__global__ void Adam::cuda_adam_update(const GV& gv, float beta1, float beta2, float epsilon, float timestep, float lr)
{
	float* grad_d;
	float* M_d;
	float* V_d;

	checkCuda(cudaMalloc(&grad_d, gv.shape.size));
	checkCuda(cudaMalloc(&M_d, gv.shape.size));
	checkCuda(cudaMalloc(&V_d, gv.shape.size));

	checkCuda(cudaMemcpy(&grad_d, gv.G, gv.shape.size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(&M_d, gv.M.get(), gv.shape.size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(&V_d, gv.V.get(), gv.shape.size, cudaMemcpyHostToDevice));

	dim3 blocksPerGrid(TILENO(gv.shape.H), TILENO(gv.shape.C));
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);

	elemwiseOpKernel<SumOp, MulOp> <<<blocksPerGrid, threadsPerBlock>>>(beta1, M_d, 1 - beta1, grad_d, M_d, gv.shape.H, gv.shape.W, gv.shape.C);
	elemwiseOpKernel<MulOp> << <blocksPerGrid, threadsPerBlock >> > (grad_d, grad_d, gv.shape.H, gv.shape.W, gv.shape.C);
	elemwiseOpKernel<SumOp, MulOp> << <blocksPerGrid, threadsPerBlock >> > (beta2, V_d, 1 - beta2, grad_d, V_d, gv.shape.H, gv.shape.W, gv.shape.C);

	elemwiseOpKernel<MulOp> << <blocksPerGrid, threadsPerBlock >> > (1 / (1 - pow(beta1, timestep)), M_d, M_d, gv.shape.H, gv.shape.W, gv.shape.C);
	elemwiseOpKernel<MulOp> << <blocksPerGrid, threadsPerBlock >> > (1 / (1 - pow(beta2, timestep)), V_d, V_d, gv.shape.H, gv.shape.W, gv.shape.C);

	checkCuda(cudaFree(grad_d));

	float* X_d;
	
	checkCuda(cudaMalloc(&X_d, gv.shape.size));

	checkCuda(cudaMemcpy(&X_d, gv.X, gv.shape.size, cudaMemcpyHostToDevice));

	adam_update_inner << <blocksPerGrid, threadsPerBlock >> > (X_d, M_d, V_d, lr, epsilon, gv.shape);

	checkCuda(cudaMemcpy(gv.M.get(), M_d, gv.shape.size, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(gv.V.get(), V_d, gv.shape.size, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(gv.X, X_d, gv.shape.size, cudaMemcpyDeviceToHost));

	checkCuda(cudaFree(M_d));
	checkCuda(cudaFree(V_d));
	checkCuda(cudaFree(X_d));
}

