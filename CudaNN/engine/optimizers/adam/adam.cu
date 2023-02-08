#include "adam.h"
#include "../optimizers.h"
#include "../../utils/cuda_utils.cuh"
#include "../../utils/general_kernels.cuh"

__global__ void Adam::update()
{
	for (size_t i = 0; i < weights.size(); ++i)
	{
		auto& wgp = weights[i];
		auto& mvp = mv_pairs[i];

		auto& weight = wgp.first;
		auto& grad = wgp.second;
		auto& shape = grad.get_shape();

		float* grad_d;
		float* M_d;
		float* V_d;

		checkCuda(cudaMalloc(&grad_d, shape.size));
		checkCuda(cudaMalloc(&M_d, shape.size));
		checkCuda(cudaMalloc(&V_d, shape.size));

		checkCuda(cudaMemcpy(grad_d, grad.raw(), shape.size, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(M_d, mvp.first.raw(), shape.size, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(V_d, mvp.second.raw(), shape.size, cudaMemcpyHostToDevice));

		DEFAULT_CUDA_DIMS_FROM_SHAPE(shape);

		auto f1(static_cast<std::function<void(size_t, float*, float*)>>(
			[&](size_t ind, float* M_d, float* grad) { M_d[ind] = beta1 * M_d[ind] + (1 - beta1) * grad[ind]; }
		));
		elemwiseOpKernel << <blocksPerGrid, threadsPerBlock >> > (shape.H, shape.W, shape.C, f1, M_d, grad_d);

		auto f2(static_cast<std::function<void(size_t, float*, float*)>>(
			[&](size_t ind, float* V_d, float* grad) { V_d[ind] = beta2 * V_d[ind] + (1 - beta2) * grad[ind] * grad[ind]; }
		));
		elemwiseOpKernel << <blocksPerGrid, threadsPerBlock >> > (shape.H, shape.W, shape.C, f2, V_d, grad_d);

		float* weight_d = grad_d;

		checkCuda(cudaMemcpy(weight_d, weight, shape.size, cudaMemcpyHostToDevice));

		auto f3(static_cast<std::function<void(size_t, float*, float*, float*)>>(
			[&](size_t ind, float* M_d, float* V_d, float* W_d)
			{
				float beta1_t = std::powf(beta1, timestep);
		float beta2_t = std::powf(beta2, timestep);
		float M_hat_d = M_d[ind] / (1 - beta1_t);
		float V_hat_d = V_d[ind] / (1 - beta2_t);
		W_d[ind] -= (lr * M_hat_d) / (std::sqrtf(V_hat_d) + epsilon);
			}
		));
		elemwiseOpKernel << <blocksPerGrid, threadsPerBlock >> > (shape.H, shape.W, shape.C, f3, M_d, V_d, weight_d);

		checkCuda(cudaMemcpy(weight->raw(), weight_d, shape.size, cudaMemcpyDeviceToHost));
		checkCuda(cudaMemcpy(mvp.first.raw(), M_d, shape.size, cudaMemcpyDeviceToHost));
		checkCuda(cudaMemcpy(mvp.second.raw(), V_d, shape.size, cudaMemcpyDeviceToHost));

		checkCuda(cudaFree(weight_d));
		checkCuda(cudaFree(M_d));
		checkCuda(cudaFree(V_d));
	}
}