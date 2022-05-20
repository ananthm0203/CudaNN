#include "optimizers.h"
#include "cuda_utils.cuh"
#include "kernel_nn.cuh"
#include <cmath>


void Adam::add_gradient(size_t grad_r, size_t grad_c)
{
	m_vector.emplace_back(std::unique_ptr<float[]>(new float[grad_r * grad_c]));
	v_vector.emplace_back(std::unique_ptr<float[]>(new float[grad_r * grad_c]));
}
void Adam::update(vector<GWPair> gwpairs, size_t timestep)
{
	GWPair *gwpair;
	double alpha;
	for (size_t i = 0; i < gwpairs.size(); ++i)
	{
		// NOTE: This entire operation can be heavily optimized, mostly by doing calculations
		// w.r.t. the gradient in one code segment to minimize the amount of allocations and frees
		
		// Update biased first moment //
		gwpair = &gwpairs[i];
		float* grad_d;
		float* grad_d_prod;
		auto grad_memsize = gwpair->M * gwpair->N;
		dim3 blocksPerGrid((gwpair->M - 1) / TILE_WIDTH + 1, (gwpair->N - 1) / TILE_WIDTH + 1);
		dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
		// (1 - B_1) * gradient
		checkCuda(cudaMalloc(&grad_d, grad_memsize));
		checkCuda(cudaMalloc(&grad_d_prod, grad_memsize));
		checkCuda(cudaMemcpy(grad_d, gwpair->gradient.release(), grad_memsize, cudaMemcpyHostToDevice));
		elemwiseOpKernel <MulOp, float> << <blocksPerGrid, threadsPerBlock >> > (1 - beta1, grad_d, grad_d_prod, gwpair->M, gwpair->N);
		//checkCuda(cudaFree(&grad_d)); // To be freed later
		// B_1 * 1st moment
		float* m_d;
		checkCuda(cudaMalloc(&m_d, grad_memsize));
		checkCuda(cudaMemcpy(m_d, m_vector[i].release(), grad_memsize, cudaMemcpyHostToDevice));
		elemwiseOpKernel <MulOp, float> << <blocksPerGrid, threadsPerBlock >> > (beta1, m_d, m_d, gwpair->M, gwpair->N);
		// 1st moment <- (B_1 * 1st moment) + ((1 - B_1) * gradient)
		elemwiseOpKernel <SumOp, float> << <blocksPerGrid, threadsPerBlock >> > (grad_d_prod, m_d, m_d, gwpair->M, gwpair->N);
		checkCuda(cudaMemcpy(m_vector[i].release(), m_d, grad_memsize, cudaMemcpyDeviceToHost));
		checkCuda(cudaFree(m_d));
		//checkCuda(cudaFree(&grad_d_prod)); // To be freed later

		// Update raw, biased second moment //
		// gradient ^ 2
		elemwiseOpKernel <PowOp, float> << <blocksPerGrid, threadsPerBlock >> > (2, grad_d, grad_d_prod, gwpair->M, gwpair->N);
		// (1 - B_2) * (gradient ^ 2)
		elemwiseOpKernel <MulOp, float> << <blocksPerGrid, threadsPerBlock >> > (1 - beta2, grad_d, grad_d_prod, gwpair->M, gwpair->N);
		checkCuda(cudaFree(&grad_d));
		// B_2 * 2nd moment
		float* v_d;
		checkCuda(cudaMalloc(&v_d, grad_memsize));
		checkCuda(cudaMemcpy(v_d, v_vector[i].release(), grad_memsize, cudaMemcpyHostToDevice));
		elemwiseOpKernel <MulOp, float> << <blocksPerGrid, threadsPerBlock >> > (beta1, v_d, v_d, gwpair->M, gwpair->N);
		// 1st moment <- (B_2 * 2nd moment) + ((1 - B_2) * (gradient ^ 2))
		elemwiseOpKernel <SumOp, float> << <blocksPerGrid, threadsPerBlock >> > (grad_d_prod, v_d, v_d, gwpair->M, gwpair->N);
		checkCuda(cudaMemcpy(v_vector[i].release(), v_d, grad_memsize, cudaMemcpyDeviceToHost));
		//checkCuda(cudaFree(&v_d)); // Will be freed later
		checkCuda(cudaFree(grad_d_prod));

		// Calculate biased-correction coefficient alpha //
		alpha = lr * std::sqrt(1 - std::powf(beta2, timestep)) / (1 - std::powf(beta1, timestep));

		// Update weights //
		// sqrt(2nd moment)
		float* v_sqrt_d;
		checkCuda(cudaMalloc(&v_sqrt_d, grad_memsize));
		elemwiseOpKernel <PowOp, float> << <blocksPerGrid, threadsPerBlock >> > (0.5, v_sqrt_d, gwpair->M, gwpair->N);
		checkCuda(cudaFree(v_d));
		// sqrt(2nd moment) + epsilon
		elemwiseOpKernel <SumOp, float> << <blocksPerGrid, threadsPerBlock >> > (epsilon, v_sqrt_d, gwpair->M, gwpair->N);
		// 1st moment / (sqrt(2nd moment) + epsilon)
		// NOTE: It might be possible to make this allocation non-redundant, but I don't care for now
		checkCuda(cudaMalloc(&m_d, grad_memsize));
		checkCuda(cudaMemcpy(m_d, m_vector[i].release(), grad_memsize, cudaMemcpyHostToDevice));
		elemwiseOpKernel <DivOp, float> << <blocksPerGrid, threadsPerBlock >> > (m_d, v_sqrt_d, m_d, gwpair->M, gwpair->N);
		checkCuda(cudaFree(&v_sqrt_d));
		// -alpha * (1st moment / (sqrt(2nd moment) + epsilon))
		elemwiseOpKernel <MulOp, float> << <blocksPerGrid, threadsPerBlock >> > (-alpha, m_d, gwpair->M, gwpair->N);
		// weights = weights - alpha * (1st moment / (sqrt(2nd moment) + epsilon))
		float* weights_d;
		checkCuda(cudaMalloc(&weights_d, grad_memsize));
		checkCuda(cudaMemcpy(weights_d, gwpair->weights, grad_memsize, cudaMemcpyHostToDevice));
		elemwiseOpKernel <SumOp, float> << <blocksPerGrid, threadsPerBlock >> > (m_d, weights_d, gwpair->M, gwpair->N);
		checkCuda(cudaMemcpy(gwpair->weights, weights_d, grad_memsize, cudaMemcpyDeviceToHost));
		checkCuda(cudaFree(weights_d));
		checkCuda(cudaFree(m_d));
	}
}