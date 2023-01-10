#include "add.h"

void Add::forward()
{
	auto& shape = out.get_shape();

	float* out_d;
	float* in_d;

	checkCuda(cudaMalloc(&out_d, shape.size));
	checkCuda(cudaMalloc(&in_d, shape.size));

	checkCuda(cudaMemcpy(out_d, ins[0]->raw(), shape.size, cudaMemcpyHostToDevice));

	DEFAULT_CUDA_DIMS_FROM_SHAPE(shape);

	for (size_t i = 1; i < ins.size(); ++i)
	{
		cudaMemcpy(in_d, ins[i]->raw(), shape.size, cudaMemcpyHostToDevice);
		elemwiseOpKernel << <blocksPerGrid, threadsPerBlock >> > (shape.H, shape.W, shape.C,
			std::function<void(size_t, float*, float*)>(
				[](size_t ind, float* G1, float* G2)
				{
					G1[ind] += G2[ind];
				}
				),
			out_d, in_d);
	}

	checkCuda(cudaFree(in_d));
	checkCuda(cudaFree(out_d));
}
