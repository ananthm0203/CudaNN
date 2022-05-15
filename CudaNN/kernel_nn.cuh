#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <stdint.h>
#include "cuda_utils.cuh"


/*
 * Calculates the product of two matrices using the GPU
 * Inputs:
 *	A = [M x N] matrix
 *	B = [N x L] matrix
 * Outputs:
 *	C = [M x L] matrix
 */
void matMul(float* A, float* B, float* C, uint32_t M, uint32_t N, uint32_t L);
/*
 * Calculates the transpose of a matrix using GPU shared memory
 * Inputs:
 *	A = [M x N] matrix
 * Outputs:
 *	T = [N x M] matrix transpose of A
 */
void transpose(float* A, float* T, uint32_t M, uint32_t N);
/*
 * Adds matrix A to matrix B
 * Inputs:
 *  A = [M x N] matrix
 *  B = [M x N] matrix
 * Outputs:
 *  B += A
 */
void matMulAdd(float* X, float* W, float* B, float* R, uint32_t M, uint32_t N, uint32_t L, bool use_bias);

/*
 * Returns a convolved matrix given a preimage and a kernel
 * Inputs:
 *	A = [M x N] matrix
 *	K = [A x B] matrix (the kernel)
 *	padding = padding
 * Outputs:
 *	O = [L x O] matrix
 */

#endif // KERNEL_CUH
