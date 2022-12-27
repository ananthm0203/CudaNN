#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "../utils/cuda_utils.cuh"
#include "activations.cuh"
#include "../utils/shape.h"

#include <vector>

class Activation
{
public:

	virtual void forward(float* z) = 0;
	virtual void backprop(float* z, float* z_dest) = 0;
	virtual void set_shape(const Shape& shape)
	{
		this->shape = shape;
		this->grad_shape = shape;
		grad_same_shape = true;
	}

	const Shape& get_shape() const { return shape; }
	const Shape& get_grad_shape() const { return grad_shape; }
	const bool in_place_grad() const { return grad_same_shape; }

protected:

	Activation(bool grad_same_shape) : shape(0, 0, 0), grad_shape(0, 0, 0),
		grad_same_shape(grad_same_shape) {};

	Shape shape;
	Shape grad_shape;
	bool grad_same_shape;
};

class ReLU : public Activation
{
public:

	ReLU() : Activation(true) {};

	void forward(float* z)
	{
		dim3 blocksPerGrid(CEILDIV(shape.H, BLCK_X), CEILDIV(shape.W, BLCK_Y), CEILDIV(shape.C, BLCK_Z));
		dim3 threadsPerBlock(BLCK_X, BLCK_Y, BLCK_Z);
		reluKernel << <blocksPerGrid, threadsPerBlock >> > (z, shape.H, shape.W, shape.C);
	}
	void backprop(float* z, float* z_dest)
	{
		dim3 blocksPerGrid(CEILDIV(shape.H, BLCK_X), CEILDIV(shape.W, BLCK_Y), CEILDIV(shape.C, BLCK_Z));
		dim3 threadsPerBlock(BLCK_X, BLCK_Y, BLCK_Z);
		reluBackpropKernel << <blocksPerGrid, threadsPerBlock >> > (z, shape.H, shape.W, shape.C);
	}
};

class Softmax : public Activation
{
public:

	Softmax() : Activation(false) {};

	void set_shape(const Shape& shape)
	{
		assert(shape.W == shape.H == 1 && shape.C <= 1024);
		this->shape = shape;
		this->grad_shape = Shape(1, shape.C, shape.C);
	}

	void forward(float* z)
	{
		dim3 blocksPerGrid(CEILDIV(shape.C, warpSize) * warpSize);
		dim3 threadsPerBlock(warpSize);
		softmax1DKernel1024M << <blocksPerGrid, threadsPerBlock >> > (z, shape.C);
	}
	void backprop(float* z, float* z_dest)
	{
		dim3 blocksPerGrid(CEILDIV(shape.C, warpSize) * warpSize);
		dim3 threadsPerBlock(warpSize);
		softmax1DBackpropKernel1024M << <blocksPerGrid, threadsPerBlock >> > (z, z_dest, shape.C);
	}
};

#endif // ACTIVATIONS_H
