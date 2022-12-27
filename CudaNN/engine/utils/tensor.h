#pragma once

#include "shape.h"
#include "general_kernels.cuh"

#include <memory>
#include <cassert>
#include <vector>

class Layer;

class Tensor
{
public:

	Tensor() = default;

	Tensor(const Shape& shape) : shape(shape), X(std::make_shared<float[]>(shape.size)) {};

	Tensor(size_t H, size_t W, size_t C)
		: shape(H, W, C), X(std::make_shared<float[]>(W * H * C)) {};

	Tensor(const Tensor& other) : shape(other.shape), X(other.X) {};

	Tensor(const Tensor& other, size_t H, size_t W, size_t C) : shape(H, W, C), X(other.X)
	{
		assert(shape.size == other.shape.size);
	}

	//Tensor& operator+(const Tensor& other)
	//{
	//	assert(shape == other.shape);
	//	elemwiseOpKernel<SumOp>(other.X.get(), X.get(), X.get(), shape.H, shape.W, shape.C);
	//	return *this;
	//}

	//// Elementwise Multiplication
	//Tensor& operator*(const Tensor& other)
	//{
	//	assert(shape == other.shape);
	//	elemwiseOpKernel<MulOp>(other.X.get(), X.get(), X.get(), shape.H, shape.W, shape.C);
	//	return *this;
	//}

	//// Matrix Multiplication
	//Tensor matmul(const Tensor& other)
	//{
	//	assert(shape.W == other.shape.H && shape.C == other.shape.C);
	//	Tensor res(shape.H, other.shape.W, shape.C);
	//	matMulKernel(X.get(), other.X.get(), res.X.get(), shape.H, shape.W, other.shape.W, shape.C);
	//	return res;
	//}

	// TODO: Convolution

	const Shape& get_shape() const { return shape; }

	float* raw() { return X.get(); }
	void copy_to(float* buf) { memcpy(buf, X.get(), shape.size); }

	Layer* get_prev_layer() const { return prev_layer; }
	void set_prev_layer(Layer* layer) { prev_layer = layer; }

	const std::vector<Layer*>& get_next_layers() { return next_layers; }
	void add_next_layer(Layer* layer) { next_layers.push_back(layer); }

private:

	Shape shape;
	std::shared_ptr<float[]> X;

	Layer* prev_layer;
	std::vector<Layer*> next_layers;
};
