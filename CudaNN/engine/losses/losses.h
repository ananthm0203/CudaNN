#pragma once

#include "../utils/tensor.h"

class Loss
{
public:
	virtual float operator()(Tensor& y_true, Tensor& y_pred) = 0;
	virtual Tensor grad(Tensor& y_true, Tensor& y_pred) = 0;
};

class CrossEntropy : public Loss
{
public:
	float operator()(Tensor& y_true, Tensor& y_pred)
	{
		assert(y_true.get_shape() == y_pred.get_shape() &&
			y_true.get_shape().H == y_true.get_shape().W == 1 &&
			y_true.get_shape().C <= 1024);
		return cuda_crossentropy(y_true, y_pred);
	}

	Tensor grad(Tensor& y_true, Tensor& y_pred)
	{
		assert(y_true.get_shape() == y_pred.get_shape() &&
			y_true.get_shape().H == y_true.get_shape().W == 1 &&
			y_true.get_shape().C <= 1024);
		return cuda_crossentropy_grad(y_true, y_pred);
	}

private:

	float cuda_crossentropy(Tensor& y_true, Tensor& y_pred);
	Tensor cuda_crossentropy_grad(Tensor& y_true, Tensor& y_pred);
};
