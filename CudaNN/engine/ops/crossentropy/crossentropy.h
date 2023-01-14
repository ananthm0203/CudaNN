#pragma once

#include "../op.h"

class CrossEntropy
	: public Op
{
public:

	Tensor* operator()(Tensor* y_pred, Tensor* y_true)
	{
		assert(y_pred->get_shape() == y_true->get_shape());
		assert(y_pred->get_shape().H == y_pred->get_shape().W == 1 && y_pred->get_shape().W <= 1024);
		handle_inputs(y_pred, y_true);
		this->y_pred = y_pred;
		this->y_true = y_true;
		out = Tensor(Shape(1, 1, 1), Tensor::LayerType::Output);
		return &out;
	}

	void forwards();
	void backwards();

protected:

	Tensor* y_pred, * y_true;
	Tensor out;
};


class SoftmaxCrossEntropyWithLogits
	: public Op
{
public:

	Tensor* operator()(Tensor* y_pred, Tensor* y_true)
	{
		assert(y_pred->get_shape() == y_true->get_shape());
		assert(y_pred->get_shape().H == y_pred->get_shape().W == 1 && y_pred->get_shape().W <= 1024);
		handle_inputs(y_pred, y_true);
		this->y_pred = y_pred;
		this->y_true = y_true;
		softmax = Tensor(y_pred->get_shape(), Tensor::LayerType::Input);
		out = Tensor(Shape(1, 1, 1), Tensor::LayerType::Output);
		return &out;
	}

	void forwards();
	void backwards();

private:

	Tensor* y_pred, * y_true;
	Tensor out, softmax;
};
