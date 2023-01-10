#pragma once

#include "../op.h"

class MMul : public Op
{
public:

	Tensor* operator()(Tensor* lhs, Tensor* rhs)
	{
		assert(lhs->get_shape().maligned(rhs->get_shape()));
		handle_inputs(lhs, rhs);
		this->lhs = lhs;
		this->rhs = rhs;
		out = Tensor(Shape(lhs->get_shape().H, rhs->get_shape().W, lhs->get_shape().C),
			Tensor::LayerType::Output);
		return &out;
	}

	void forward();
	void backwards();

private:

	Tensor* lhs, * rhs;
	Tensor out;
};

//class EMul : public Op
//{
//public:
//
//	Tensor& operator()(Tensor* lhs, Tensor* rhs)
//	{
//		assert(lhs->get_shape().maligned(rhs->get_shape()));
//		handle_inputs(lhs, rhs);
//		this->lhs = lhs;
//		this->rhs = rhs;
//		out = Tensor(lhs->get_shape().H, rhs->get_shape().W, lhs->get_shape().C);
//		return out;
//	}
//
//	void forward();
//	void backwards();
//
//private:
//
//	Tensor* lhs, * rhs;
//	Tensor out;
//};
