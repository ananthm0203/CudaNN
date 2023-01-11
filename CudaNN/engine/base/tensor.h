#pragma once

#include "shape.h"
#include "../gradient/gradient.h"
#include "../vars/initializers/initializers.h"

#include <memory>
#include <cassert>
#include <vector>
#include <unordered_set>

class Op;
class Tensor;

typedef std::shared_ptr<std::vector<Op*>> Hist_t;

class History
{
public:

	History()
		: local_hist(std::make_shared<std::vector<Op*>>())
	{
	};
	History(const History& other) = default;

	void merge_local(History& other)
	{
		auto* loc_hist_v = local_hist.get();
		auto* other_loc_hist_v = local_hist.get();
		loc_hist_v->insert(loc_hist_v->end(), other_loc_hist_v->begin(), other_loc_hist_v->end());
		other.local_hist = local_hist;
	}

	bool operator==(const History& other) { return local_hist.get() == other.local_hist.get(); }
	bool operator!=(const History& other) { return !(*this == other); }

	void add_op(Op* op) { local_hist.get()->push_back(op); }
	void add_updateable(Tensor* tensor) { updateables.get()->insert(tensor); }

private:

	Hist_t local_hist;
	std::shared_ptr<std::unordered_set<Tensor*>> updateables;
};

class Tensor
{
public:

	enum class LayerType
	{
		Input,
		Output,
		Weight
	};

	Tensor() = default;

	Tensor(const Shape& shape, Tensor::LayerType layertype)
		: shape(shape), X(std::make_shared<float[]>(shape.size)),
		grad(layertype == Tensor::LayerType::Input ? CudaGradient() : CudaGradient(shape)),
		layertype(layertype)
	{
	};

	Tensor(const Tensor& other)
		: shape(other.shape), X(std::make_shared<float[]>(shape.size)),
		grad(other.grad.empty() ? CudaGradient() : CudaGradient(shape)),
		layertype(other.layertype)
	{
		memcpy(X.get(), other.X.get(), shape.size);
		if (!other.grad.empty())
		{
			other.grad.copy_grad_to(grad);
		}
	};

	Tensor& operator=(const Tensor& other)
	{
		shape = other.shape;
		X = std::make_shared<float[]>(shape.size);
		grad = other.grad.empty() ? CudaGradient() : CudaGradient(shape);
		layertype = other.layertype;
		memcpy(X.get(), other.X.get(), shape.size);
		if (!other.grad.empty())
		{
			other.grad.copy_grad_to(grad);
		}
		return *this;
	}

	Tensor& operator=(Tensor&& other) = default;

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

	Op* from() { return _from; }
	void set_from(Op* op) { _from = op; }

	CudaGradient& gradient() { return grad; }
	void update_gradient(Op* op, const CudaGradient& other_grad)
	{
		if (grad.raw())
		{
			grad.accum_from_wrapped(reinterpret_cast<uintptr_t>(op), other_grad);
		}
	}
	void update_gradient(Op* op, float* other_grad_d)
	{
		if (grad.raw())
		{
			grad.accum_from_raw(reinterpret_cast<uintptr_t>(op), other_grad_d);
		}
	}
	bool frozen() { return grad.empty(); }
	bool updateable() { return layertype == Tensor::LayerType::Weight; }
	/*void freeze() { no_grad = true; }
	void unfreeze() { no_grad = false; }*/

	void initialize(Initializer& init)
	{
		if (layertype != LayerType::Weight || initialized)
		{
			return;
		}
		for (size_t i = 0; i < shape.size; ++i)
		{
			init(X.get(), i);
		}
		initialized = true;
	}

private:

	Shape shape;
	std::shared_ptr<float[]> X;

	Op* _from;
	CudaGradient grad;
	Tensor::LayerType layertype;

	bool initialized = false;
};
