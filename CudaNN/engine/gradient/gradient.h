#pragma once

#include "../base/shape.h"

#include <memory>
#include <cassert>

class CudaGradient
{
public:

	CudaGradient() : shape(0, 0, 0), grad(nullptr) {};
	CudaGradient(const Shape& shape)
		: shape(shape), grad(std::make_unique<float[]>(shape.size)),
		last_handler_id(0)
	{
	};

	void accum_from_raw(uintptr_t handler_id, float* other_grad_d);
	void accum_from_wrapped(uintptr_t handler_id, const CudaGradient& grad);
	void set_first_backwards(uintptr_t id) { last_handler_id = id; }
	float* raw() { return grad.get(); }
	void copy_grad_to(CudaGradient& other_grad) const
	{
		assert(grad.get() && other_grad.grad.get() && shape == other_grad.shape);
		memcpy(other_grad.grad.get(), grad.get(), shape.size);
	}
	const Shape& get_shape() { return shape; }
	bool empty() const { return static_cast<bool>(grad.get()); }

private:

	Shape shape;
	std::unique_ptr<float[]> grad;
	uintptr_t last_handler_id;
};

// TODO: Separate into own classes. Then, add subtypes for normal, CUDA, and no-calc
class CudaGradientHandler
{
public:

	CudaGradientHandler() = delete;
	CudaGradientHandler(CudaGradient* dest) : grad_ptr(dest)
	{
		if (dest)
		{
			dest->set_first_backwards(reinterpret_cast<uintptr_t>(this));
		}
	}

	void flow_back(float* grad_d)
	{
		if (grad_ptr)
		{
			grad_ptr->accum_from_raw(reinterpret_cast<uintptr_t>(this), grad_d);
		}
	}

	void no_calc_flow_back(const CudaGradient& grad)
	{
		if (grad_ptr)
		{
			grad_ptr->accum_from_wrapped(reinterpret_cast<uintptr_t>(this), grad);
		}
	}

	bool null_grad_ptr()
	{
		return static_cast<bool>(grad_ptr);
	}

private:
	CudaGradient* grad_ptr;
};
