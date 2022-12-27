#ifndef DENSE_H
#define DENSE_H

#include "../../activations/activations.h"
#include "../layer.h"
// #include "optimizers.h"

#include <memory>

class Dense : public Layer
{
public:

	explicit Dense(size_t C_out, Activation* activation = nullptr, bool use_bias = true)
		: activation(activation), use_bias(use_bias)
	{
		_output_shape.C = C_out;
	};

	Tensor& operator()(Tensor* in)
	{
		auto& in_shape = in->get_shape();
		assert(in_shape.H == in_shape.W == 1);
		_output_shape.H = _output_shape.W = 1;
		this->in = in;
		in->add_next_layer(this);

		// Initialize weights and biases
		weights = std::make_unique<float[]>(_output_shape.C * in_shape.C);
		w_grad = std::make_unique<float[]>(_output_shape.C * in_shape.C);
		if (use_bias)
		{
			bias = std::make_unique<float[]>(_output_shape.C);
			b_grad = std::make_unique<float[]>(_output_shape.C);
		}

		// Deal with activation
		if (activation)
		{
			activation->set_shape(_output_shape); // Assume activations do not change output_shape
			z = std::make_unique<float[]>(_output_shape.size);
			assert(activation->in_place_grad() ||
				(activation->get_grad_shape().W == activation->get_grad_shape().C == _output_shape.C));
			z = std::make_unique<float[]>(_output_shape.size);
		}
		
		// Gradients
		out_grad_shape.H = out_grad_shape.W = 1;
		out_grad_shape.C = in_shape.C;
		out_grad = Tensor(out_grad_shape);
		in->get_prev_layer()->set_in_grad(&out_grad);

		// Initialize output
		out = Tensor(_output_shape);
		out.set_prev_layer(this);
		return out;
	}

	~Dense() = default;

	// Operations
	// To be done on CUDA
	void forward()
	{
		denseForward(this->in->get_shape().C, _output_shape.C,
			this->in->raw(), weights.get(), bias.get(), z.get(), out.raw(), activation);
	}
	void backward()
	{
		denseBackprop(in->get_shape().C, _output_shape.C,
			in_grad->raw(), out_grad.raw(), z.get(), in->raw(), 
			weights.get(), bias.get(), w_grad.get(), b_grad.get(), activation);
	}

	std::vector<Layer_GV> get_gvs()
	{
		Layer_GV gv{ .X = weights.get(), .G = weights.get(),
			.shape = Shape(_output_shape.C, in->get_shape().C, 1)};
		std::vector<Layer_GV> v;
		v.emplace_back(gv);
		return v;
	}

private:

	Activation* activation;

	std::unique_ptr<float[]> weights;
	std::unique_ptr<float[]> bias;
	std::unique_ptr<float[]> z;

	std::unique_ptr<float[]> w_grad;
	std::unique_ptr<float[]> b_grad;

	bool use_bias;
};
#endif // DENSE_H

