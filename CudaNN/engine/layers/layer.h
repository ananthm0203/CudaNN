#ifndef LAYER_H
#define LAYER_H

#include "../utils/shape.h"
#include "../utils/tensor.h"

#include <memory>
#include <vector>

typedef struct Layer_GVPair
{
	float* X;
	float* G;
	Shape shape;
} Layer_GV;

class Layer
{
public:

	Layer(const Layer& other) = delete;

	virtual ~Layer() = default;

	const Shape& output_shape() const { return out.get_shape(); };

	virtual void forward() = 0;
	virtual void backwards() = 0;

	virtual std::vector<Layer_GV> get_gvs() = 0;

	void set_in_grad(Tensor* in_grad) { this->in_grad = in_grad; }

	Tensor& get_output() { return out; }

protected:

	Layer() = default;

	Tensor* in;
	Tensor out;
	Tensor out_grad;
	Tensor* in_grad;
	Shape _output_shape;
	Shape out_grad_shape;
};

#endif // LAYER_H
