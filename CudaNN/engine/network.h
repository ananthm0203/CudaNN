#ifndef NETWORK_H
#define NETWORK_H

#include "layers/layer.h"
#include "optimizers/optimizers.h"
#include "losses/losses.h"

#include <memory>
#include <vector>
#include <queue>

class Network
{
public:
	
	Network(Tensor* in, Tensor* out, Tensor* target, Loss* loss, Optimizer* optim)
		: out(out), target(target), optim(optim)
	{
		assert(out->get_shape() == target->get_shape());
		out->get_prev_layer()->set_in_grad(&this->loss_grad);
		for (auto& i : in->get_next_layers())
		{
			q.push(i);
		}
		while (!q.empty())
		{
			auto l = q.front();
			for (auto& gv : l->get_gvs())
			{
				optim->add_gradient(gv.X, gv.G, gv.shape);
			}
			for (auto nl : l->get_output().get_next_layers())
			{
				q.push(nl);
			}
		}
	}

	void forward()
	{
		q.push(in);
		while (!q.empty())
		{
			auto l = q.front();
			l->forward();
			for (auto nl : l->get_output().get_next_layers())
			{
				q.push(nl);
			}
		}
	}

	void backward()
	{
		/*float l = loss->operator()(*target, *out);
		std::cout << "loss: " << l << std::endl;*/
		loss_grad = loss->grad(*target, *out);
		optim->update();
	}

private:

	Layer* in;
	Tensor* out;
	Tensor* target;
	Loss* loss;
	Tensor loss_grad;
	Optimizer* optim;
	std::queue<Layer*> q;
};

#endif //NETWORK_H

