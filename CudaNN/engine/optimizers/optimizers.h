#pragma once

#include "../base/tensor.h"

#include <memory>
#include <vector>
#include <unordered_set>

typedef std::pair<Tensor*, Tensor> WGPair;

class Optimizer
{
public:

	Optimizer(float lr) : lr(lr) {};
	virtual ~Optimizer() = default;

	virtual void add_weight(Tensor* tensor)
	{
		weights.push_back(std::make_pair(tensor, Tensor(tensor->get_shape(), Tensor::LayerType::Input)));
	}

	void set_batch_size(size_t batch_size)
	{
		this->batch_size = batch_size;
	}

	void update_grads();
	void reset_grads()
	{
		for (auto& wgp : weights)
		{
			memset(wgp.second.raw(), 0, wgp.second.get_shape().size);
		}
	}

	virtual void update() = 0;

protected:

	float lr;
	size_t batch_size;
	std::vector<WGPair> weights;
};
