#pragma once

#include "base/tensor.h"
#include "optimizers/optimizers.h"
#include "ops/op.h"

#include <memory>
#include <vector>
#include <queue>

class Network
{
public:

	Network(Op* in, Optimizer* optim, size_t batch_size)
		: optim(optim)
	{
	};



private:

	Optimizer* optim;
}

