#pragma once

#include "../utils/shape.h"

#include <memory>
#include <vector>
#include <unordered_set>

class GV
{
public:
	GV(float* X, float* G, const Shape& shape)
		: X(X), G(G), M(std::make_unique<float[]>(shape.size)),
		V(std::make_unique<float[]>(shape.size)), shape(shape)
	{
	};

	float* X;
	float* G;
	std::unique_ptr<float[]> M;
	std::unique_ptr<float[]> V;
	Shape shape;
};

class Optimizer
{
public:
	Optimizer(float lr) : lr(lr) {};
	virtual ~Optimizer() = default;

	virtual void add_gradient(float* X, float* G, const Shape& shape)
	{
		gvs.emplace(GV(X, G, shape));
	}
	virtual void update() = 0;

protected:
	float lr;
	std::unordered_set<GV> gvs;
};

class Adam : Optimizer
{
public:
	Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8)
		: Optimizer(lr), beta1(beta1), beta2(beta2), epsilon(epsilon), timestep(0)
	{
	};

	void update()
	{
		for (const auto& gv : gvs)
		{
			cuda_adam_update(gv, beta1, beta2, epsilon, timestep, lr);
		}
		++timestep;
	}

private:
	float beta1;
	float beta2;
	float epsilon;
	float timestep;

	void cuda_adam_update(const GV& gv, float beta1, float beta2, float epsilon, float timestep, float lr);
};

