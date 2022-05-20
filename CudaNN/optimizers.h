#pragma once

#include <memory>
#include <vector>

using std::unique_ptr;
using std::make_unique;
using std::vector;

typedef struct GWPair
{
	float* weights; // Weights is going to be owned by a Layer, and the ops we're performing *should* be safe
	unique_ptr<float> gradient;
	size_t M;
	size_t N;
} GWPair;

class Optimizer
{
public:
	Optimizer(float lr) : lr(lr) {};
	virtual ~Optimizer() = default;

protected:
	float lr;
};

class Adam : Optimizer
{
public:
	Adam(size_t num_grads, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8) :
		Optimizer(lr), beta1(beta1), beta2(beta2), epsilon(epsilon), m_vector{}, v_vector{}
	{};
	~Adam() = default;
	void add_gradient(size_t grad_r, size_t grad_c);
	void update(vector<GWPair> gwpairs, size_t timestep);

private:
	float beta1;
	float beta2;
	float epsilon;
	vector<unique_ptr<float>> m_vector;
	vector<unique_ptr<float>> v_vector;
};

