#ifndef ACTIVATION_H
#define ACTIVATION_H

struct Activation
{
	virtual void operator()(float* z) = 0;
	virtual void backprop(float * z) = 0;
};

struct ReLU : Activation
{
	void operator()(float* z);
	void backprop(float* z);
};


#endif // ACTIVATION_H
