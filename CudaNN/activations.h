#ifndef ACTIVATION_H
#define ACTIVATION_H

struct Activation
{
	virtual float* operator()(float* z) = 0;
	virtual float* backprop(float * z) = 0;
};

struct ReLU
{

};


#endif // ACTIVATION_H
