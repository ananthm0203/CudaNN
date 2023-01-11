#pragma once

#include <random>
#include <cmath>

typedef struct Initializer
{
	virtual void operator()(float* raw, size_t ind) = 0;
} Initializer;

typedef struct ZeroInitializer
	: public Initializer
{
	void operator()(float* raw, size_t ind)
	{
		raw[ind] = 0;
	}

} ZeroInitializer;

typedef struct NormalInitializer
	: public Initializer
{
	NormalInitializer(float loc = 0.0, float scale = 1.0)
		: normal_distribution(loc, scale)
	{
	};

	void operator()(float* raw, size_t ind)
	{
		raw[ind] = normal_distribution(generator);
	}

private:

	std::default_random_engine generator;
	std::normal_distribution<float> normal_distribution;

} NormalInitializer;

typedef struct UniformInitializer
	: public Initializer
{
	UniformInitializer(float a = 0.0, float b = 1.0)
		: uniform_distribution(a, b)
	{
	};

	void operator()(float* raw, size_t ind)
	{
		raw[ind] = uniform_distribution(generator);
	}

private:

	std::default_random_engine generator;
	std::uniform_real_distribution<float> uniform_distribution;

} UniformInitializer;
