#pragma once

struct Shape
{

	Shape() : H(0), W(0), C(0), size(0) {};

	Shape(size_t H, size_t W, size_t C = 1) : H(H), W(W), C(C), size(H * W * C) {};

	bool operator==(const Shape& other) const
	{
		return H == other.H && W == other.W && C == other.C;
	}

	size_t H, W, C, size;
};
