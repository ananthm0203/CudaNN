#ifndef LAYER_H
#define LAYER_H
#include <memory>
#include "activations.h"

using std::unique_ptr;
using std::make_unique;

class Layer
{
public:
	explicit Layer(size_t channelIn, size_t channelOut, bool useBias = true);
	virtual ~Layer() = default;
private:
	size_t channelIn;
	size_t channelOut;
	unique_ptr<float> weights;
	unique_ptr<float> bias;
};

#endif // LAYER_H
