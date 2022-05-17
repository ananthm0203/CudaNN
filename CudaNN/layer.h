#ifndef LAYER_H
#define LAYER_H
#include <memory>
#include "activations.h"

using std::unique_ptr;
using std::make_unique;

class Layer
{
public:
	explicit Layer(size_t channelIn, size_t channelOut, bool useBias = true, unique_ptr<Activation> activation = nullptr);
	virtual ~Layer() = default;
private:
	size_t channelIn;
	size_t channelOut;
	unique_ptr<float> weights;
	unique_ptr<float> bias;
	unique_ptr<Activation> activation;
};

#endif // LAYER_H
