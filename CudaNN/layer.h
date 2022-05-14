#ifndef LAYER_H
#define LAYER_H
#include <memory>

using std::unique_ptr;
using std::make_unique;

class Layer
{
public:
	explicit Layer(size_t channelIn, size_t channelOut, bool useBias=true);
	virtual ~Layer() = default;
	virtual unique_ptr<float> forward() = 0;
	virtual unique_ptr<float> backwards() = 0;
private:
	size_t channelIn;
	size_t channelOut;
	unique_ptr<float> weights;
	unique_ptr<float> bias;
};

#endif // LAYER_H
