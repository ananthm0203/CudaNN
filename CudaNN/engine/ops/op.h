#pragma once

#include "../base/shape.h"
#include "../base/tensor.h"
#include "../gradient/gradient.h"

#include <type_traits>
#include <unordered_map>

class Op
{
public:

	Op(const Op& other) = delete;

	virtual ~Op() = default;

	virtual void forward() = 0;
	virtual void backwards() = 0;

protected:

	void handle_input_history(History& other)
	{
		if (h != other)
		{
			h.merge_local(other);
		}
	}

	void handle_input(Tensor* input)
	{
		handle_input_history(input->from()->h); // Handle history
		if (input->updateable())
		{
			h.add_updateable(input);
		}
		input->gradient().set_first_backwards(reinterpret_cast<uintptr_t>(this));
	}

	template<typename... T, typename = typename std::enable_if<
		(true && ... && std::is_same_v<T, Tensor*>), void>::type >
	void handle_inputs(T... inputs)
	{
		Tensor* input_arr[] = { inputs... };
		for (size_t i = 0; i < sizeof...(inputs); ++i)
		{
			handle_input(static_cast<Tensor*>(input_arr[i]));
		}
		h.add_op(this);
	}

	void handle_inputs(std::vector<Tensor*> inputs)
	{
		for (auto& input : inputs)
		{
			handle_input(input);
		}
		h.add_op(this);
	}

	//template<typename... T, typename = typename std::enable_if<
	//	(true && ... && std::is_same_v<T, Tensor*>), void>::type >
	//	void handle_outputs(T... outputs)
	//{
	//	Tensor* output_arr[] = { outputs... };
	//	for (size_t i = 0; i < sizeof...(outputs); ++i)
	//	{
	//		Tensor* output = output_arr[i];
	//		// I *think* this is a reasonable assumption?
	//		in_grads.insert(std::make_pair<Tensor*, CudaGradient>(output, CudaGradient(output->shape)));
	//	}
	//}

	//std::vector<CudaGradientHandler> out_grads; // Will be stored in correspondence with the inputs
	//std::unordered_map<Tensor*, CudaGradient> in_grads;
	History h;
};