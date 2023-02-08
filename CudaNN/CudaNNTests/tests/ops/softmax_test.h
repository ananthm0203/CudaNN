#pragma once

#include "../../../engine/ops/softmax/softmax.h"
#include "../..//../engine/vars/input.h"
#include "../../gtest/gtest.h"

#include <iostream>
#include <vector>

namespace
{
	class SoftmaxTest : public ::testing::Test
	{
	protected:
		/*Softmax softmax;
		Tensor target, input;*/
		std::vector<std::vector<float>> raw_ins, targets, grads;

		virtual void SetUp()
		{
			auto& raw_in_0 = raw_ins[0];
			auto& target_0 = targets[0];
			auto& grads_0 = grads[0];

			constexpr size_t num_elems = 100;

			raw_in_0.resize(num_elems);

			for (size_t i = 0; i < num_elems; ++i)
			{
				raw_in_0[i] = i - (num_elems / 2.0f);
			}

			target_0 = softmaxCPU(raw_in_0);
			grads_0 = softmaxGrad(raw_in_0, target_0);
		}

		std::vector<float> softmaxCPU(const std::vector<float>& ins)
		{
			auto num_elems = ins.size();
			auto max_elem = *std::max_element(ins.begin(), ins.end());

			std::vector<float> out(ins.size());

			float elem_sum = 0;

			for (size_t i = 0; i < num_elems; ++i)
			{
				elem_sum += out[i] = std::exp(ins[i] - max_elem);
			}

			for (size_t i = 0; i < num_elems; ++i)
			{
				out[i] /= elem_sum;
			}

			return out;
		}

		std::vector<float> softmaxGrad(const std::vector<float>& ins, const std::vector<float>& out)
		{
			auto num_elems = ins.size();

			std::vector<float> grad(ins.size());
			
			for (size_t i = 0; i < num_elems; ++i)
			{
				float tmp_sum = 0;
				for (size_t j = 0; j < num_elems; ++j)
				{
					tmp_sum += out[j] * (i == j ?
						(ins[i] * (1 - ins[j])) :
						-ins[i] * ins[j]);
				}
				grad[i] = tmp_sum;
			}

			return grad;
		}
	};

	TEST_F(SoftmaxTest, ShouldCalculateSoftmaxFrom1DTensor)
	{
		// Get input data
		auto& raw_in = raw_ins[0];
		auto& target = targets[0];
		auto num_elems = raw_in.size();

		// Build Op
		Shape shape(1, num_elems, 1);
		Input input(shape, raw_in);

		Softmax softmax;

		Tensor* out = softmax(input());

		// Calculate op
		softmax.forward();

		// Assert equivalence
		ASSERT_EQ(out->get_shape(), Shape(1, num_elems, 1));
		for (size_t i = 0; i < num_elems; ++i)
		{
			ASSERT_NEAR(static_cast<float>(target[i]), out->raw()[i], 0.0001);
		}
	}

	TEST_F(SoftmaxTest, ShouldCalculateSoftmaxGradFrom1DTensorAndGrad)
	{
		// Generate input data
		auto& raw_in = raw_ins[0];
		auto& target = targets[0];
		auto& grad = grads[0];
		auto num_elems = raw_in.size();

		// Build Op
		Shape shape(1, num_elems, 1);

		// Raw Tensor because Input tensors skip gradient calculation
		Tensor input_tensor(shape, Tensor::LayerType::Output);
		input_tensor.copy_from(&raw_in[0]);
		
		Softmax softmax;

		Tensor* out = softmax(&input_tensor);

		// Update output directly to avoid redundant calculation
		out->copy_from(&target[0]);

		// Calculate op
		softmax.backwards();

		// Get Gradient
		auto& in_grad = input_tensor.gradient();

		// Assert equivalence
		ASSERT_EQ(in_grad.get_shape(), Shape(1, num_elems, 1));
		for (size_t i = 0; i < num_elems; ++i)
		{
			ASSERT_NEAR(static_cast<float>(grad[i]), in_grad.raw()[i], 0.0001);
		}
	}
};
