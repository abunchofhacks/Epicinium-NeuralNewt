/**
 * Part of Epicinium NeuralNewt
 * developed by A Bunch of Hacks.
 *
 * Copyright (c) 2020 A Bunch of Hacks
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * [authors:]
 * Daan Mulder (daan@abunchofhacks.coop)
 * Sander in 't Veld (sander@abunchofhacks.coop)
 */

#include "module.hpp"

#include "libs/aftermath/newtbrain.hpp"
#include "libs/aftermath/position.hpp"
#include "neuralnewtbrain.hpp"

#include "setting.hpp"


Module::Module(std::unordered_map<std::string, Setting>& settings) :
	_settings(settings),
	_planes(NeuralNewtBrain::NUM_PLANES),
	_planeX(Position::MAX_COLS),
	_planeY(Position::MAX_ROWS),
	_actionSize(NewtBrain::Output::SIZE),
	_conv1(register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(
		_planes,
		int(_settings["num_channels"]),
		3).stride(1).padding(1).bias(false)))),
	_conv2(register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(
		int(_settings["num_channels"]),
		int(_settings["num_channels"]),
		3).stride(1).padding(1).bias(false)))),
	_conv3(register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(
		int(_settings["num_channels"]),
		int(_settings["num_channels"]),
		3).stride(1).bias(false)))),
	_conv4(register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(
		int(_settings["num_channels"]),
		int(_settings["num_channels"]),
		3).stride(1).bias(false)))),
	_fc1(register_module("fc1", torch::nn::Linear(
		int(_settings["num_channels"]) * (_planeX - 4) * (_planeY - 4), _actionSize * 2))),
	_fc2(register_module("fc2", torch::nn::Linear(_actionSize * 2, _actionSize))),
	_fc3(register_module("fc3", torch::nn::Linear(_actionSize, _actionSize)))
{
	// We do not use the module's training mode for evolutionary training.
	eval();
}

Module::Module(Module&& other) :
	_settings(other._settings),
	_planes(NeuralNewtBrain::NUM_PLANES),
	_planeX(Position::MAX_COLS),
	_planeY(Position::MAX_ROWS),
	_actionSize(NewtBrain::Output::SIZE),
	_conv1(register_module("conv1", std::move(other._conv1))),
	_conv2(register_module("conv2", std::move(other._conv2))),
	_conv3(register_module("conv3", std::move(other._conv3))),
	_conv4(register_module("conv4", std::move(other._conv4))),
	_fc1(register_module("fc1", std::move(other._fc1))),
	_fc2(register_module("fc2", std::move(other._fc2))),
	_fc3(register_module("fc3", std::move(other._fc3)))
{}

Module& Module::operator=(Module&& other)
{
	if (this != &other)
	{
		_settings = other._settings;
		_conv1 = register_module("conv1", std::move(other._conv1));
		_conv2 = register_module("conv2", std::move(other._conv2));
		_conv3 = register_module("conv3", std::move(other._conv3));
		_conv4 = register_module("conv4", std::move(other._conv4));
		_fc1 = register_module("fc1", std::move(other._fc1));
		_fc2 = register_module("fc2", std::move(other._fc2));
		_fc3 = register_module("fc3", std::move(other._fc3));
	}
	return *this;
}

void Module::reset()
{
	*this = Module(_settings);
}

// This is the Conv2d::forward implementation of libtorch v1.4.0
// (build hash 7f73f1d591afba823daa4a99a939217fb54d7688). The implementation
// changed in v1.6.0, so this might be incompatible with newer versions.
torch::Tensor convForward(const torch::nn::Conv2d& conv,
	const torch::Tensor& input)
{
	if (torch::get_if<torch::enumtype::kCircular>(&conv->options.padding_mode()))
	{
		std::vector<int64_t> expanded_padding = {
			((*conv->options.padding())[1] + 1) / 2,
			(*conv->options.padding())[1] / 2,
			((*conv->options.padding())[0] + 1) / 2,
			(*conv->options.padding())[0] / 2
		};
		return torch::nn::functional::detail::conv2d(
			torch::nn::functional::detail::pad(input, expanded_padding, torch::kCircular, 0),
			conv->weight, conv->bias,
			conv->options.stride(),
			/*padding=*/0,
			conv->options.dilation(),
			conv->options.groups()
		);
	}
	return torch::nn::functional::detail::conv2d(
		input,
		conv->weight,
		conv->bias,
		conv->options.stride(),
		conv->options.padding(),
		conv->options.dilation(),
		conv->options.groups()
	);
}

// We prevent calling the forward functions of the underlying modules so we can
// declare this function const and thus guarantee it is thread-safe.
torch::Tensor Module::forward(torch::Tensor& s) const
{
	s = torch::relu(convForward(_conv1, s));
	s = torch::relu(convForward(_conv2, s));
	s = torch::relu(convForward(_conv3, s));
	s = torch::relu(convForward(_conv4, s));
	s = s.view({-1,
		long(int(_settings["num_channels"]) * (_planeX - 4) * (_planeY - 4))});

	s = torch::relu(torch::linear(s, _fc1->weight, _fc1->bias));
	s = torch::relu(torch::linear(s, _fc2->weight, _fc2->bias));

	torch::Tensor pi = torch::linear(s, _fc3->weight, _fc3->bias);

	return torch::sigmoid(pi);
}
