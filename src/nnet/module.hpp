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

#pragma once

#include <torch/torch.h>

class Setting;


class Module : public torch::nn::Cloneable<Module>
{
private:
	friend class NeuralNewtBrain;

	std::unordered_map<std::string, Setting>& _settings;
	size_t _planes, _planeX, _planeY;
	size_t _actionSize;
	torch::nn::Conv2d _conv1;
	torch::nn::Conv2d _conv2;
	torch::nn::Conv2d _conv3;
	torch::nn::Conv2d _conv4;
	torch::nn::Linear _fc1;
	torch::nn::Linear _fc2;
	torch::nn::Linear _fc3;

public:
	Module(std::unordered_map<std::string, Setting>& settings);
	Module(const Module&) = default;
	Module(Module&& other);
	Module& operator=(const Module&) = default;
	Module& operator=(Module&& other);
	~Module() = default;

	void reset() override;

	torch::Tensor forward(torch::Tensor& s) const;
};
