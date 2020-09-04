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

#include "libs/aftermath/newtbrain.hpp"

#include "brainname.hpp"

#include <unordered_map>
#include <queue>

class Setting;
class Module;


class NeuralNewtBrain : public NewtBrain
{
public:
	static const size_t NUM_PLANES;

	static NeuralNewtBrain mutate(const NeuralNewtBrain& brain,
		size_t round, float deviationFactor, float selectionChance);
	static std::pair<NeuralNewtBrain, NeuralNewtBrain> combine(
		const NeuralNewtBrain& brain1, const NeuralNewtBrain& brain2,
		size_t round);

private:
	std::unordered_map<std::string, Setting>& _settings;
	std::shared_ptr<Module> _module;
	BrainNamePtr _name;

	size_t _count = 0;
	std::vector<int8_t> _input;
	std::queue<Output> _output;

public:
	NeuralNewtBrain(std::unordered_map<std::string, Setting>& settings,
		const BrainNamePtr& name);
	NeuralNewtBrain(const std::shared_ptr<Module>& module,
		std::unordered_map<std::string, Setting>& settings,
		const BrainNamePtr& name);
	NeuralNewtBrain(const NeuralNewtBrain&) = delete;
	NeuralNewtBrain(NeuralNewtBrain&& other);
	NeuralNewtBrain& operator=(const NeuralNewtBrain&) = delete;
	NeuralNewtBrain& operator=(NeuralNewtBrain&&) = default;
	~NeuralNewtBrain() = default;

private:
	NeuralNewtBrain(const NeuralNewtBrain& brain, const BrainNamePtr& name);

	static std::vector<int8_t> encode(const AICommander& input);

	virtual void prepare(const AICommander& input) override;
	virtual Output evaluate() override;

public:
	bool save(const std::string& folder, const std::string& filename);

	void load(const std::string& folder, const std::string& filename);

	std::string mediumName() const { return _name->mediumName(); }
	std::string shortName() const { return _name->shortName(); }
};
