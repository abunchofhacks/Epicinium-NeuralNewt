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

#include "libs/aftermath/ailibrary.hpp"
#include "libs/aftermath/aineuralnewt.hpp"
#include "libs/aftermath/player.hpp"
#include "libs/aftermath/difficulty.hpp"

#include <torch/torch.h>

#include "setting.hpp"
#include "nnet/module.hpp"
#include "nnet/neuralnewtbrain.hpp"

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__ ((visibility ("default")))
#endif


static std::unordered_map<std::string, Setting> _settings = {
	{"timing", false},
	{"cuda", false},
	{"num_channels", 48}
};
static std::shared_ptr<Module> _module;

extern "C"
{
	EXPORT void setup(int argc, const char* const argv[]);
	EXPORT AINeuralNewt* allocate(
		const char* player, const char* difficulty,
		const char* rulesetname, char character);
	EXPORT void deallocate(AINeuralNewt* ptr);

	void setup(int argc, const char* const argv[])
	{
		AILibrary::setup("libneuralnewt", argc, argv);
		_module = std::make_shared<Module>(_settings);
		auto name = std::make_shared<RestoredBrainName>("default", 0);
		auto brain = std::make_shared<NeuralNewtBrain>(_module, _settings,
			name);
		brain->load("ai", "default.brain");
	}

	AINeuralNewt* allocate(
		const char* player, const char* difficulty,
		const char* rulesetname, char character)
	{
		// TODO add plog?
		//LOGD << "Allocating " << difficulty << " " << player << ""
		//	" AINeuralNewt named '" << character << "'"
		//	" with ruleset " << rulesetname;
		auto name = std::make_shared<RestoredBrainName>("default", 0);
		auto brain = std::make_shared<NeuralNewtBrain>(_module, _settings,
			name);
		return new AINeuralNewt(parsePlayer(player),
			parseDifficulty(difficulty), rulesetname, character, brain);
	}

	void deallocate(AINeuralNewt* ptr)
	{
		delete ptr;
	}
}
