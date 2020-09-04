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
 */

#pragma once

#include <unordered_map>
#include <vector>
#include <memory>
#include <ctime>

#include "gamedirector.hpp"

class Setting;
class NeuralNewtBrain;
class AIHungryHippo;
class AIQuickQuack;
class AIRampantRhino;

typedef GameDirector<AIHungryHippo, AIQuickQuack, AIRampantRhino> Director;


class NewtBrainTrainer
{
private:
	std::unordered_map<std::string, Setting>& _settings;
	std::string _rulesetname;
	std::time_t _startTime;
	std::vector<std::shared_ptr<NeuralNewtBrain>> _brains;
	size_t _round;

public:
	NewtBrainTrainer(std::unordered_map<std::string, Setting>& settings,
		const std::string& rulesetname);

private:
	Director::RoundResults playRound();
	void evolveBrains();
	void saveBrains();
	Director::RoundResults sortBrains(const Director::RoundResults& results);

public:
	void resume(std::string session, size_t round, bool initEvolve);
	void train();
};
