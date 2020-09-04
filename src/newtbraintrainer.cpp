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

#include "newtbraintrainer.hpp"

#include "libs/aftermath/aihungryhippo.hpp"
#include "libs/aftermath/aiquickquack.hpp"
#include "libs/aftermath/airampantrhino.hpp"

#include <torch/torch.h>

#include "setting.hpp"
#include "nnet/neuralnewtbrain.hpp"


NewtBrainTrainer::NewtBrainTrainer(
		std::unordered_map<std::string, Setting>& settings,
		const std::string& rulesetname) :
	_settings(settings),
	_rulesetname(rulesetname),
	_startTime(std::time(nullptr)),
	_round(0)
{
	if (settings.count("torch_threads"))
		torch::set_num_threads(settings["torch_threads"]);
	if (settings["cuda"] && !torch::cuda::is_available())
	{
		settings["cuda"] = false;
	}
	if (settings["cuda"]) std::cout << "YAAY CUDA!" << std::endl;
	else std::cout << "aww no CUDA" << std::endl;
}

Director::RoundResults NewtBrainTrainer::playRound()
{
	std::chrono::high_resolution_clock::time_point start;
	static bool timing = _settings["timing"];
	size_t count = 0;
	if (timing) start = std::chrono::high_resolution_clock::now();

	Director director(_settings, _rulesetname, _brains);
	// Round robin (TODO do we want something else?)
	for (size_t i = 0; i < _brains.size(); i++)
	{
		for (size_t j = i + 1; j < _brains.size(); j++)
		{
			if ((i + j) % 2 == 0) director.addPopGame(i, j);
			else director.addPopGame(j, i);
			if (timing) count++;
		}
		for (size_t j = 0; j < size_t(_settings["num_AI_games"]); j++)
		{
			director.addAIGame<AIHungryHippo>(i, j % 2 == 0);
			director.addAIGame<AIQuickQuack>(i, j % 2 == 0);
			director.addAIGame<AIRampantRhino>(i, j % 2 == 0);
			if (timing) count += 3;
		}
	}
	Director::RoundResults results = director.play();

	if (timing)
	{
		auto end = std::chrono::high_resolution_clock::now();
		float d =
			std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
			.count() / 1000.0f;
		std::cout << "Playing round took " << d << "s (" << (d / count)
			<< "s per game)" << std::endl;
	}

	return results;
}

void NewtBrainTrainer::saveBrains()
{
	if (!_settings["save_brains"]) return;

	static bool timing = _settings["timing"];
	std::chrono::high_resolution_clock::time_point start;
	size_t count = 0;
	if (timing) start = std::chrono::high_resolution_clock::now();

	std::string folder = "brains/" + std::to_string(_startTime);
	std::ofstream brainList;
	bool first = true;

	for (auto& brain : _brains)
	{
		std::cout << "saving brain " << brain->mediumName() << " as "
			<< brain->shortName() << std::endl;
		bool saved = brain->save(folder, brain->shortName() + ".pth.tar");
		if (timing && saved) count++;
		if (first)
		{
			brainList.open(folder + "/round" + std::to_string(_round) + ".txt",
				std::ofstream::trunc);
			first = false;
		}
		brainList << brain->shortName() << std::endl;
	}

	if (timing)
	{
		auto end = std::chrono::high_resolution_clock::now();
		float d =
			std::chrono::duration_cast<std::chrono::microseconds>(end - start)
			.count() / 1000.0f;
		std::cout << "Saving brains took " << d << "ms (" << (d / count)
			<< "ms per brain)" << std::endl;
	}
}

// Sorts the brains based on a round's resuts so the best-performing are first.
// Also returns the sorted results.
// Inspired by: https://stackoverflow.com/a/17074810
Director::RoundResults NewtBrainTrainer::sortBrains(
	const Director::RoundResults& results)
{
	static bool timing = _settings["timing"];
	std::chrono::high_resolution_clock::time_point start;
	if (timing) start = std::chrono::high_resolution_clock::now();

	Director::RoundResults sortedResults;
	static size_t numPools = _settings["num_pools"];
	static size_t brainsPerPool = _settings["brains_per_pool"];
	for (size_t i = 0; i < numPools; i++)
	{
		std::vector<size_t> permutation(brainsPerPool);
		std::iota(permutation.begin(), permutation.end(), 0);
		// Sort in descending order of score.
		std::stable_sort(permutation.begin(), permutation.end(),
			[&results, i](size_t a, size_t b) {
				return results.totalScores[a + i * brainsPerPool]
					> results.totalScores[b + i * brainsPerPool];
			}
		);

		// Sort the results in a new object, but sort the brains in place
		// because they are HUGE.
		std::vector<bool> brainDone(brainsPerPool, false);
		for (size_t j = 0; j < brainsPerPool; j++)
		{
			size_t k = permutation[j];
			sortedResults.names.push_back(results.names[k + i * brainsPerPool]);
			sortedResults.popScores.push_back(
				results.popScores[k + i * brainsPerPool]);
			sortedResults.aiScores[0].push_back(
				results.aiScores[0][k + i * brainsPerPool]);
			sortedResults.aiScores[1].push_back(
				results.aiScores[1][k + i * brainsPerPool]);
			sortedResults.aiScores[2].push_back(
				results.aiScores[2][k + i * brainsPerPool]);
			sortedResults.totalScores.push_back(
				results.totalScores[k + i * brainsPerPool]);
			sortedResults.wins.push_back(results.wins[k + i * brainsPerPool]);
			sortedResults.draws.push_back(results.draws[k + i * brainsPerPool]);
			sortedResults.losses.push_back(
				results.losses[k + i * brainsPerPool]);
			if (brainDone[j]) continue;
			brainDone[j] = true;
			size_t prev_k = j;
			while (j != k)
			{
				std::swap(_brains[prev_k + i * brainsPerPool],
					_brains[k + i * brainsPerPool]);
				brainDone[k] = true;
				prev_k = k;
				k = permutation[k];
			}
		}
	}

	if (timing)
	{
		auto end = std::chrono::high_resolution_clock::now();
		float d =
			std::chrono::duration_cast<std::chrono::microseconds>(end - start)
			.count() / 1000.0f;
		std::cout << "Sorting brains took " << d << "ms" << std::endl;
	}

	return sortedResults;
}

void NewtBrainTrainer::evolveBrains()
{
	static bool timing = _settings["timing"];
	std::chrono::high_resolution_clock::time_point start;
	size_t coCount = 0;
	size_t muCount = 0;
	if (timing) start = std::chrono::high_resolution_clock::now();

	static float deviationFactor = _settings["mutation_deviation_factor"];
	static float selectionChance =
		std::min(float(_settings["mutation_selection_chance"]), 1.0f);

	static size_t numPools = _settings["num_pools"];
	static size_t brainsPerPool = _settings["brains_per_pool"];
	static size_t numParents = brainsPerPool / 5;
	static size_t numKeep = numParents * 2;

	for (size_t i = 0; i < numPools; i++)
	{
		size_t j = numKeep;

		for (size_t k = 1; k < numParents && j < brainsPerPool - numParents;
			k++)
		{
			for (size_t l = 0; l < k && j < brainsPerPool - numParents; l++)
			{
				std::tie(
					_brains[j + i * brainsPerPool],
					_brains[j + i * brainsPerPool + 1]
				) = [](auto&& pair) {
					auto ptr1 = std::make_shared<NeuralNewtBrain>(
						std::move(pair.first));
					auto ptr2 = std::make_shared<NeuralNewtBrain>(
						std::move(pair.second));
					return std::make_pair(ptr1, ptr2);
				}(NeuralNewtBrain::combine(
					*_brains[l + i * brainsPerPool],
					*_brains[k + i * brainsPerPool],
					_round)
				);
				if (timing) coCount++;
				j += 2;
			}
		}

		for (size_t k = 0; k < numParents && j < brainsPerPool; k++)
		{
			_brains[j + i * brainsPerPool] = std::make_shared<NeuralNewtBrain>(
				NeuralNewtBrain::mutate(*_brains[k + i * brainsPerPool], _round,
				deviationFactor, selectionChance));
			if (timing) muCount++;
			j++;
		}
	}

	if (timing)
	{
		auto end = std::chrono::high_resolution_clock::now();
		float d =
			std::chrono::duration_cast<std::chrono::microseconds>(end - start)
			.count() / 1000.0f;
		if (muCount > 0) std::cout << "Evolving brains took " << d << "ms ("
			<< (d / _brains.size()) << "ms per brain)" << std::endl;
	}
}

void NewtBrainTrainer::resume(std::string session, size_t round,
	bool initEvolve)
{
	bool timing = _settings["timing"];
	std::chrono::high_resolution_clock::time_point start;
	if (timing) start = std::chrono::high_resolution_clock::now();

	_round = round;
	std::string folder = "brains/" + session;
	std::string filename = folder + "/round" + std::to_string(_round) + ".txt";

	std::ifstream file(filename);
	if (!file) throw std::runtime_error("Error while resuming: file "
		+ filename + " cannot be opened");
	std::string line;
	size_t i = 0;
	while (std::getline(file, line))
	{
		std::string name, filename;
		// If the line contains a space, everything before is the brain name and
		// after is the filename. If not, assume the entire line is both.
		size_t delimiter = line.find(' ');
		if (delimiter != std::string::npos)
		{
			name = line.substr(0, delimiter);
			filename = line.substr(delimiter + 1);
		}
		else
		{
			name = filename = line;
		}

		_brains.push_back(std::make_shared<NeuralNewtBrain>(_settings,
			std::make_shared<RestoredBrainName>(name, _round)));
		_brains.back()->load(folder, filename + ".pth.tar");
		i++;
		if (i > size_t(_settings["num_pools"])
			* size_t(_settings["brains_per_pool"]))
		{
			throw std::runtime_error("Number of brains in " + filename
				+ " exceeds number of brains in settings, I cannot handle"
				  " that");
		}
	}
	if (timing)
	{
		auto end = std::chrono::high_resolution_clock::now();
		float d =
			std::chrono::duration_cast<std::chrono::microseconds>(end - start)
			.count() / 1000.0f;
		std::cout << "Resuming brains took " << d << "ms (" << (d / i)
			<< "ms per brain)" << std::endl;
	}
	if (i < size_t(_settings["num_pools"])
		* size_t(_settings["brains_per_pool"]))
	{
		std::cerr << "WARNING: number of brains in " << filename
			<< " less than number of brains in settings, this may cause"
			   " unexpected behaviour" << std::endl;
	}
	std::cout << "Resumed from " << filename << std::endl;
	if (initEvolve)
	{
		evolveBrains();
		_round++;
	}
}

void NewtBrainTrainer::train()
{
	std::chrono::high_resolution_clock::time_point start;
	static bool timing = _settings["timing"];
	if (timing) start = std::chrono::high_resolution_clock::now();

	static size_t numRounds = _settings["num_rounds"];
	static size_t numPools = _settings["num_pools"];
	static size_t brainsPerPool = _settings["brains_per_pool"];
	static bool verbose = _settings["verbose"];

	if (_brains.size() == 0)
	{
		for (size_t i = 0; i < numPools * brainsPerPool; i++)
		{
			_brains.push_back(std::make_shared<NeuralNewtBrain>(_settings,
				std::make_shared<SeedBrainName>(i)));
		}
	}

	if (timing)
	{
		auto end = std::chrono::high_resolution_clock::now();
		float d =
			std::chrono::duration_cast<std::chrono::microseconds>(end - start)
			.count() / 1000.0f;
		std::cout << "Initializing brains took " << d << "ms" << std::endl;
	}

	saveBrains();

	while (_round < numRounds)
	{
		if (verbose) std::cout << "ROUND " << _round << std::endl;
		Director::RoundResults results = playRound();
		Director::RoundResults sortedResults = sortBrains(results);
		if (verbose) std::cout << sortedResults << std::endl;
		evolveBrains();
		_round++;
		saveBrains();
	}
}
