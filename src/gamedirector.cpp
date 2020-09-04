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

#include "gamedirector_spec.hpp"

#include "libs/aftermath/aineuralnewt.hpp"
#include "libs/aftermath/difficulty.hpp"
#include "libs/aftermath/map.hpp"
#include "libs/jsoncpp/json.h"

#include <iostream>
#include <random>

#include "setting.hpp"
#include "nnet/neuralnewtbrain.hpp"


static std::default_random_engine gen;
static std::bernoulli_distribution bDis;
static std::uniform_int_distribution<size_t> uDis;

template <class ...Ts> size_t GameDirector<Ts...>::brainsPerPool;

template <class ...Ts>
void GameDirector<Ts...>::updatePopGame(const PopGame& game,
	RoundResults& round)
{
	size_t i = game.idx1;
	size_t j = game.idx2;
	round.popScores[i] += game.results.ai1score;
	round.popScores[j] += game.results.ai2score;
	round.totalScores[i] += game.results.ai1score;
	round.totalScores[j] += game.results.ai2score;
	if (game.results.draw)
	{
		round.draws[i]++;
		round.draws[j]++;
	}
	else if (game.results.ai1defeated)
	{
		round.losses[i]++;
		round.wins[j]++;
	}
	else if (game.results.ai2defeated)
	{
		round.wins[i]++;
		round.losses[j]++;
	}
}

template<typename T1, typename T2 = std::nullptr_t, typename ...tail>
struct index
{
	static const size_t value = std::is_same<T1, T2>::value ? 0
		: index<T1, tail...>::value + 1;
};
template<typename T1, typename T2>
struct index<T1, T2>
{
	static const size_t value = std::is_same<T1, T2>::value ? 0 : 1;
};

template <class ...Ts>
template <class T>
void GameDirector<Ts...>::updateAIGame(const struct AIGame<T>& game,
	RoundResults& round)
{
	size_t i = game.idx;
	int score = game.first ? game.results.ai1score : game.results.ai2score;
	round.aiScores[index<T, Ts...>::value][i] += score;
	round.totalScores[i] += score;
	if (game.results.draw) round.draws[i]++;
	else if ((game.first && game.results.ai1defeated)
		|| (!game.first && game.results.ai2defeated)) round.losses[i]++;
	else if ((game.first && game.results.ai2defeated)
		|| (!game.first && game.results.ai1defeated)) round.wins[i]++;
}

template <class ...Ts>
GameDirector<Ts...>::GameDirector(
		std::unordered_map<std::string, Setting>& settings,
		const std::string& rulesetname,
		const std::vector<std::shared_ptr<NeuralNewtBrain>>& brains) :
	_settings(settings),
	_rulesetname(rulesetname),
	_brains(brains)
{
	brainsPerPool = _settings["brains_per_pool"];
	bDis = std::bernoulli_distribution(_settings["recording_chance"]);
	uDis = std::uniform_int_distribution<size_t>(0, _settings["map_names"].size() - 1);
}

template <class ...Ts>
void GameDirector<Ts...>::turn(std::unique_ptr<Game>& game)
{
	auto& automaton = game->automaton;
	auto& phase = game->phase;
	auto& ai1 = game->ai1;
	auto& ai2 = game->ai2;

	bool draw = false;

	while (phase != Phase::DECAY)
	{
		switch (phase)
		{
			case Phase::GROWTH:
			case Phase::ACTION:
			{
				if (automaton->active())
				{
					ChangeSet cset = automaton->act();
					ai1->receiveChanges(cset.get(ai1->player()));
					ai2->receiveChanges(cset.get(ai2->player()));
				}
				else phase = Phase::RESTING;
			}
			break;

			case Phase::RESTING:
			{
				if (automaton->gameover())
				{
					phase = Phase::DECAY;
					break;
				}
				else if (automaton->globalScore() <= 0)
				{
					draw = true;
					phase = Phase::DECAY;
					break;
				}
				else if (game->turns >= 100)
				{
					draw = true;
					phase = Phase::DECAY;
					break;
				}

				ChangeSet cset = automaton->hibernate();
				ai1->receiveChanges(cset.get(ai1->player()));
				ai2->receiveChanges(cset.get(ai2->player()));
				phase = Phase::PLANNING;
			}
			break;

			case Phase::PLANNING:
			{
				if (!game->planning)
				{
					// The AIs are now ready to calculate the next set of
					// orders, but that will be done elsewhere.
					game->planning = true;
					return;
				}
				ChangeSet cset = automaton->awake();
				ai1->receiveChanges(cset.get(ai1->player()));
				ai2->receiveChanges(cset.get(ai2->player()));
				phase = Phase::STAGING;
				game->planning = false;
			}
			break;

			case Phase::STAGING:
			{
				automaton->receive(ai1->player(), ai1->orders());
				automaton->receive(ai2->player(), ai2->orders());

				ChangeSet cset = automaton->prepare();
				ai1->receiveChanges(cset.get(ai1->player()));
				ai2->receiveChanges(cset.get(ai2->player()));

				phase = Phase::ACTION;
				game->turns++;
			}
			break;

			case Phase::DECAY:
			break;
		}
	}

	if (phase != Phase::DECAY) return;

	GameResults& results = game->results;
	results.ai1score = automaton->score(ai1->player());
	results.ai2score = automaton->score(ai2->player());
	results.ai1defeated = automaton->defeated(ai1->player());
	results.ai2defeated = automaton->defeated(ai2->player());
	results.draw = draw;
	results.turns = game->turns;
	game->done = true;
}

template <class ...Ts>
std::shared_ptr<AICommander> GameDirector<Ts...>::makeNNCommander(
	const std::shared_ptr<NeuralNewtBrain>& brain, size_t i,
	Json::Value& metadata)
{
	std::shared_ptr<AICommander> ai = std::make_shared<AINeuralNewt>(
		Player(i + 1),
		Difficulty::HARD,
		_rulesetname,
		'A' + i,
		brain);

	Json::Value json = Json::objectValue;
	json["player"] = ::stringify(ai->player());
	json["difficulty"] = ::stringify(ai->difficulty());
	json["character"] = ai->characterstring();
	json["displayname"] = ai->displayname();
	json["ainame"] = brain->shortName();
	json["authors"] = ai->authors();
	metadata[int(i)] = json;

	return ai;
}

template <class ...Ts>
void GameDirector<Ts...>::setupPopGame(std::unique_ptr<PopGame>& game)
{
	Json::Value metadata = Json::objectValue;
	metadata["online"] = false;
	metadata["planningtime"] = 0;
	metadata["bots"] = Json::arrayValue;

	game->ai1 = makeNNCommander(_brains[game->idx1], 0, metadata["bots"]);
	game->ai2 = makeNNCommander(_brains[game->idx2], 1, metadata["bots"]);
	game->results.ai1name = _brains[game->idx1]->mediumName();
	game->results.ai2name = _brains[game->idx2]->mediumName();

	static std::vector<Player> players = getPlayers(2);
	static std::vector<std::string> mapnames = _settings["map_names"];
	std::string mapname = mapnames[uDis(gen)];
	game->automaton.reset(new Automaton(players, _rulesetname));
	game->automaton->load(mapname, false);
	if (bDis(gen)) game->automaton->startRecording(metadata);
	game->phase = Phase::GROWTH;
	game->turns = 0;
}

template <class ...Ts>
template <class T>
std::shared_ptr<AICommander> GameDirector<Ts...>::makeCommander(size_t i,
	Json::Value& metadata)
{
	std::shared_ptr<AICommander> ai = std::make_shared<T>(
		Player(i + 1),
		Difficulty::HARD,
		_rulesetname,
		'A' + i);

	Json::Value json = Json::objectValue;
	json["player"] = ::stringify(ai->player());
	json["difficulty"] = ::stringify(ai->difficulty());
	json["character"] = ai->characterstring();
	json["displayname"] = ai->displayname();
	json["ainame"] = ai->ainame();
	json["authors"] = ai->authors();
	metadata[int(i)] = json;

	return ai;
}

template <class ...Ts>
template <class T>
void GameDirector<Ts...>::setupAIGame(std::unique_ptr<AIGame<T>>& game)
{
	Json::Value metadata = Json::objectValue;
	metadata["online"] = false;
	metadata["planningtime"] = 0;
	metadata["bots"] = Json::arrayValue;

	if (game->first)
	{
		game->ai1 = makeNNCommander(_brains[game->idx], 0, metadata["bots"]);
		game->ai2 = makeCommander<T>(1, metadata["bots"]);
		game->results.ai1name = _brains[game->idx]->mediumName();
		game->results.ai2name = game->ai2->ainame();
	}
	else
	{
		game->ai1 = makeCommander<T>(0, metadata["bots"]);
		game->ai2 = makeNNCommander(_brains[game->idx], 1, metadata["bots"]);
		game->results.ai1name = game->ai1->ainame();
		game->results.ai2name = _brains[game->idx]->mediumName();
	}

	static std::vector<Player> players = getPlayers(2);
	static std::vector<std::string> mapnames = _settings["map_names"];
	std::string mapname = mapnames[uDis(gen)];
	game->automaton.reset(new Automaton(players, _rulesetname));
	game->automaton->load(mapname, false);
	if (bDis(gen)) game->automaton->startRecording(metadata);
	game->phase = Phase::GROWTH;
	game->turns = 0;
}

template <class ...Ts>
void GameDirector<Ts...>::addPopGame(size_t brain1Idx, size_t brain2Idx)
{
	std::unique_ptr<PopGame> game(new PopGame());
	game->idx1 = brain1Idx;
	game->idx2 = brain2Idx;
	setupPopGame(game);
	_games.push_back(std::move(game));
}

template <class ...Ts>
template <class T>
void GameDirector<Ts...>::addAIGame(size_t brainIdx, bool first)
{
	std::unique_ptr<AIGame<T>> game(new AIGame<T>());
	game->idx = brainIdx;
	game->first = first;
	setupAIGame(game);
	_games.push_back(std::move(game));
}

template <class ...Ts>
typename GameDirector<Ts...>::RoundResults GameDirector<Ts...>::play()
{
	RoundResults results;
	for (const auto& brain : _brains)
	{
		results.names.push_back(brain->mediumName());
		results.popScores.push_back(0);
		for (size_t i = 0; i < sizeof...(Ts); i++)
		{
			results.aiScores[i].push_back(0);
		}
		results.totalScores.push_back(0);
		results.wins.push_back(0);
		results.draws.push_back(0);
		results.losses.push_back(0);
	}
	while (true)
	{
		for (auto gamePtr = _games.begin(); gamePtr != _games.end(); /**/)
		{
			auto& game = *gamePtr;
			turn(game);
			if (game->done)
			{
				const GameResults& gameResults = game->results;
				game->update(results);
				if (_settings["verbose"]) std::cout << gameResults;
				gamePtr = _games.erase(gamePtr);
			}
			else
			{
				game->ai1finished = false;
				game->ai2finished = false;
				game->ai1->preprocess();
				game->ai2->preprocess();
				gamePtr++;
			}
		}
		if (_games.empty()) break;

		bool allFinished = false;
		while (!allFinished)
		{
			allFinished = true;
			for (auto& game : _games)
			{
				if (!game->ai1finished)
				{
					game->ai1->process();
					allFinished = false;
				}
				if (!game->ai2finished)
				{
					game->ai2->process();
					allFinished = false;
				}
			}

			for (auto& game : _games)
			{
				auto& ai1finished = game->ai1finished;
				auto& ai2finished = game->ai2finished;
				if (!ai1finished)
				{
					ai1finished = game->ai1->postprocess();
					if (!ai1finished) allFinished = false;
				}
				if (!ai2finished)
				{
					ai2finished = game->ai2->postprocess();
					if (!ai2finished) allFinished = false;
				}
			}
		}
	}
	return results;
}
