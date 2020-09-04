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

#include "libs/aftermath/automaton.hpp"
#include "libs/jsoncpp/json-forwards.h"

class Setting;
class NeuralNewtBrain;
class AICommander;


template<typename T1, typename T2 = std::nullptr_t, typename ...tail>
struct any_is_same
{
	static const bool value = std::is_same<T1, T2>::value
		|| any_is_same<T1, tail...>::value;
};
template<typename T1, typename T2>
struct any_is_same<T1, T2> : std::is_same<T1, T2> {};


template <class ...Ts>
class GameDirector
{
private:
	struct GameResults
	{
		std::string ai1name;
		std::string ai2name;
		int ai1score;
		int ai2score;
		bool ai1defeated;
		bool ai2defeated;
		bool draw;
		size_t turns;
	};
	friend std::ostream& operator<<(std::ostream& os,
		const struct GameDirector::GameResults& results)
	{
		os << (results.draw ? "Drawn in " : "Decided in ") << results.turns
			<< " turns: " << results.ai1name << " (" << results.ai1score << ")"
			<< ", " << results.ai2name << " (" << results.ai2score << ")"
			<< std::endl;
		return os;
	}
public:
	struct RoundResults
	{
		std::vector<std::string> names;
		std::vector<int> popScores;
		std::array<std::vector<int>, sizeof...(Ts)> aiScores;
		std::vector<int> totalScores;
		std::vector<int> wins;
		std::vector<int> draws;
		std::vector<int> losses;
	};
	friend std::ostream& operator<<(std::ostream& os,
		const struct GameDirector::RoundResults& results)
	{
		int poolPopScore = 0;
		int totalPopScore = 0;
		std::array<int, sizeof...(Ts)> poolAIScore;
		poolAIScore.fill(0);
		std::array<int, sizeof...(Ts)> totalAIScore;
		totalAIScore.fill(0);
		int poolScore = 0;
		int totalScore = 0;
		size_t poolWins = 0;
		size_t totalWins = 0;
		size_t poolDraws = 0;
		size_t totalDraws = 0;
		size_t poolLosses = 0;
		size_t totalLosses = 0;
		for (size_t i = 0; i < results.names.size(); i++)
		{
			os << results.names[i] << ": " << results.totalScores[i] << " ("
				<< results.popScores[i];
			for (size_t j = 0; j < results.aiScores.size(); j++)
			{
				os << "+" << results.aiScores[j][i];
				poolAIScore[j] += results.aiScores[j][i];
				totalAIScore[j] += results.aiScores[j][i];
				totalAIScore[j] += results.aiScores[j][i];
			}
			os << ") w/d/l: " << results.wins[i] << "/" << results.draws[i]
				<< "/" << results.losses[i] << "\n";
			poolPopScore += results.popScores[i];
			totalPopScore += results.popScores[i];
			poolScore += results.totalScores[i];
			totalScore += results.totalScores[i];
			poolWins += results.wins[i];
			totalWins += results.wins[i];
			poolDraws += results.draws[i];
			totalDraws += results.draws[i];
			poolLosses += results.losses[i];
			totalLosses += results.losses[i];
			if ((i + 1) % brainsPerPool == 0)
			{
				os << "Pool score: " << poolScore << " (" << poolPopScore;
				for (auto& score : poolAIScore)
				{
					os << "+" << score;
					score = 0;
				}
				os << ") w/d/l: " << poolWins << "/" << poolDraws << "/"
					<< poolLosses << "\n--------\n";
				poolPopScore = poolScore = poolWins = poolDraws = poolLosses =
					0;
			}
		}
		os << "Total score: " << totalScore << " (" << totalPopScore;
		for (auto& score : totalAIScore)
		{
			os << "+" << score;
		}
		os << ") w/d/l: " << totalWins << "/" << totalDraws << "/"
			<< totalLosses << std::endl;
		return os;
	}

private:
	struct Game
	{
		std::shared_ptr<AICommander> ai1, ai2;
		std::unique_ptr<Automaton> automaton;
		Phase phase;
		size_t turns;
		bool planning = false;
		bool ai1finished = false;
		bool ai2finished = false;
		bool done = false;
		GameResults results;
		virtual void update(RoundResults& round) const = 0;
	};
	struct PopGame : public Game
	{
		size_t idx1, idx2;
		void update(RoundResults& round) const override
			{ updatePopGame(*this, round); }
	};
	template <class T>
	struct AIGame : std::enable_if<any_is_same<T, Ts...>::value>, public Game
	{
		size_t idx;
		bool first;
		void update(RoundResults& round) const override
			{ updateAIGame(*this, round); }
	};

	static size_t brainsPerPool;

	std::unordered_map<std::string, Setting>& _settings;
	std::string _rulesetname;
	const std::vector<std::shared_ptr<NeuralNewtBrain>>& _brains;
	std::vector<std::unique_ptr<Game>> _games;

public:
	GameDirector(std::unordered_map<std::string, Setting>& settings,
		const std::string& rulesetname,
		const std::vector<std::shared_ptr<NeuralNewtBrain>>& brains);

private:
	static void updatePopGame(const PopGame& game, RoundResults& round);
	template <class T> static void updateAIGame(const AIGame<T>& game,
		RoundResults& round);

	void turn(std::unique_ptr<Game>& game);
	std::shared_ptr<AICommander> makeNNCommander(
		const std::shared_ptr<NeuralNewtBrain>& brain, size_t i,
		Json::Value& metadata);
	void setupPopGame(std::unique_ptr<PopGame>& game);
	template <class T>
		std::shared_ptr<AICommander> makeCommander(size_t i,
		Json::Value& metadata);
	template <class T>
		void setupAIGame(std::unique_ptr<AIGame<T>>& game);

public:
	void addPopGame(size_t brain1Idx, size_t brain2Idx);
	template <class T> void addAIGame(size_t brainIdx, bool first);

	RoundResults play();
};
