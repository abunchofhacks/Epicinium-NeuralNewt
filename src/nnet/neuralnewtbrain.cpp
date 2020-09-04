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

#include "neuralnewtbrain.hpp"

#include <regex>
#include <random>
#include <cmath>
#ifdef _MSC_VER
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#include <torch/torch.h>

#include "libs/aftermath/aicommander.hpp"
#include "libs/aftermath/position.hpp"
#include "libs/aftermath/board.hpp"
#include "libs/aftermath/tiletype.hpp"
#include "libs/aftermath/unittype.hpp"
#include "libs/aftermath/cell.hpp"

#include "setting.hpp"
#include "module.hpp"


enum BoardPlane : uint8_t
{
	P_TILETYPE,
	P_TILEOWNER,
	P_TILESTACKS,
	P_TILEPOWER,
	P_GROUNDTYPE,
	P_GROUNDOWNER,
	P_GROUNDSTACKS,
	P_AIRTYPE,
	P_AIROWNER,
	P_AIRSTACKS,
	P_HUMIDITY,
	P_CHAOS,
	P_GAS,
	P_SNOW,
	P_FROSTBITE,
	P_FIRESTORM,
	P_BONEDROUGHT,
	P_DEATH,
	P_VISION,
};

static constexpr size_t NUM_BOARDPLANES = ((size_t) P_VISION) + 1;

static constexpr size_t PLANESIZE = Position::MAX_ROWS * Position::MAX_COLS;

static constexpr int plix(BoardPlane plane, size_t offset)
{
	return ((int) plane) * PLANESIZE + offset;
}

#ifdef ORDERSENCODED
static constexpr size_t OLDORDERSCAP = 10;
static constexpr size_t NEWORDERSCAP = 5;
static constexpr size_t NUM_ORDERS = OLDORDERSCAP + NEWORDERSCAP;
static constexpr size_t PLANES_PER_ORDER = 4;
static constexpr size_t NUM_ORDERPLANES = NUM_ORDERS * PLANES_PER_ORDER;
#else
static constexpr size_t NUM_ORDERPLANES = 2;
#endif

static constexpr size_t NUM_MONEYPLANES = 10;
static constexpr size_t NUM_TIMEPLANES = 3;

const size_t NeuralNewtBrain::NUM_PLANES = NUM_BOARDPLANES
	+ NUM_ORDERPLANES + NUM_MONEYPLANES + NUM_TIMEPLANES;

static std::default_random_engine gen;

// We are not backpropagating, so no need for gradient calculation.
static torch::NoGradGuard no_grad;

// Modelled after the Fisher-Yates shuffle implementation as given in
// https://stackoverflow.com/a/9345144
std::vector<size_t> randomIndices(size_t size, size_t nRandom)
{
	std::vector<size_t> seq(size);
	auto begin = seq.begin();
	std::iota(begin, seq.end(), 0);
	size_t randomLeft = nRandom;
	while (randomLeft--)
	{
		auto r = begin;
		std::uniform_int_distribution<size_t> dis(0, size - 1);
		std::advance(r, dis(gen));
		std::swap(*begin, *r);
		++begin;
		--size;
	}
	return std::vector<size_t>(begin, begin + nRandom);
}

// Inspired by https://stackoverflow.com/a/7616783/8843413
float stdev(const torch::Tensor& t)
{
	std::vector<float> v(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());

	float sum = std::accumulate(v.begin(), v.end(), 0.0f);
	float mean = sum / v.size();

	std::vector<float> diff(v.size());
	std::transform(v.begin(), v.end(), diff.begin(),
		[mean](float x) { return x - mean; });
	float sq_sum =
		std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0f);
	return std::sqrt(sq_sum / v.size());
}

NeuralNewtBrain NeuralNewtBrain::mutate(const NeuralNewtBrain& brain,
	size_t round, float deviationFactor, float selectionChance)
{
	NeuralNewtBrain muBrain(brain,
		std::make_shared<MuBrainName>(brain._name, round));
	auto muPar = muBrain._module->parameters();
	for (size_t i = 0; i < muPar.size(); i++)
	{
		// Flatten the tensor, since for mutation, we are not interested in
		// altering the convolution kernels as a unit, as we are for
		// combination.
		torch::Tensor flat = muPar[i].flatten();

		float dev = stdev(flat.to(torch::kCPU, torch::kFloat));
		std::normal_distribution<float> dis(0.0f,
			(deviationFactor / sqrtf(round + 1)) * dev);

		// Alter half the parameters by adding a number that was sampled from a
		// normal distribution centered around 0.
		std::vector<size_t> indices =
			randomIndices(flat.size(0), selectionChance * flat.size(0));
		std::vector<float> mutations(flat.size(0), 0.0f);
		for (size_t j : indices)
		{
			mutations[j] = dis(gen);
		}
		torch::Tensor mutationsTensor = torch::tensor(mutations, torch::kFloat);
		if (flat.scalar_type() != torch::kFloat || flat.device() != torch::kCPU)
		{
			mutationsTensor =
				mutationsTensor.to(flat.device(), flat.scalar_type());
		}
		flat += mutationsTensor;
	}
	return muBrain;
}

std::pair<NeuralNewtBrain, NeuralNewtBrain> NeuralNewtBrain::combine(
	const NeuralNewtBrain& brain1, const NeuralNewtBrain& brain2, size_t round)
{
	NeuralNewtBrain coBrain1(brain1,
		std::make_shared<CoBrainName>(brain1._name, brain2._name, round));
	NeuralNewtBrain coBrain2(brain2,
		std::make_shared<CoBrainName>(brain2._name, brain1._name, round));
	std::vector<torch::Tensor> co1Par = coBrain1._module->parameters();
	std::vector<torch::Tensor> co2Par = coBrain2._module->parameters();
	const std::vector<torch::Tensor>& b1Par = brain1._module->parameters();
	const std::vector<torch::Tensor>& b2Par = brain2._module->parameters();
	for (size_t i = 0; i < co1Par.size(); i++)
	{
		// Copy randomly half the parameters from brain2, leaving the other half
		// as brain1.
		std::vector<size_t> indices =
			randomIndices(co1Par[i].size(0), co1Par[i].size(0) / 2);
		for (size_t j : indices)
		{
			co1Par[i][j] = b2Par[i][j];
			co2Par[i][j] = b1Par[i][j];
		}
	}
	return std::make_pair(std::move(coBrain1), std::move(coBrain2));
}

NeuralNewtBrain::NeuralNewtBrain(
		std::unordered_map<std::string, Setting>& settings,
		const BrainNamePtr& name) :
	_settings(settings),
	_module(new Module(settings)),
	_name(name)
{
	if (_settings["cuda"]) _module->to(torch::kCUDA, torch::kHalf);
	else _module->to(torch::kFloat);
}

NeuralNewtBrain::NeuralNewtBrain(
		const std::shared_ptr<Module>& module,
		std::unordered_map<std::string, Setting>& settings,
		const BrainNamePtr& name) :
	_settings(settings),
	_module(module),
	_name(name)
{
	if (_settings["cuda"]) _module->to(torch::kCUDA, torch::kHalf);
	else _module->to(torch::kFloat);
}

NeuralNewtBrain::NeuralNewtBrain(NeuralNewtBrain&& other) :
	NewtBrain(),
	_settings(other._settings),
	_module(std::move(other._module)),
	_name(std::move(other._name))
{}

NeuralNewtBrain::NeuralNewtBrain(const NeuralNewtBrain& brain,
		const BrainNamePtr& name) :
	_settings(brain._settings),
	_module(std::dynamic_pointer_cast<Module>(brain._module->clone())),
	_name(name)
{}

#ifdef ORDERSENCODED
static inline void encodeOrder(const Board& board,
	std::vector<int8_t>& data, size_t offset,
	const Order& order)
{
	DEBUG_ASSERT(offset + PLANESIZE <= data.size());
	std::fill(data.begin() + offset, data.begin() + offset + PLANESIZE, 0);
	if (order.type != Order::Type::NONE)
	{
		Position pos = order.subject.position;
		size_t i = pos.row * Position::MAX_COLS + pos.col;
		data[offset + i] = (int8_t) order.subject.type;
	}
	offset += PLANESIZE;

	DEBUG_ASSERT(offset + PLANESIZE <= data.size());
	switch (order.type)
	{
		case Order::Type::NONE:
		{
			// Fill with a positive value to differentiate from not giving
			// an order, because (int8_t) Order::Type::NONE == 0.
			std::fill(data.begin() + offset, data.begin() + offset + PLANESIZE,
				(int8_t) Order::TYPE_SIZE);
		}
		break;
		case Order::Type::MOVE:
		{
			std::fill(data.begin() + offset, data.begin() + offset + PLANESIZE,
				0);
			Cell current = board.cell(order.subject.position);
			for (const Move& move : order.moves)
			{
				current = current + move;
				// Convert to pos and then back to index, because the "stride"
				// should be MAX_COLS and not board._width.
				Position pos = current.pos();
				size_t i = pos.row * Position::MAX_COLS + pos.col;
				data[offset + i] = (int8_t) order.type;
			}
		}
		break;
		case Order::Type::GUARD:
		case Order::Type::FOCUS:
		case Order::Type::LOCKDOWN:
		case Order::Type::SHELL:
		case Order::Type::BOMBARD:
		case Order::Type::EXPAND:
		case Order::Type::PRODUCE:
		{
			std::fill(data.begin() + offset, data.begin() + offset + PLANESIZE,
				0);
			Position pos = order.target.position;
			size_t i = pos.row * Position::MAX_COLS + pos.col;
			data[offset + i] = (int8_t) order.type;
		}
		break;
		case Order::Type::BOMB:
		case Order::Type::CAPTURE:
		case Order::Type::SHAPE:
		case Order::Type::SETTLE:
		case Order::Type::UPGRADE:
		case Order::Type::CULTIVATE:
		case Order::Type::HALT:
		{
			std::fill(data.begin() + offset, data.begin() + offset + PLANESIZE,
				0);
			Position pos = order.subject.position;
			size_t i = pos.row * Position::MAX_COLS + pos.col;
			data[offset + i] = (int8_t) order.type;
		}
		break;
	}
	offset += PLANESIZE;

	DEBUG_ASSERT(offset + PLANESIZE <= data.size());
	switch (order.type)
	{
		case Order::Type::NONE:
		case Order::Type::MOVE:
		case Order::Type::GUARD:
		case Order::Type::FOCUS:
		case Order::Type::LOCKDOWN:
		case Order::Type::SHELL:
		case Order::Type::BOMBARD:
		case Order::Type::BOMB:
		case Order::Type::CAPTURE:
		case Order::Type::PRODUCE:
		case Order::Type::HALT:
		{
			std::fill(data.begin() + offset, data.begin() + offset + PLANESIZE,
				0);
		}
		break;
		case Order::Type::EXPAND:
		{
			std::fill(data.begin() + offset, data.begin() + offset + PLANESIZE,
				0);
			Position pos = order.target.position;
			size_t i = pos.row * Position::MAX_COLS + pos.col;
			data[offset + i] = (int8_t) order.tiletype;
		}
		break;
		case Order::Type::SHAPE:
		case Order::Type::SETTLE:
		case Order::Type::UPGRADE:
		{
			std::fill(data.begin() + offset, data.begin() + offset + PLANESIZE,
				0);
			Position pos = order.subject.position;
			size_t i = pos.row * Position::MAX_COLS + pos.col;
			data[offset + i] = (int8_t) order.tiletype;
		}
		break;
		case Order::Type::CULTIVATE:
		{
			std::fill(data.begin() + offset, data.begin() + offset + PLANESIZE,
				0);
			Cell center = board.cell(order.subject.position);
			for (Cell other : board.area(center, 1, 2))
			{
				// Convert to pos and then back to index, because the "stride"
				// should be MAX_COLS and not ai._board._width.
				Position pos = other.pos();
				size_t i = pos.row * Position::MAX_COLS + pos.col;
				data[offset + i] = (int8_t) order.tiletype;
			}
		}
		break;
	}
	offset += PLANESIZE;

	DEBUG_ASSERT(offset + PLANESIZE <= data.size());
	switch (order.type)
	{
		case Order::Type::NONE:
		case Order::Type::MOVE:
		case Order::Type::GUARD:
		case Order::Type::FOCUS:
		case Order::Type::LOCKDOWN:
		case Order::Type::SHELL:
		case Order::Type::BOMBARD:
		case Order::Type::BOMB:
		case Order::Type::CAPTURE:
		case Order::Type::EXPAND:
		case Order::Type::SHAPE:
		case Order::Type::SETTLE:
		case Order::Type::UPGRADE:
		case Order::Type::CULTIVATE:
		case Order::Type::HALT:
		{
			std::fill(data.begin() + offset, data.begin() + offset + PLANESIZE,
				0);
		}
		break;
		case Order::Type::PRODUCE:
		{
			std::fill(data.begin() + offset, data.begin() + offset + PLANESIZE,
				0);
			Position pos = order.target.position;
			size_t i = pos.row * Position::MAX_COLS + pos.col;
			data[offset + i] = (int8_t) order.unittype;
		}
		break;
	}
	offset += PLANESIZE;
}
#endif

std::vector<int8_t> NeuralNewtBrain::encode(const AICommander& ai)
{
	DEBUG_ASSERT(TILETYPE_SIZE < 128);
	DEBUG_ASSERT(UNITTYPE_SIZE < 128);
	DEBUG_ASSERT(PLAYER_SIZE < 128);

	std::vector<int8_t> data(NUM_PLANES * PLANESIZE);

	for (Cell index : ai._board)
	{
		Position pos = index.pos();
		DEBUG_ASSERT(pos.row >= 0 && pos.row <= Position::MAX_ROWS);
		DEBUG_ASSERT(pos.col >= 0 && pos.col <= Position::MAX_COLS);
		size_t i = pos.row * Position::MAX_COLS + pos.col;

		const TileToken& tile = ai._board.tile(index);
		data[plix(P_TILETYPE, i)] = (int8_t) tile.type;
		data[plix(P_TILEOWNER, i)] = (int8_t) ((tile.owner == ai._player)
			? Player::SELF : tile.owner);
		data[plix(P_TILESTACKS, i)] = tile.stacks;
		data[plix(P_TILEPOWER, i)] = tile.power;

		const UnitToken& ground = ai._board.ground(index);
		data[plix(P_GROUNDTYPE, i)] = (int8_t) ground.type;
		data[plix(P_GROUNDOWNER, i)] = (int8_t) ((ground.owner == ai._player)
			? Player::SELF : ground.owner);
		data[plix(P_GROUNDSTACKS, i)] = ground.stacks;

		const UnitToken& air = ai._board.air(index);
		data[plix(P_AIRTYPE, i)] = (int8_t) air.type;
		data[plix(P_AIROWNER, i)] = (int8_t) ((air.owner == ai._player)
			? Player::SELF : air.owner);
		data[plix(P_AIRSTACKS, i)] = air.stacks;

		DEBUG_ASSERT(ai._board.bypass(index).type == UnitType::NONE);

		DEBUG_ASSERT(ai._board.temperature(index) == 0);
		data[plix(P_HUMIDITY, i)] = ai._board.humidity(index);
		data[plix(P_CHAOS, i)] = ai._board.chaos(index);
		data[plix(P_GAS, i)] = ai._board.gas(index);
		DEBUG_ASSERT(ai._board.radiation(index) == 0);

		data[plix(P_SNOW, i)] = ai._board.snow(index);
		data[plix(P_FROSTBITE, i)] = ai._board.frostbite(index);
		data[plix(P_FIRESTORM, i)] = ai._board.firestorm(index);
		data[plix(P_BONEDROUGHT, i)] = ai._board.bonedrought(index);
		data[plix(P_DEATH, i)] = ai._board.death(index);

		data[plix(P_VISION, i)] = ai._board.current(index);
	}

	size_t i = NUM_BOARDPLANES * PLANESIZE;

#ifdef ORDERSENCODED
	// The cap on old orders is not enforced by the Automaton, and it is not
	// an error if this occurs in release. But we think this will not occur in
	// real games, so we use a debug assertion to confirm that suspicion.
	DEBUG_ASSERT(ai._unfinishedOrders.size() <= OLDORDERSCAP);
	// The cap on new orders is relatively save because we will not change
	// Bible::newOrderLimit() before release; it is not an error if this
	// occurs in release as we will simply ignore further orders.
	DEBUG_ASSERT(ai._newOrders.size() <= NEWORDERSCAP);

	int ordernum = 0 - ((int) OLDORDERSCAP);

	if (ai._unfinishedOrders.size() < OLDORDERSCAP)
	{
		size_t n = OLDORDERSCAP - ai._unfinishedOrders.size();
		size_t len = n * PLANES_PER_ORDER * PLANESIZE;
		std::fill(data.begin() + i, data.begin() + i + len, 0);
		i += len;
		ordernum += n;
	}

	for (const Order& order : ai._unfinishedOrders)
	{
		encodeOrder(ai._board, data, i, order);
		i += PLANES_PER_ORDER * PLANESIZE;
		ordernum += 1;
		if (ordernum >= 0) break;
	}

	DEBUG_ASSERT(ordernum == 0);

	for (const Order& order : ai._newOrders)
	{
		encodeOrder(ai._board, data, i, order);
		i += PLANES_PER_ORDER * PLANESIZE;
		ordernum += 1;
		if ((size_t) ordernum >= NEWORDERSCAP) break;
	}

	if ((size_t) ordernum < NEWORDERSCAP)
	{
		size_t n = NEWORDERSCAP - ordernum;
		size_t len = n * PLANES_PER_ORDER * PLANESIZE;
		std::fill(data.begin() + i, data.begin() + i + len, 0);
		i += len;
		ordernum += n;
	}

	DEBUG_ASSERT(ordernum == NEWORDERSCAP);
#else
	// For now we only store the number of old and new orders.
	// The cap on old orders is not enforced by the Automaton, and it is not
	// an error if this occurs in release. But we think this will not occur in
	// real games, so we use a debug assertion to confirm that suspicion.
	DEBUG_ASSERT(ai._unfinishedOrders.size() < 128);
	DEBUG_ASSERT(i + PLANESIZE <= data.size());
	std::fill(data.begin() + i, data.begin() + i + PLANESIZE,
		(int8_t) std::min(ai._unfinishedOrders.size(), (size_t) 127));
	i += PLANESIZE;
	DEBUG_ASSERT(ai._newOrders.size() < 128);
	DEBUG_ASSERT(i + PLANESIZE <= data.size());
	std::fill(data.begin() + i, data.begin() + i + PLANESIZE,
		(int8_t) std::min(ai._newOrders.size(), (size_t) 127));
	i += PLANESIZE;
#endif

	// It is probably best to store the money continuously, so we have it spill
	// over into "buckets". So 100 is stored as (100, 0, 0, ...), 101 is stored
	// as (100, 1, 0, ...) and e.g. 274 is stored as (100, 100, 74, 0, ...).
	DEBUG_ASSERT(ai._money >= 0);
	for (int offset = 0; offset < 1000; offset += 100)
	{
		DEBUG_ASSERT(i + PLANESIZE <= data.size());
		std::fill(data.begin() + i, data.begin() + i + PLANESIZE,
			(int8_t) std::min(std::max(0, ai._money - offset), 100));
		i += PLANESIZE;
	}

	// Everything past year 100 is "extreme lategame" anyway.
	DEBUG_ASSERT(ai._year >= 0);
	DEBUG_ASSERT(i + PLANESIZE <= data.size());
	std::fill(data.begin() + i, data.begin() + i + PLANESIZE,
		(int8_t) std::min(std::max(0, ai._year), 100));
	i += PLANESIZE;

	DEBUG_ASSERT(SEASON_SIZE < 128);
	DEBUG_ASSERT(i + PLANESIZE <= data.size());
	std::fill(data.begin() + i, data.begin() + i + PLANESIZE,
		(int8_t) ai._season);
	i += PLANESIZE;

	DEBUG_ASSERT(DAYTIME_SIZE < 128);
	DEBUG_ASSERT(i + PLANESIZE <= data.size());
	std::fill(data.begin() + i, data.begin() + i + PLANESIZE,
		(int8_t) ai._daytime);
	i += PLANESIZE;

	// The phase is always PLANNING.

	DEBUG_ASSERT(i == data.size());

	return data;
}

void NeuralNewtBrain::prepare(const AICommander& ai)
{
	std::vector<int8_t> data = encode(ai);
	_input.insert(_input.end(), data.begin(), data.end());
	_count++;
}

NewtBrain::Output NeuralNewtBrain::evaluate()
{
	DEBUG_ASSERT(_count > 0);

	// Do we still need to generate the output?
	if (_output.size() == 0)
	{
		std::chrono::high_resolution_clock::time_point start;
		static bool timing = _settings["timing"];
		static float ds = 0.0f;
		static size_t evals = 0;
		static size_t counts = 0;
		if (timing) start = std::chrono::high_resolution_clock::now();

		// Generate all the output at once with the NN.
		torch::Tensor dataTensor = torch::from_blob(
			&_input[0],
			{
				long(_count),
				long(NUM_PLANES),
				long(Position::MAX_COLS),
				long(Position::MAX_ROWS),
			},
			torch::kInt8
		).clone().to(_settings["cuda"] ? torch::kHalf : torch::kFloat);
		_input.clear();
		if (_settings["cuda"]) dataTensor = dataTensor.contiguous().cuda();

		torch::Tensor resultTensor = _module->forward(dataTensor);

		resultTensor = resultTensor.to(torch::kCPU, torch::kFloat);
		for (size_t i = 0; i < _count; i++)
		{
			_output.emplace();
			std::vector<float> output(
				resultTensor.data_ptr<float>() + i * NewtBrain::Output::SIZE,
				resultTensor.data_ptr<float>()
					+ (i + 1) * NewtBrain::Output::SIZE
			);
			_output.back().assign(output);
		}

		if (timing)
		{
			auto end = std::chrono::high_resolution_clock::now();
			ds += std::chrono::duration_cast<std::chrono::microseconds>
				(end - start).count() / 1000.0f;
			evals++;
			counts += _count;
			if (evals % 100 == 0)
			{
				std::cout << "NeuralNewtBrain evaluations averaged "
					<< (ds / 100) << "ms (" << (ds / counts) << "ms per output)"
					<< std::endl;
				ds = 0.0f;
				evals = 0;
				counts = 0;
			}
		}
	}

	// We have already generated all the output, return the first.
	DEBUG_ASSERT(_count == _output.size());
	Output output = _output.front();
	_output.pop();
	_count--;
	return output;
}

// Source:
// https://github.com/Kolkir/mlcpp/blob/master/mask_rcnn_pytorch/stateloader.cpp
bool is_empty(at::Tensor x)
{
	if (x.defined() && x.dim() > 0 && x.size(0) != 0 && x.numel() > 0)
		return false;
	else return true;
}
void save_state_dict(const torch::nn::Module& module,
	const std::string& filename)
{
	torch::serialize::OutputArchive archive;
	auto params = module.named_parameters(true /*recurse*/);
	auto buffers = module.named_buffers(true /*recurse*/);
	for (const auto& val : params)
	{
		//if (!is_empty(val.value()))
		{
			archive.write(val.key(), val.value());
		}
	}
	for (const auto& val : buffers)
	{
		//if (!is_empty(val.value()))
		{
			archive.write(val.key(), val.value(), /*is_buffer*/ true);
		}
	}
	archive.save_to(filename);
}
void load_state_dict(torch::nn::Module& module, const std::string& filename,
	const std::string& ignore_name_regex = "")
{
	torch::serialize::InputArchive archive;
	archive.load_from(filename, torch::kCPU);
	torch::NoGradGuard no_grad;
	std::regex re(ignore_name_regex);
	std::smatch m;
	auto params = module.named_parameters(true /*recurse*/);
	auto buffers = module.named_buffers(true /*recurse*/);
	bool typeDiscrepancyDetected = false;
	for (auto& val : params)
	{
		if (!std::regex_match(val.key(), m, re))
		{
			try
			{
				archive.read(val.key(), val.value());
			}
			catch (const torch::Error&)
			{
				typeDiscrepancyDetected = true;
				module.to(torch::kHalf);
				archive.read(val.key(), val.value());
			}
		}
	}
	for (auto& val : buffers)
	{
		if (!std::regex_match(val.key(), m, re))
		{
			archive.read(val.key(), val.value(), /*is_buffer*/ true);
		}
	}
	if (typeDiscrepancyDetected)
	{
		module.to(torch::kFloat);
	}
}

bool NeuralNewtBrain::save(const std::string& folder,
	const std::string& filename)
{
	std::string filepath = folder + "/" + filename;
	struct stat buffer;
	if (stat(folder.c_str(), &buffer) != 0)
	{
		std::cout << "Checkpoint Directory does not exist! Making directory "
			<< folder << std::endl;
		mkdir(
			folder.c_str()
#ifdef __unix__
			, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH
#endif
		);
	}
	if (stat(filepath.c_str(), &buffer) != 0)
	{
		save_state_dict(*_module, filepath);
		return true;
	}
	return false;
}

void NeuralNewtBrain::load(const std::string& folder,
	const std::string& filename)
{
	// https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
	std::string filepath = folder + "/" + filename;
	struct stat buffer;
	if (stat(filepath.c_str(), &buffer) != 0)
	{
		throw std::runtime_error("No model in path " + filepath);
	}
	load_state_dict(*_module, filepath);
	if (_settings["cuda"]) _module->to(torch::kCUDA, torch::kHalf);
	else _module->to(torch::kFloat);
}
