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

#include "brainname.hpp"

#include <cassert>

#include "libs/openssl/sha.h"


static const size_t MEDIUM_NAME_DEPTH = 3;

BrainName::BrainName(const std::string& mediumName,
		const std::string& shortName, size_t round) :
	_mediumName(mediumName),
	_shortName(shortName),
	_round(round)
{}

std::string sha256(const std::string& data)
{
	uint8_t digest[SHA256_DIGEST_LENGTH];

	SHA256_CTX ctx;
	SHA256_Init(&ctx);

	SHA256_Update(&ctx, data.data(), data.size());

	SHA256_Final(digest, &ctx);

	char buffer[SHA256_DIGEST_LENGTH * 2 + 1];
	for (size_t i = 0; i < SHA256_DIGEST_LENGTH; i++)
	{
		sprintf(buffer + i * 2, "%02x", digest[i]);
	}
	return std::string(buffer);
}

SeedBrainName::SeedBrainName(size_t n) :
	BrainName("s" + std::to_string(n), _mediumName, 0)
{}

std::string SeedBrainName::shortenedName(size_t) const
{
	return shortName();
}

std::string SeedBrainName::longName() const
{
	return shortName();
}

std::string muBrainNameShortenedName(size_t depth,
	const std::string& parentShort, size_t round)
{
	assert(depth > 0);
	return "m" + std::to_string(round) + "(" + parentShort + ")";
}

std::string muBrainNameLongName(const std::string& parentLong, size_t round)
{
	return "m" + std::to_string(round) + "(" + parentLong + ")";
}

MuBrainName::MuBrainName(const BrainNamePtr& parent, size_t round) :
	BrainName(
		muBrainNameShortenedName(MEDIUM_NAME_DEPTH,
			parent->shortenedName(MEDIUM_NAME_DEPTH - 1), round),
		sha256("m" + std::to_string(round) + "(" + parent->shortName() + ")"),
		round
	),
	_parent(parent)
{}

std::string MuBrainName::shortenedName(size_t depth) const
{
	if (depth == 0) return shortName().substr(0, 8);
	return muBrainNameShortenedName(depth, _parent->shortenedName(depth - 1),
		_round);
}

std::string MuBrainName::longName() const
{
	return muBrainNameLongName(_parent->longName(), _round);
}

std::string coBrainNameShortenedName(size_t depth,
	const std::string& parent1Short, const std::string& parent2Short,
	size_t round)
{
	assert(depth > 0);
	return "c" + std::to_string(round) + "(" + parent1Short + "," + parent2Short
		+ ")";
}

std::string coBrainNameLongName(const std::string& parent1Long,
	const std::string& parent2Long, size_t round)
{
	return "c" + std::to_string(round) + "(" + parent1Long + "," + parent2Long
		+ ")";
}

CoBrainName::CoBrainName(const BrainNamePtr& parent1,
		const BrainNamePtr& parent2, size_t round):
	BrainName(
		coBrainNameShortenedName(MEDIUM_NAME_DEPTH,
			parent1->shortenedName(MEDIUM_NAME_DEPTH - 1),
			parent2->shortenedName(MEDIUM_NAME_DEPTH - 1), round),
		sha256("c" + std::to_string(round) + "(" + parent1->shortName() + ","
			+ parent2->shortName() + ")"),
		round
	),
	_parent1(parent1),
	_parent2(parent2)
{}

std::string CoBrainName::shortenedName(size_t depth) const
{
	if (depth == 0) return shortName().substr(0, 8);
	return coBrainNameShortenedName(depth, _parent1->shortenedName(depth - 1),
		_parent2->shortenedName(depth - 1), _round);
}

std::string CoBrainName::longName() const
{
	return coBrainNameLongName(_parent1->longName(), _parent2->longName(),
		_round);
}

RestoredBrainName::RestoredBrainName(const std::string& name, size_t round) :
	BrainName(name.substr(0, 8), name, round)
{}

std::string RestoredBrainName::shortenedName(size_t depth) const
{
	if (depth < MEDIUM_NAME_DEPTH) return shortName().substr(0, 8);
	return mediumName();
}

std::string RestoredBrainName::longName() const
{
	return mediumName();
}
